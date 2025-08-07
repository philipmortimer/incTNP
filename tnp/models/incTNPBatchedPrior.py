#incTNP with batching strategy explored. This variant supports a start token (allowing for conditioning on empty context)
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TNPTransformerFullyMaskedEncoder
from ..utils.helpers import preprocess_observations
from .base import BatchedCausalTNPPrior
from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations
from ..networks.kv_cache import init_kv_cache
from ..networks.kv_cache_fixed import init_kv_cache_fixed
from .incUpdateBase import IncUpdateEff, IncUpdateEffFixed
import torch.distributions as td


class IncTNPBatchedEncoderPrior(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerFullyMaskedEncoder],
        xy_encoder: nn.Module,
        embed_dim: int, # This is dz and is used for the learnable empty token
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        # Learnable empty token used to represent start / no context
        self.empty_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.empty_token, mean=0.0, std=0.02)

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]",
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]",
        "return: [m, n_t_or_n_minus_one, dz]",
    )
    def forward(
        self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None,
        xc: Optional[torch.Tensor] = None, yc: Optional[torch.Tensor] = None, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Checks that it either provides (x,y) OR (xc, yc, xt) but not both. This is used to determine whether train / prediction is happening
        assert (xc is None and yc is None and xt is None and y is not None and x is not None) or (xc is not None and yc is not None and xt is not None and x is None and y is None), "Invalid encoder call. Can't differentiate between prediction or training call"

        if x is not None and y is not None: return self.train_encoder(x, y)
        else: return self.predict_encoder(xc, yc , xt)

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "return: [m, nc, dz]"
    )
    def _preprocess_context(self, xc: torch.Tensor, yc:torch.Tensor):
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1) # Adds flag
        xc_encoded = self.x_encoder(xc)
        yc_encoded = self.y_encoder(yc)
        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        return self.xy_encoder(zc)

    @check_shapes(
        "xt: [m, nt, dx]", "return: [m, nt, dz]"
    )
    def _preprocess_targets(self, xt: torch.Tensor, dy: int):
        m, nt, _ = xt.shape
        # Creates yt of zeros plus a bool flag
        yt = torch.zeros(m, nt, dy).to(xt)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)
        # Encodes
        xt_encoded = self.x_encoder(xt)
        yt_encoded = self.y_encoder(yt)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        return self.xy_encoder(zt)


    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def predict_encoder(self, xc: torch.Tensor, yc:torch.Tensor, xt:torch.Tensor):
        # At prediction time we essentially become identically to incTNP basic
        # (I.e.) just self attention over the context points and no cross attention mask.
        m, nc, _ = xc.shape
        zc = self._preprocess_context(xc, yc)
        zt = self._preprocess_targets(xt, yc.shape[2])

        # Adds dummy start token to zc
        start_token = self.empty_token.expand(m, -1, -1)
        zc = torch.cat((start_token, zc), dim=1)

        # Causal Masking
        #mask_sa = torch.tril(torch.ones(nc+1, nc+1, dtype=torch.bool, device=zc.device), diagonal=0)
        #mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n + 1, n + 1]

        zt = self.transformer_encoder(zc, zt, mask_ca=None, mask_sa=None, use_causal=True)
        
        return zt

    # Incrementally updates context using kv caching. Essentially just the SA branch.
    @check_shapes(
        "zc_new: [m, nc_new, dz]", "mask_sa_big: [m, nc_max, nc_max]"
    )
    def update_context(self, zc_new: torch.Tensor, kv_cache: dict, mask_sa_big: Optional[torch.Tensor] = None, use_flash: bool=False) -> torch.Tensor:
        zc_updated = self.transformer_encoder.encode_context(zc_new, kv_cache=kv_cache, use_flash=use_flash)

    # Once the context has been conditioned on, this is used to run predictions. Essentially just the CA branch.
    @check_shapes(
        "zt: [m, nt, dz]", "return: [m, nt, dz]"
    )
    def query(self, zt: torch.Tensor, kv_cache: dict, use_flash: bool=False) -> torch.Tensor:
        return self.transformer_encoder.query(zt, kv_cache, use_flash=use_flash)

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]","return: [m, n, dz]"
    )
    def train_encoder(self, x: torch.Tensor, y:torch.Tensor):
        m, n, dy = y.shape
        # Treats sequence as just x and y. y_tgt is set to just be 0s to
        y_like = torch.zeros((m, n, dy)).to(y)
        y_tgt = torch.cat((y_like, torch.ones(y_like.shape[:-1] + (1,)).to(y)), dim=-1)

        y_ctx = torch.cat((y, torch.zeros(y.shape[:-1] + (1,)).to(y)), dim=-1)

        # Encodes x and y
        x_encoded = self.x_encoder(x)
        x_tgt_encoded = x_encoded
        y_ctx_encoded = self.y_encoder(y_ctx)
        y_tgt_encoded = self.y_encoder(y_tgt)

        # Embeds data
        zc = torch.cat((x_encoded, y_ctx_encoded), dim=-1)
        zt = torch.cat((x_tgt_encoded, y_tgt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Adds dummy start token to zc
        start_token = self.empty_token.expand(m, -1, -1)
        zc = torch.cat((start_token, zc), dim=1)

        # Creates masks. 
        # A target point can only attend to preceding context points (plus dummy token)
        mask_ca = torch.tril(torch.ones(n, n + 1, dtype=torch.bool, device=zc.device), diagonal=0)
        #mask_ca = mask_ca.unsqueeze(0).expand(m, -1, -1) # [m, n, n + 1]
        # Causal masking for context -> a context point can only attend to itself and previous context points (including dummy token).
        #mask_sa = torch.tril(torch.ones(n + 1, n + 1, dtype=torch.bool, device=zc.device), diagonal=0)
        #mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n + 1, n + 1]

        zt = self.transformer_encoder(zc, zt, mask_sa=None, use_causal=True, mask_ca=mask_ca)
        
        assert len(zt.shape) == 3 and zt.shape[0] == m and zt.shape[1] == n, "Return encoder shape wrong"
        return zt


class IncTNPBatchedPrior(BatchedCausalTNPPrior, IncUpdateEff, IncUpdateEffFixed):
    def __init__(
        self,
        encoder: IncTNPBatchedEncoderPrior,
        decoder: TNPDecoder,
        likelihood: nn.Module,
        order_ctx_greedy: str="False" # Use context orderin false by default
    ):
        super().__init__(encoder, decoder, likelihood)
        self.order_ctx_greedy = order_ctx_greedy


    # Logic for effecient incremental context updates
    def init_inc_structs_fixed(self, m: int, max_nc: int, xt:torch.Tensor, dy: int, device: str, use_flash: bool=False):
        # Adds empty token
        start_token = self.encoder.empty_token.expand(m, -1, -1)
        dz = start_token.shape[2]
        layers = len(self.encoder.transformer_encoder.mhsa_layers)
        heads = self.encoder.transformer_encoder.mhsa_layers[0].attn.num_heads
        head_dim = int(round(self.encoder.transformer_encoder.mhsa_layers[0].attn.scale ** -2))
        self.kv_cache_inc = init_kv_cache_fixed(layers=layers, batch_size=m, max_nc=max_nc+1, dz=dz, 
            heads=heads, k_dim=head_dim, v_dim=head_dim, device=device)
        self.encoder.transformer_encoder.encode_context_fixedkv(start_token, self.kv_cache_inc, use_flash=use_flash)
        # Caches target points
        self.target_encoding_cache_zt = self.encoder._preprocess_targets(xt, dy) # [m, nt, dz]

    # Adds new context points
    def update_ctx_fixed(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False):
        zc = self.encoder._preprocess_context(xc, yc)
        self.encoder.transformer_encoder.encode_context_fixedkv(zc, self.kv_cache_inc, use_flash=use_flash)

    def query_fixed(self, tgt_start_ind: int, tgt_end_ind: int, use_flash: bool=False) -> td.Normal:
        # xt shape must [m, nt, dx] -> only uses nt value tho so can be junk or truncated before hand
        zt = self.target_encoding_cache_zt[:, tgt_start_ind:tgt_end_ind, :]
        dec = self.decoder(self.encoder.transformer_encoder.query_fixedkv(zt, self.kv_cache_inc, use_flash=use_flash), zt)
        if use_flash: # torch 16 combined with random gen (not from actual dataset can lead to some nans. instead of fixing with proper data loader this hacky sol is quick)
            prior_noise = self.likelihood.min_noise
            dec = torch.nan_to_num(dec)
            #self.likelihood.min_noise += 1e-2
        #print(dec)
        dist = self.likelihood(dec)

        if use_flash:
            self.likelihood.min_noise = prior_noise
        return dist

    # Logic for effecient incremental context updates
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool=False):
        self.kv_cache_inc = init_kv_cache()
        # Adds empty token
        start_token = self.encoder.empty_token.expand(m, -1, -1)
        self.encoder.update_context(start_token, self.kv_cache_inc,use_flash=use_flash)


    # Adds new context points
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False):
        zc = self.encoder._preprocess_context(xc, yc)
        self.encoder.update_context(zc, self.kv_cache_inc,use_flash=use_flash)

    def query(self, xt: torch.Tensor, dy: int, use_flash: bool=False) -> td.Normal:
        zt = self.encoder._preprocess_targets(xt, dy)
        return self.likelihood(self.decoder(self.encoder.query(zt, self.kv_cache_inc, use_flash=use_flash), xt))

    # Greedy Context ordering algorithm using KV caching. Note it is incremental (so given an initial context set + a number of new ctx points it can pick the best order)
    # This approach assumes we already have all context points.
    @check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]"
    )
    @torch.no_grad()
    def kv_cached_greedy_variance_ctx_builder(self, xc: torch.Tensor, yc: torch.Tensor, policy: str = "best", select="logp"):
        assert policy in {"best", "worst", "median"}, "Invalid policy"
        assert select in {"logp", "var"}, "Invalid selection criteria"
        device = xc.device
        # When deciding the context set ordering, start with an empty context set and build up greedily (or according to some shallow strategy)
        _, _, dx = xc.shape
        m, nc, dy = yc.shape

        # Tracks which context points have been picked
        picked_mask = torch.zeros(m, nc, dtype=torch.bool, device=device) # Stores whether a context point has been picked

        start_token = self.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        dz = start_token.shape[2]

        kv_cache = init_kv_cache()
        self.encoder.update_context(start_token, kv_cache)

        # Incrementally builds up representation
        batch_idx = torch.arange(m, device=device)
        ordered_indices = torch.full((m, nc), -1, dtype=torch.long, device=device)
        for step in range(nc):
            # Gathers all unpicked points into a call (vectorised / batch together for speed)
            not_picked_idx = (~picked_mask).nonzero(as_tuple=False)
            n_remaining = nc - step
            idx_remaining = not_picked_idx[:, 1].reshape(m, n_remaining) # [m, n_remaining]
            xt_candidates = xc[batch_idx.unsqueeze(1), idx_remaining] # [m, n_remaining, dx]
            yt_candidates = yc[batch_idx.unsqueeze(1), idx_remaining]

            # Query - performs inference on already conditioned context
            zt_candidates = self.encoder._preprocess_targets(xt_candidates, dy)
            pred_dist = self.likelihood(self.decoder(self.encoder.query(zt_candidates, kv_cache), xt_candidates)) # [m, n_remaining, dy] 
            
            if select == "logp": metric = (pred_dist.log_prob(yt_candidates).sum(dim=(-1)) / (n_remaining * dy))
            elif select == "var": metric = pred_dist.variance.mean(-1)

            # Selection strategy
            if policy == "best": selected_points = metric.argmax(dim=1) # [m]
            elif policy == "worst": selected_points = metric.argmin(dim=1) # [m]
            else: selected_points = metric.kthvalue(k=(n_remaining // 2) + 1, dim=1)[1] # [m] - median

            # Selects points per batch
            selected_points_global = idx_remaining[batch_idx, selected_points] # [m]
            picked_mask[batch_idx, selected_points_global] = True # Shows that point has been picked
            ordered_indices[:, step] = selected_points_global

            # Updates context representation
            added_xc = xc[batch_idx, selected_points_global].unsqueeze(1)
            added_yc = yc[batch_idx, selected_points_global].unsqueeze(1)
            new_zc = self.encoder._preprocess_context(added_xc, added_yc)
            zc = self.encoder.update_context(new_zc, kv_cache)
        xc_new = xc[batch_idx.unsqueeze(1), ordered_indices]
        yc_new = yc[batch_idx.unsqueeze(1), ordered_indices]

        return xc_new, yc_new


    # Greedy Context ordering algorithm. Note it is incremental (so given an initial context set + a number of new ctx points it can pick the best order)
    # This approach assumes we already have all context points.
    @check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]"
    )
    @torch.no_grad()
    def greedy_variance_ctx_builder(self, xc: torch.Tensor, yc: torch.Tensor, policy: str = "best"):
        assert policy in {"best", "worst", "median"}, "Invalid policy"
        device = xc.device
        # When deciding the context set ordering, start with an empty context set and build up greedily (or according to some shallow strategy)
        _, _, dx = xc.shape
        m, nc, dy = yc.shape

        # Tracks which context points have been picked
        picked_mask = torch.zeros(m, nc, dtype=torch.bool, device=device) # Stores whether a context point has been picked

        # Incrementally builds up representation
        batch_idx = torch.arange(m, device=device)
        xc_new, yc_new = torch.empty((m, 0, dx), device=xc.device), torch.empty((m, 0, dy), device=yc.device)
        for step in range(nc):
            # Gathers all unpicked points into a call (vectorised / batch together for speed)
            not_picked_idx = (~picked_mask).nonzero(as_tuple=False)
            n_remaining = nc - step
            idx_remaining = not_picked_idx[:, 1].reshape(m, n_remaining) # [m, n_remaining]
            xt_candidates = xc[batch_idx.unsqueeze(1), idx_remaining] # [m, n_remaining, dx]
            yt_candidates = yc[batch_idx.unsqueeze(1), idx_remaining]

            # Runs inference
            pred_dist = self.likelihood(self.decoder(self.encoder(xc=xc_new, yc=yc_new, xt=xt_candidates), xt_candidates)) # [m, n_remaining, dy] 
            #var = pred_dist.variance.mean(-1) # [m, n_remaining]
            var = (pred_dist.log_prob(yt_candidates).sum(dim=(-1)) / (n_remaining * dy))


            # Selection strategy
            if policy == "best": selected_points = var.argmax(dim=1) # [m]
            elif policy == "worst": selected_points = var.argmin(dim=1) # [m]
            else: selected_points = var.kthvalue(k=(n_remaining // 2) + 1, dim=1)[1] # [m] - median

            # Selects points per batch
            selected_points_global = idx_remaining[batch_idx, selected_points] # [m]

            # Updates context representation
            added_xc = xc[batch_idx, selected_points_global].unsqueeze(1)
            added_yc = yc[batch_idx, selected_points_global].unsqueeze(1)
            xc_new = torch.cat([xc_new, added_xc], dim=1)
            yc_new = torch.cat([yc_new, added_yc], dim=1)
            picked_mask[batch_idx, selected_points_global] = True # Shows that point has been picked
        return xc_new, yc_new