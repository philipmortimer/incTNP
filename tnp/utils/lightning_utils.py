import dataclasses
import time
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn

from ..data.base import Batch
from ..data.hadISD import HadISDBatch
from ..data.hadISD import get_true_temp, scale_pred_temp_dist
from .np_functions import np_loss_fn, np_pred_fn


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: Callable = np_loss_fn,
        pred_fn: Callable = np_pred_fn,
        plot_fn: Optional[Callable] = None,
        plot_interval: int = 1,
    ):
        super().__init__()

        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.plot_fn = plot_fn
        self.plot_interval = plot_interval

        # Keep these for plotting.
        self.val_batches: List[Batch] = []

        # Keep these for analysing.
        self.test_outputs: List[Any] = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        loss = self.loss_fn(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        if batch_idx < 5:
            # Only keep first 5 batches for logging.
            self.val_batches.append(batch)

        pred_dist = self.pred_fn(self.model, batch)

        # Compute metrics to track.
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu().mean()

        self.log("val/loglik", loglik, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            self.log(
                "val/gt_loglik", gt_loglik, on_step=False, on_epoch=True, prog_bar=True
            )

        # Handles temperature prediction case - distribution must be normal but is for the models we use.
        if isinstance(batch, HadISDBatch):
            # Reconstructs mean and variance vectors to be the correct units
            pred_dist_temp = scale_pred_temp_dist(batch, pred_dist)
            yt_correct_units = get_true_temp(batch, batch.yt)

            # Computes track statistics
            loglik_temp = pred_dist_temp.log_prob(yt_correct_units).sum() / yt_correct_units[..., 0].numel()
            rmse_temp = nn.functional.mse_loss(pred_dist_temp.mean, yt_correct_units).sqrt().cpu().mean() 

            self.log("val/loglik_temp", loglik_temp, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/rmse_temp", rmse_temp, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {}
        pred_dist = self.pred_fn(self.model, batch)

        # Compute metrics to track.
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        result["loglik"] = loglik.cpu()
        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()
        result["rmse"] = rmse

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            result["gt_loglik"] = gt_loglik.cpu()


        # Handles temperature prediction case - distribution must be normal but is for the models we use.
        if isinstance(batch, HadISDBatch):
            # Reconstructs mean and variance vectors to be the correct units
            pred_dist_temp = scale_pred_temp_dist(batch, pred_dist)
            yt_correct_units = get_true_temp(batch, batch.yt)

            # Computes track statistics
            loglik_temp = pred_dist_temp.log_prob(yt_correct_units).sum() / yt_correct_units[..., 0].numel()
            rmse_temp = nn.functional.mse_loss(pred_dist_temp.mean, yt_correct_units).sqrt().cpu().mean() 

            result["loglik_temp"] = loglik_temp
            result["rmse_temp"] = rmse_temp

        self.test_outputs.append(result)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_batches) == 0:
            return

        if self.plot_fn is not None and self.current_epoch % self.plot_interval == 0:
            self.plot_fn(
                self.model, self.val_batches, f"epoch-{self.current_epoch:04d}"
            )

        self.val_batches = []

    def configure_optimizers(self):
        return self.optimiser


class LogPerformanceCallback(pl.Callback):

    def __init__(self):
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.forward_start_time = 0.0
        self.between_step_time = 0.0

    @pl.utilities.rank_zero_only
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "performance/between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
        )
        self.forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.forward_start_time
        pl_module.log(
            "performance/forward_time",
            forward_time,
            on_step=True,
            on_epoch=False,
        )
        self.backward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_after_backward(trainer, pl_module)
        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "performance/backward_time",
            backward_time,
            on_step=True,
            on_epoch=False,
        )

    @pl.utilities.rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.update_count += 1

        # Calculate total elapsed time
        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        # Calculate updates per second
        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        # Log updates per second to wandb using pl_module.log
        pl_module.log(
            "performance/average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            "performance/last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        self.between_step_time = time.time()


def _batch_to_cpu(batch: Batch):
    batch_kwargs = {
        field.name: (
            getattr(batch, field.name).cpu()
            if isinstance(getattr(batch, field.name), torch.Tensor)
            else getattr(batch, field.name)
        )
        for field in dataclasses.fields(batch)
    }
    return type(batch)(**batch_kwargs)
