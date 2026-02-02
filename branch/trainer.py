# -*- coding: utf-8 -*-
"""Branch-only trainers.

Goal
----
Train/evaluate a *single* branch without instantiating the coupled STAMP pipeline:
- predictor-only: LLM predictor or Mamba predictor
- recon-only: Mamba reconstruction

These trainers intentionally reuse:
- dataloaders from `lib/`
- models from `model/`
- loss from `lib.metrics.masked_mse_loss`

Checkpoints
-----------
We save branch-only checkpoints that only contain the relevant state_dict, to avoid
confusion with the coupled Trainer checkpoint format.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from lib.logger import get_logger


def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = False, disable: bool = False):
    # local import to avoid hard dep when tqdm not installed (but it's in this repo)
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True, disable=disable)


class PredOnlyTrainer:
    """Train predictor branch only.

    Model interface expected:
        pred_model(x, mas=None) -> y_hat, where
          x:   [B, window_size-n_pred, N, C]
          y:   [B, n_pred, N, C]
    """

    def __init__(
        self,
        pred_model,
        pred_loss,
        pred_optimizer,
        train_loader,
        val_loader,
        test_loader,
        args,
        scaler,
        logger=None,
    ):
        self.pred_model = pred_model
        self.pred_loss = pred_loss
        self.pred_optimizer = pred_optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler

        ckpt_dir = getattr(self.args, "log_dir_pth", None) or self.args.log_dir
        log_dir = getattr(self.args, "log_dir_log", None) or self.args.log_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.best_path = os.path.join(ckpt_dir, f"best_pred_only_{self.args.data}_{self.args.model}.pth")
        self.logger = logger or get_logger(
            log_dir,
            name=self.args.model,
            debug=getattr(self.args, "debug", False),
            data=self.args.data,
            tag="train",
            model=self.args.model,
            run_id=getattr(self.args, "run_id", None),
            console=True,
        )

    def _split_batch(self, batch_m):
        if bool(getattr(self.args, "is_mas", False)):
            batch, mas = batch_m
            batch = batch.to(self.args.device, non_blocking=True)
            mas = mas.to(self.args.device, non_blocking=True)
            mas = mas[:, : self.args.window_size - self.args.n_pred, ...]
        else:
            batch, mas = batch_m[0], None
            batch = batch.to(self.args.device, non_blocking=True)
        return batch, mas

    def _forward(self, batch, mas=None, training: bool = True):
        self.pred_model.train(training)
        x = batch[:, : self.args.window_size - self.args.n_pred, ...]
        target = batch[:, -self.args.n_pred :, ...]
        out = self.pred_model(x, mas=mas)

        if bool(getattr(self.args, "real_value", False)):
            target_np = self.scaler.inverse_transform(
                target.reshape(-1, self.args.n_pred, self.args.nnodes * self.args.out_channels)
                .detach()
                .cpu()
                .numpy()
            )
            target = (
                torch.from_numpy(target_np)
                .float()
                .view(-1, self.args.n_pred, self.args.nnodes, self.args.out_channels)
                .to(batch.device)
            )
        return out, target

    def train_epoch(self, epoch: int) -> float:
        self.pred_model.train(True)
        losses = []
        start = time.time()

        pbar = _progress(
            self.train_loader,
            desc=f"Train {epoch}/{self.args.epochs}",
            total=len(self.train_loader) if hasattr(self.train_loader, "__len__") else None,
            leave=False,
            disable=bool(getattr(self.args, "debug", False)),
        )
        for batch_m in pbar:
            batch, mas = self._split_batch(batch_m)

            self.pred_optimizer.zero_grad(set_to_none=True)
            out, tgt = self._forward(batch, mas=mas, training=True)
            loss = self.pred_loss(out, tgt)
            loss.backward()

            if bool(getattr(self.args, "grad_clip", False)):
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), float(getattr(self.args, "max_grad_norm", 1.0)))

            self.pred_optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix({"pred": f"{loss.item():.4f}"})

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.logger.info(f"[PredOnly][Train] epoch={epoch} loss={mean_loss:.6f} time={time.time()-start:.2f}s")
        return mean_loss

    def val_epoch(self, epoch: int, data_loader) -> float:
        self.pred_model.eval()
        losses = []
        start = time.time()

        with torch.no_grad():
            pbar = _progress(
                data_loader,
                desc=f"Val {epoch}/{self.args.epochs}",
                total=len(data_loader) if hasattr(data_loader, "__len__") else None,
                leave=False,
                disable=bool(getattr(self.args, "debug", False)),
            )
            for batch_m in pbar:
                batch, mas = self._split_batch(batch_m)
                out, tgt = self._forward(batch, mas=mas, training=False)
                loss = self.pred_loss(out, tgt)
                losses.append(float(loss.item()))
                pbar.set_postfix({"pred": f"{loss.item():.4f}"})

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.logger.info(f"[PredOnly][Val] epoch={epoch} loss={mean_loss:.6f} time={time.time()-start:.2f}s")
        return mean_loss

    def save_checkpoint(self):
        state = {
            "pred_state_dict": self.pred_model.state_dict(),
            "pred_optimizer": (self.pred_optimizer.state_dict() if self.pred_optimizer is not None else None),
            "config": self.args,
        }
        tmp = self.best_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.best_path)
        self.logger.info(f"Saved checkpoint: {self.best_path}")

    def train(self):
        best = float("inf")
        bad = 0
        history = []

        for epoch in range(1, int(self.args.epochs) + 1):
            tr = self.train_epoch(epoch)
            val_loader = self.val_loader if self.val_loader is not None else self.test_loader
            va = self.val_epoch(epoch, val_loader)
            history.append({"train_loss": tr, "val_loss": va})

            if va < best:
                best = va
                bad = 0
                self.save_checkpoint()
            else:
                bad += 1

            if bool(getattr(self.args, "early_stop", False)) and bad >= int(getattr(self.args, "early_stop_patience", 10)):
                self.logger.info("Early stop triggered")
                break

        return history


class ReconOnlyTrainer:
    """Train reconstruction (AE) branch only.

    Model interface expected:
        ae_model(x_flat) -> recon_flat
        x_flat: [B, window_size*nnodes*out_channels]
    """

    def __init__(
        self,
        ae_model,
        ae_loss,
        ae_optimizer,
        train_loader,
        val_loader,
        test_loader,
        args,
        scaler,
        logger=None,
    ):
        self.ae_model = ae_model
        self.ae_loss = ae_loss
        self.ae_optimizer = ae_optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler

        ckpt_dir = getattr(self.args, "log_dir_pth", None) or self.args.log_dir
        log_dir = getattr(self.args, "log_dir_log", None) or self.args.log_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.best_path = os.path.join(ckpt_dir, f"best_recon_only_{self.args.data}_{self.args.model}.pth")
        self.logger = logger or get_logger(
            log_dir,
            name=self.args.model,
            debug=getattr(self.args, "debug", False),
            data=self.args.data,
            tag="train",
            model=self.args.model,
            run_id=getattr(self.args, "run_id", None),
            console=True,
        )

        self.ae_channels = int(self.args.window_size * self.args.nnodes * self.args.out_channels)

    def _batch_to_flat(self, batch_m):
        if bool(getattr(self.args, "is_mas", False)):
            batch, _mas = batch_m
        else:
            batch = batch_m[0]
        batch = batch.to(self.args.device, non_blocking=True)

        x_flat = batch.reshape(-1, self.ae_channels)
        if bool(getattr(self.args, "real_value", False)):
            x_np = self.scaler.inverse_transform(
                x_flat.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels)
                .detach()
                .cpu()
                .numpy()
            )
            x_flat = torch.from_numpy(x_np).float().view(-1, self.ae_channels).to(batch.device)
        return x_flat

    def train_epoch(self, epoch: int) -> float:
        self.ae_model.train(True)
        losses = []
        start = time.time()

        pbar = _progress(
            self.train_loader,
            desc=f"Train {epoch}/{self.args.epochs}",
            total=len(self.train_loader) if hasattr(self.train_loader, "__len__") else None,
            leave=False,
            disable=bool(getattr(self.args, "debug", False)),
        )
        for batch_m in pbar:
            x_flat = self._batch_to_flat(batch_m)

            self.ae_optimizer.zero_grad(set_to_none=True)
            out_flat = self.ae_model(x_flat)
            loss = self.ae_loss(out_flat, x_flat)
            loss.backward()

            if bool(getattr(self.args, "grad_clip", False)):
                torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), float(getattr(self.args, "max_grad_norm", 1.0)))

            self.ae_optimizer.step()
            losses.append(float(loss.item()))
            pbar.set_postfix({"recon": f"{loss.item():.4f}"})

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.logger.info(f"[ReconOnly][Train] epoch={epoch} loss={mean_loss:.6f} time={time.time()-start:.2f}s")
        return mean_loss

    def val_epoch(self, epoch: int, data_loader) -> float:
        self.ae_model.eval()
        losses = []
        start = time.time()

        with torch.no_grad():
            pbar = _progress(
                data_loader,
                desc=f"Val {epoch}/{self.args.epochs}",
                total=len(data_loader) if hasattr(data_loader, "__len__") else None,
                leave=False,
                disable=bool(getattr(self.args, "debug", False)),
            )
            for batch_m in pbar:
                x_flat = self._batch_to_flat(batch_m)
                out_flat = self.ae_model(x_flat)
                loss = self.ae_loss(out_flat, x_flat)
                losses.append(float(loss.item()))
                pbar.set_postfix({"recon": f"{loss.item():.4f}"})

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.logger.info(f"[ReconOnly][Val] epoch={epoch} loss={mean_loss:.6f} time={time.time()-start:.2f}s")
        return mean_loss

    def save_checkpoint(self):
        state = {
            "ae_state_dict": self.ae_model.state_dict(),
            "ae_optimizer": (self.ae_optimizer.state_dict() if self.ae_optimizer is not None else None),
            "config": self.args,
        }
        tmp = self.best_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.best_path)
        self.logger.info(f"Saved checkpoint: {self.best_path}")

    def train(self):
        best = float("inf")
        bad = 0
        history = []

        for epoch in range(1, int(self.args.epochs) + 1):
            tr = self.train_epoch(epoch)
            val_loader = self.val_loader if self.val_loader is not None else self.test_loader
            va = self.val_epoch(epoch, val_loader)
            history.append({"train_loss": tr, "val_loss": va})

            if va < best:
                best = va
                bad = 0
                self.save_checkpoint()
            else:
                bad += 1

            if bool(getattr(self.args, "early_stop", False)) and bad >= int(getattr(self.args, "early_stop_patience", 10)):
                self.logger.info("Early stop triggered")
                break

        return history
