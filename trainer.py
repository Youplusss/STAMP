import os
import torch
import torch.nn.functional as F
from typing import Optional
from lib.logger import get_logger
import time
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm

from lib.adv_losses import ramp_weight, pred_total_loss, ae_total_loss, energy_transform
def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = False, disable: bool = False):
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True, disable=disable)


def _set_requires_grad(model: torch.nn.Module, flag: bool) -> None:
    """(可选)在耦合/对抗更新时冻结另一分支参数，减少无用梯度和显存占用。"""
    for p in model.parameters():
        p.requires_grad_(flag)


class Trainer(object):
    def __init__(self, pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.pred_model = pred_model
        self.pred_loss = pred_loss
        self.pred_optimizer = pred_optimizer

        self.ae_model = ae_model
        self.ae_loss = ae_loss
        self.ae_optimizer = ae_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        # Output dirs: prefer explicit subdirs (expe/pth, expe/log) when available
        self.ckpt_dir = getattr(self.args, 'log_dir_pth', None) or self.args.log_dir
        self.log_dir = getattr(self.args, 'log_dir_log', None) or self.args.log_dir

        self.best_path = os.path.join(self.ckpt_dir, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        # self.best_path = os.path.join(self.args.log_dir, 'best_model_unsup_weights_init_' + self.args.data + "_" + self.args.model + '.pth') #用该版本读取无监督节点权重
        # log_dir_transfer 在原始仓库里并非所有脚本都显式定义，这里做兼容处理
        log_dir_transfer = getattr(self.args, 'log_dir_transfer', None) or self.ckpt_dir
        self.transfer_path = os.path.join(log_dir_transfer, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(self.log_dir) == False and not args.debug:
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = get_logger(self.log_dir, name=args.model, debug=args.debug, data=args.data, tag='train', model=args.model, run_id=getattr(args, 'run_id', None), console=True)
        # self.logger = get_logger(args.log_dir, name=args.model + '_unsup', debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(self.log_dir))

        # fix: accidental whitespace split in attribute names
        self.ae_channels = self.args.window_size * self.args.nnodes * self.args.in_channels

        # BEGAN-style equilibrium variable (used only when adv_mode == 'began')
        self._began_k = float(getattr(self.args, 'adv_began_k_init', 0.0))

        # single-branch training switches
        # self.only_pred = bool(getattr(self.args, 'only_pred', False))
        # self.only_recon = bool(getattr(self.args, 'only_recon', False))
        # if self.only_pred and self.only_recon:
        #     raise ValueError('only_pred and only_recon cannot both be True')

        # optimizers may be None when the corresponding branch is disabled
        # if (self.pred_optimizer is None) and (not self.only_recon):
        #     raise ValueError('pred_optimizer is None but forecast branch is enabled')
        # if (self.ae_optimizer is None) and (not self.only_pred):
        #     raise ValueError('ae_optimizer is None but recon branch is enabled')

        # 回退版本：不考虑 preset=None/单分支训练；要求两分支都存在优化器
        if self.pred_optimizer is None:
            raise ValueError('pred_optimizer cannot be None (requires both branches)')
        if self.ae_optimizer is None:
            raise ValueError('ae_optimizer cannot be None (requires both branches)')

    def pred_model_batch(self, batch, training=True, mas = None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        # if self.args.is_mas:
        #     mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform \
                (target.reshape(-1, self.args.n_pred , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
            target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes, self.args.out_channels).to(batch.device)

            x = self.scaler.inverse_transform(x.reshape(-1, (self.args.window_size - self.args.n_pred) , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
            x = torch.from_numpy(x).float().view(-1, self.args.window_size - self.args.n_pred, self.args.nnodes, self.args.out_channels).to(batch.device)

        generate_batch = torch.cat([x, output], dim=1)
        return output, target, generate_batch

    def ae_model_batch(self, batch, training = True):
        self.ae_model.train(training)

        ## [B, T, N, C] --> [B, T * N * C]
        batch = batch.reshape(-1, self.ae_channels)

        output = self.ae_model(batch)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.in_channels).detach().cpu().numpy())
            batch = torch.from_numpy(batch).view(-1, self.ae_channels).float().to(output.device)

        return output, batch

    # -------------------------- adversarial helpers --------------------------
    def _use_adv(self) -> int:
        """Which loss design to use.

        0: no adversarial coupling; use weighted sum baseline (pred + ae)
        1: legacy adversarial training (match tmp/trainer.py legacy4)
        2: current stabilized adversarial design (pred_total_loss/ae_total_loss)

        Backward compatibility:
        - True => 2
        - False => 0
        """
        v = getattr(self.args, 'use_adv', 2)
        if isinstance(v, bool):
            return 2 if v else 0
        try:
            iv = int(v)
        except Exception:
            return 2
        if iv in (0, 1, 2):
            return iv
        return 2

    def _adv_strategy(self) -> str:
        # legacy4: original 4-step schedule in STAMP code (potentially unstable for Mamba)
        # 4step   : 4-step but with stabilized objectives / controllable weights
        # 2step   : 2-step GAN-style (1 pred update + 1 AE update per batch) (recommended)
        return str(getattr(self.args, 'adv_train_strategy', 'legacy4')).lower()

    def _adv_scope(self) -> str:
        # full: use reconstruction error on the whole generated window
        # pred: only use the last n_pred steps (recommended; reduces conflict)
        return str(getattr(self.args, 'adv_scope', 'full')).lower()

    def _adv_mode(self) -> str:
        # legacy / hinge / softplus / exp
        return str(getattr(self.args, 'adv_mode', 'legacy')).lower()

    def _adv_margin(self) -> float:
        return float(getattr(self.args, 'adv_margin', 0.1))

    def _adv_margin_mode(self) -> str:
        return str(getattr(self.args, 'adv_margin_mode', 'rel')).lower()

    def _adv_tau(self) -> float:
        return float(getattr(self.args, 'adv_tau', 1.0))

    def _adv_pred_objective(self) -> str:
        # pred branch coupling objective (see lib/adv_losses.py)
        return str(getattr(self.args, 'adv_pred_objective', 'adv')).lower()

    def _adv_margin_floor(self) -> float:
        return float(getattr(self.args, 'adv_margin_floor', 0.0))

    def _adv_margin_high(self) -> float:
        return float(getattr(self.args, 'adv_margin_high', -1.0))

    def _adv_tau_mode(self) -> str:
        return str(getattr(self.args, 'adv_tau_mode', 'abs')).lower()

    def _adv_tau_floor(self) -> float:
        return float(getattr(self.args, 'adv_tau_floor', 1e-4))

    def _adv_energy_transform(self) -> str:
        return str(getattr(self.args, 'adv_energy_transform', 'none')).lower()

    def _adv_auto_balance(self) -> bool:
        return bool(getattr(self.args, 'adv_auto_balance', False))

    def _began_gamma(self) -> float:
        return float(getattr(self.args, 'adv_began_gamma', 0.5))

    def _began_lambda_k(self) -> float:
        return float(getattr(self.args, 'adv_began_lambda_k', 0.001))

    def _adv_lambdas(self, epoch: int) -> tuple[float, float]:
        """Return (lambda_pred, lambda_ae) for current epoch.

        Uses warmup + linear ramp schedule when available.
        """

        lam_pred_max = float(getattr(self.args, 'adv_lambda_pred', 1.0))
        lam_ae_max = float(getattr(self.args, 'adv_lambda_ae', 1.0))
        warm = int(getattr(self.args, 'adv_warmup_epochs', 1))
        ramp = int(getattr(self.args, 'adv_ramp_epochs', 5))
        lam_pred = ramp_weight(epoch, warm, ramp, lam_pred_max)
        lam_ae = ramp_weight(epoch, warm, ramp, lam_ae_max)
        return lam_pred, lam_ae

    def _recon_loss_from_flat(self, out_flat: torch.Tensor, tgt_flat: torch.Tensor, scope: str = 'full') -> torch.Tensor:
        """Compute reconstruction loss on flattened vectors with optional scope.

        Parameters
        ----------
        out_flat / tgt_flat: [B, T*N*C]
        scope:
            - 'full': whole window
            - 'pred': only last n_pred steps
            - 'history': only first T-n_pred steps
        """
        B = out_flat.shape[0]
        T = self.args.window_size
        N = self.args.nnodes
        C = self.args.out_channels
        out = out_flat.view(B, T, N, C)
        tgt = tgt_flat.view(B, T, N, C)

        scope = (scope or 'full').lower()
        if scope == 'pred':
            out = out[:, -self.args.n_pred:, ...]
            tgt = tgt[:, -self.args.n_pred:, ...]
        elif scope == 'history':
            out = out[:, : T - self.args.n_pred, ...]
            tgt = tgt[:, : T - self.args.n_pred, ...]
        else:
            # full
            pass

        return self.ae_loss(out, tgt)

    def val_epoch(self, epoch, val_dataloader):
        self.pred_model.eval()
        self.ae_model.eval()

        use_adv_mode = self._use_adv()
        # In single-branch mode, adversarial coupling is meaningless; force baseline objective.
        # if self.only_pred or self.only_recon:
        #     use_adv_mode = 0

        # for mode=2 (stabilized)
        adv_strategy = self._adv_strategy()
        adv_scope = self._adv_scope()
        adv_mode = self._adv_mode()
        adv_margin = self._adv_margin()
        adv_margin_mode = self._adv_margin_mode()
        adv_tau = self._adv_tau()
        adv_pred_obj = self._adv_pred_objective()
        adv_margin_floor = self._adv_margin_floor()
        adv_margin_high = self._adv_margin_high()
        adv_tau_mode = self._adv_tau_mode()
        adv_tau_floor = self._adv_tau_floor()
        adv_energy_tr = self._adv_energy_transform()
        adv_auto_balance = self._adv_auto_balance()
        began_gamma = self._began_gamma()
        began_lambda_k = self._began_lambda_k()
        lam_pred, lam_ae = self._adv_lambdas(epoch)

        # legacy schedule coefficients (for mode=1)
        start_epoch = 0
        e = max(epoch - start_epoch, 1)
        legacy_a = 5.0 / e
        legacy_b = 1.0 - 1.0 / e
        legacy_c = 3.0 / e
        legacy_d = 1.0 - 1.0 / e
        # allow global scaling via adv_lambda_* (keep legacy defaults=1)
        legacy_b = legacy_b * float(getattr(self.args, 'adv_lambda_pred', 1.0))
        legacy_d = legacy_d * float(getattr(self.args, 'adv_lambda_ae', 1.0))

        # baseline weights (for mode=0)
        w_pred = float(getattr(self.args, 'loss_weight_pred', getattr(self.args, 'pred_loss_weight', 1.0)))
        w_ae = float(getattr(self.args, 'loss_weight_ae', getattr(self.args, 'ae_loss_weight', 1.0)))

        total_val_pred_loss_list = []
        total_val_ae_loss_list = []
        total_val_adv_loss_list = []

        loss1_list = []
        loss2_list = []
        start_val = time.time()

        with torch.no_grad():
            pbar = _progress(
                val_dataloader,
                desc=f"Val {epoch}/{self.args.epochs}",
                total=len(val_dataloader) if hasattr(val_dataloader, '__len__') else None,
                leave=False,
                disable=bool(getattr(self.args, 'debug', False)),
            )
            for batch_m in pbar:
                if self.args.is_mas:
                    batch, mas = batch_m
                    batch = batch.to(self.args.device, non_blocking=True)
                    mas = mas.to(self.args.device, non_blocking=True)
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                    batch = batch.to(self.args.device, non_blocking=True)

                output, target, generate_batch = self.pred_model_batch(batch, training=False, mas=mas)
                pred_loss = self.pred_loss(output, target)
                if not torch.isnan(pred_loss):
                    total_val_pred_loss_list.append(pred_loss.item())

                output, target = self.ae_model_batch(batch, training=False)
                ae_loss = self._recon_loss_from_flat(output, target, scope='full')
                if not torch.isnan(ae_loss):
                    total_val_ae_loss_list.append(ae_loss.item())

                if self.args.real_value:
                    generate_batch = self.scaler.transform(
                        generate_batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    generate_batch = (
                        torch.from_numpy(generate_batch)
                        .float()
                        .view(-1, self.args.window_size, self.args.nnodes, self.args.out_channels)
                        .to(batch.device)
                    )

                output2, target2 = self.ae_model_batch(generate_batch, training=False)
                adv_loss = self._recon_loss_from_flat(output2, target2, scope=adv_scope)
                if not torch.isnan(adv_loss):
                    total_val_adv_loss_list.append(adv_loss.item())

                # loss1/loss2: for logging
                if use_adv_mode == 0:
                    # no adversarial: weighted sum baseline
                    loss1_t = w_pred * pred_loss + w_ae * ae_loss
                    loss2_t = ae_loss
                elif use_adv_mode == 1:
                    # tmp/trainer.py original design (legacy4)
                    loss1_t = legacy_a * pred_loss + legacy_b * adv_loss
                    loss2_t = legacy_c * ae_loss - legacy_d * adv_loss
                else:
                    # current stabilized design
                    if adv_strategy == 'legacy4':
                        # keep legacy schedule for reference (same as use_adv=1 but under mode=2)
                        loss1_t = legacy_a * pred_loss + legacy_b * adv_loss
                        loss2_t = legacy_c * ae_loss - legacy_d * adv_loss
                    else:
                        loss1_t, _, _ = pred_total_loss(
                            pred_loss,
                            adv_loss,
                            lam_pred,
                            real_recon_loss=ae_loss,
                            pred_objective=adv_pred_obj,
                            margin=adv_margin,
                            margin_mode=adv_margin_mode,  # type: ignore[arg-type]
                            margin_floor=adv_margin_floor,
                            margin_high=adv_margin_high,
                            tau=adv_tau,
                            tau_mode=adv_tau_mode,
                            tau_floor=adv_tau_floor,
                            energy_transform_mode=adv_energy_tr,
                            auto_balance=adv_auto_balance,
                        )

                        if adv_mode == 'began':
                            E_fake = energy_transform(adv_loss, adv_energy_tr)
                            k = float(getattr(self, '_began_k', 0.0))
                            loss2_t = ae_loss - (lam_ae * k) * E_fake
                        else:
                            loss2_t, _, _, _, _ = ae_total_loss(
                                ae_loss,
                                adv_loss,
                                mode=adv_mode,  # type: ignore[arg-type]
                                lambda_ae=lam_ae,
                                margin=adv_margin,
                                margin_mode=adv_margin_mode,  # type: ignore[arg-type]
                                margin_floor=adv_margin_floor,
                                margin_high=adv_margin_high,
                                tau=adv_tau,
                                tau_mode=adv_tau_mode,
                                tau_floor=adv_tau_floor,
                                energy_transform_mode=adv_energy_tr,
                                auto_balance=adv_auto_balance,
                            )

                loss1 = float(loss1_t.item())
                loss2 = float(loss2_t.item())
                loss1_list.append(loss1)
                loss2_list.append(loss2)

                pbar.set_postfix({'pred': f"{pred_loss.item():.4f}", 'ae': f"{ae_loss.item():.4f}", 'adv': f"{adv_loss.item():.4f}", 'L1': f"{loss1:.4f}", 'L2': f"{loss2:.4f}"})

        val_pred_loss = np.array(total_val_pred_loss_list).mean()
        val_ae_loss = np.array(total_val_ae_loss_list).mean()
        val_adv_loss = np.array(total_val_adv_loss_list).mean()
        end_val = time.time()

        loss1 = np.array(loss1_list).mean()
        loss2 = np.array(loss2_list).mean()

        self.logger.info(
            '**********Val Epoch {}: average Loos1: {:.6f}, average Loss2: {:.6f}, while average predModel Loss: {:.6f}, average AE Loss: {:.6f}, average AE Generate Loss: {:.6f}, inference time: {:.3f}s'.
            format(epoch, loss1, loss2, val_pred_loss, val_ae_loss, val_adv_loss, end_val - start_val))

        return val_pred_loss, val_ae_loss, val_adv_loss, loss1, loss2

    def train_epoch(self, epoch):
        self.pred_model.train()
        self.ae_model.train()

        use_adv_mode = self._use_adv()
        # for mode=2 (stabilized)
        adv_strategy = self._adv_strategy()
        adv_scope = self._adv_scope()
        adv_mode = self._adv_mode()
        adv_margin = self._adv_margin()
        adv_margin_mode = self._adv_margin_mode()
        adv_tau = self._adv_tau()
        lam_pred, lam_ae = self._adv_lambdas(epoch)

        adv_pred_obj = self._adv_pred_objective()
        adv_margin_floor = self._adv_margin_floor()
        adv_margin_high = self._adv_margin_high()
        adv_tau_mode = self._adv_tau_mode()
        adv_tau_floor = self._adv_tau_floor()
        adv_energy_tr = self._adv_energy_transform()
        adv_auto_balance = self._adv_auto_balance()
        began_gamma = self._began_gamma()
        began_lambda_k = self._began_lambda_k()

        loss1_list = []
        loss2_list = []
        start_time = time.time()

        start_epoch = 0

        adv_freeze_other = bool(getattr(self.args, 'adv_freeze_other', True))

        do_clip = bool(getattr(self.args, 'grad_clip', False))
        max_grad_norm_default = float(getattr(self.args, 'max_grad_norm', 1.0))
        max_grad_norm_pred = float(getattr(self.args, 'clip_grad_norm_pred', max_grad_norm_default))
        max_grad_norm_ae = float(getattr(self.args, 'clip_grad_norm_ae', max_grad_norm_default))

        # legacy schedule coefficients (per-epoch)
        e = max(epoch - start_epoch, 1)
        legacy_a = 5.0 / e
        legacy_b = 1.0 - 1.0 / e
        legacy_c = 3.0 / e
        legacy_d = 1.0 - 1.0 / e
        legacy_b = legacy_b * float(getattr(self.args, 'adv_lambda_pred', 1.0))
        legacy_d = legacy_d * float(getattr(self.args, 'adv_lambda_ae', 1.0))

        # baseline weights (for mode=0)
        w_pred = float(getattr(self.args, 'loss_weight_pred', getattr(self.args, 'pred_loss_weight', 1.0)))
        w_ae = float(getattr(self.args, 'loss_weight_ae', getattr(self.args, 'ae_loss_weight', 1.0)))

        pbar = _progress(
            self.train_loader,
            desc=f"Train {epoch}/{self.args.epochs}",
            total=len(self.train_loader) if hasattr(self.train_loader, '__len__') else None,
            leave=False,
            disable=bool(getattr(self.args, 'debug', False)),
        )

        for batch_m in pbar:
            # data and target shape: [B, T, N, F]; output shape: [B, T, N, F]
            if self.args.is_mas:
                batch, mas = batch_m
                batch = batch.to(self.args.device, non_blocking=True)
                mas = mas.to(self.args.device, non_blocking=True)
                mas = mas[:, : self.args.window_size - self.args.n_pred, ...]
            else:
                batch, mas = batch_m[0], None
                batch = batch.to(self.args.device, non_blocking=True)

            # -------------------- (A) use_adv=0: no adversarial --------------------
            if use_adv_mode == 0:
                pred_loss = torch.tensor(0.0, device=batch.device)
                ae_loss2 = torch.tensor(0.0, device=batch.device)

                # full (both branches same as before)
                self.pred_optimizer.zero_grad(set_to_none=True)
                out, tgt, _ = self.pred_model_batch(batch, training=True, mas=mas)
                pred_loss = self.pred_loss(out, tgt)

                with torch.no_grad():
                    recon_flat, real_flat = self.ae_model_batch(batch, training=False)
                    ae_loss = self._recon_loss_from_flat(recon_flat, real_flat, scope='full')

                loss1_t = w_pred * pred_loss + w_ae * ae_loss
                loss1_t.backward()
                if do_clip and max_grad_norm_pred > 0:
                    torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_grad_norm_pred)
                self.pred_optimizer.step()

                self.ae_optimizer.zero_grad(set_to_none=True)
                recon_flat2, real_flat2 = self.ae_model_batch(batch, training=True)
                ae_loss2 = self._recon_loss_from_flat(recon_flat2, real_flat2, scope='full')
                ae_loss2.backward()
                if do_clip and max_grad_norm_ae > 0:
                    torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), max_grad_norm_ae)
                self.ae_optimizer.step()

                loss1_list.append(float(loss1_t.item()))
                loss2_list.append(float(ae_loss2.item()))
                pbar.set_postfix({'L': f"{loss1_t.item():.4f}", 'pred': f"{pred_loss.item():.4f}", 'ae': f"{ae_loss2.item():.4f}"})
                continue

            # -------------------- (B) use_adv=1: legacy4 (tmp/trainer.py) --------------------
            if use_adv_mode == 1:
                # 1) pred base update
                self.pred_optimizer.zero_grad(set_to_none=True)
                out, tgt, _ = self.pred_model_batch(batch, training=True, mas=mas)
                pred_loss = self.pred_loss(out, tgt)
                pred_loss.backward()
                if do_clip and max_grad_norm_pred > 0:
                    torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_grad_norm_pred)
                self.pred_optimizer.step()

                # 2) AE base update
                self.ae_optimizer.zero_grad(set_to_none=True)
                recon_flat, real_flat = self.ae_model_batch(batch, training=True)
                ae_loss = self._recon_loss_from_flat(recon_flat, real_flat, scope='full')
                ae_loss.backward()
                if do_clip and max_grad_norm_ae > 0:
                    torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), max_grad_norm_ae)
                self.ae_optimizer.step()

                # 3) pred coupled update
                self.pred_optimizer.zero_grad(set_to_none=True)
                out, tgt, gen = self.pred_model_batch(batch, training=True, mas=mas)
                pred_loss2 = self.pred_loss(out, tgt)

                if self.args.real_value:
                    gen = self.scaler.transform(
                        gen.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gen = (
                        torch.from_numpy(gen)
                        .float()
                        .view(-1, self.args.window_size, self.args.nnodes, self.args.out_channels)
                        .to(batch.device)
                    )

                if adv_freeze_other:
                    _set_requires_grad(self.ae_model, False)
                recon_gen_flat, gen_flat = self.ae_model_batch(gen, training=True)
                if adv_freeze_other:
                    _set_requires_grad(self.ae_model, True)

                adv_loss = self._recon_loss_from_flat(recon_gen_flat, gen_flat, scope=adv_scope)
                loss1_t = legacy_a * pred_loss2 + legacy_b * adv_loss
                loss1_t.backward()
                if do_clip and max_grad_norm_pred > 0:
                    torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_grad_norm_pred)
                self.pred_optimizer.step()

                # 4) AE adversarial update
                self.ae_optimizer.zero_grad(set_to_none=True)
                recon_flat3, real_flat3 = self.ae_model_batch(batch, training=True)
                ae_loss3 = self._recon_loss_from_flat(recon_flat3, real_flat3, scope='full')

                if adv_freeze_other:
                    _set_requires_grad(self.pred_model, False)
                _, _, gen2 = self.pred_model_batch(batch, training=True, mas=mas)
                if adv_freeze_other:
                    gen2 = gen2.detach()
                    _set_requires_grad(self.pred_model, True)

                recon_gen_flat2, gen_flat2 = self.ae_model_batch(gen2, training=True)
                adv_loss2 = self._recon_loss_from_flat(recon_gen_flat2, gen_flat2, scope=adv_scope)
                loss2_t = legacy_c * ae_loss3 - legacy_d * adv_loss2
                loss2_t.backward()
                if do_clip and max_grad_norm_ae > 0:
                    torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), max_grad_norm_ae)
                self.ae_optimizer.step()

                loss1_list.append(float(loss1_t.item()))
                loss2_list.append(float(loss2_t.item()))
                pbar.set_postfix({'L1': f"{loss1_t.item():.4f}", 'L2': f"{loss2_t.item():.4f}"})
                continue

            # -------------------- (C) use_adv=2: current stabilized design (existing) --------------------
            # (note: existing code below kept as-is)
            if adv_strategy == 'legacy4':
                # === 4-step legacy STAMP (kept for reference / ablation) ===
                # 1) pred base update
                self.pred_optimizer.zero_grad(set_to_none=True)
                out, tgt, _ = self.pred_model_batch(batch, training=True, mas=mas)
                pred_loss_base = self.pred_loss(out, tgt)
                pred_loss_base.backward()
                if do_clip and max_grad_norm_pred > 0:
                    torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_grad_norm_pred)
                self.pred_optimizer.step()

                # 2) AE base update
                self.ae_optimizer.zero_grad(set_to_none=True)
                recon_flat, real_flat = self.ae_model_batch(batch, training=True)
                ae_loss_base = self._recon_loss_from_flat(recon_flat, real_flat, scope='full')
                ae_loss_base.backward()
                if do_clip and max_grad_norm_ae > 0:
                    torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), max_grad_norm_ae)
                self.ae_optimizer.step()

            # 3) pred adversarial/coupled update
            self.pred_optimizer.zero_grad(set_to_none=True)
            out, tgt, gen = self.pred_model_batch(batch, training=True, mas=mas)
            pred_loss = self.pred_loss(out, tgt)

            # compute AE recon on generated window (and optionally on real window)
            # We freeze AE *parameters* during pred update (if adv_freeze_other=True),
            # but keep gradients w.r.t. AE inputs so pred can receive adversarial gradients.
            if adv_freeze_other:
                _set_requires_grad(self.ae_model, False)

            recon_gen_flat, gen_flat = self.ae_model_batch(gen, training=True)
            adv_loss = self._recon_loss_from_flat(recon_gen_flat, gen_flat, scope=adv_scope)

            # optional: use real reconstruction loss to define a *gap*-based generator objective
            real_recon_for_pred = None
            if adv_pred_obj != 'adv':
                recon_real_flat_p, real_flat_p = self.ae_model_batch(batch, training=True)
                real_recon_for_pred = self._recon_loss_from_flat(recon_real_flat_p, real_flat_p, scope='full').detach()

            if adv_freeze_other:
                _set_requires_grad(self.ae_model, True)

            loss1_t, _, _ = pred_total_loss(
                pred_loss,
                adv_loss,
                lam_pred,
                real_recon_loss=real_recon_for_pred,
                pred_objective=adv_pred_obj,
                margin=adv_margin,
                margin_mode=adv_margin_mode,
                margin_floor=adv_margin_floor,
                tau=adv_tau,
                tau_mode=adv_tau_mode,
                tau_floor=adv_tau_floor,
                energy_transform_mode=adv_energy_tr,
                auto_balance=adv_auto_balance,
            )
            loss1_t.backward()
            if do_clip and max_grad_norm_pred > 0:
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_grad_norm_pred)
            self.pred_optimizer.step()

            # 4) AE adversarial update
            self.ae_optimizer.zero_grad(set_to_none=True)
            recon_flat, real_flat = self.ae_model_batch(batch, training=True)
            ae_loss = self._recon_loss_from_flat(recon_flat, real_flat, scope='full')

            # Use the *same* generated window but detached (classic GAN discriminator step)
            gen_det = gen.detach()
            recon_gen_flat2, gen_flat2 = self.ae_model_batch(gen_det, training=True)
            adv_loss2 = self._recon_loss_from_flat(recon_gen_flat2, gen_flat2, scope=adv_scope)

            if adv_mode == 'began':
                # BEGAN-style discriminator update: L = L_real - k * E_fake
                E_real = energy_transform(ae_loss, adv_energy_tr)
                E_fake = energy_transform(adv_loss2, adv_energy_tr)
                k = float(getattr(self, '_began_k', 0.0))
                loss2_t = ae_loss - (lam_ae * k) * E_fake
                penalty_t = -(lam_ae * k) * E_fake
                gap_t = E_fake - E_real

                # update k (no grad): keep E_fake close to gamma * E_real
                with torch.no_grad():
                    delta = float(began_gamma) * float(E_real.detach().item()) - float(E_fake.detach().item())
                    self._began_k = float(min(1.0, max(0.0, k + float(began_lambda_k) * delta)))
            else:
                loss2_t, penalty_t, gap_t, _, _ = ae_total_loss(
                    ae_loss,
                    adv_loss2,
                    mode=adv_mode,  # type: ignore[arg-type]
                    lambda_ae=lam_ae,
                    margin=adv_margin,
                    margin_mode=adv_margin_mode,  # type: ignore[arg-type]
                    margin_floor=adv_margin_floor,
                    margin_high=adv_margin_high,
                    tau=adv_tau,
                    tau_mode=adv_tau_mode,
                    tau_floor=adv_tau_floor,
                    energy_transform_mode=adv_energy_tr,
                    auto_balance=adv_auto_balance,
                )
            loss2_t.backward()
            if do_clip and max_grad_norm_ae > 0:
                torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), max_grad_norm_ae)
            self.ae_optimizer.step()

            loss1_list.append(float(loss1_t.item()))
            loss2_list.append(float(loss2_t.item()))

            # concise progress display
            postfix = {
                'L1': f"{loss1_t.item():.4f}",
                'L2': f"{loss2_t.item():.4f}",
                'adv': f"{adv_loss.item():.4f}",
                'gap': f"{gap_t.item():+.4f}",
            }
            if adv_mode == 'began':
                postfix['k'] = f"{float(getattr(self, '_began_k', 0.0)):.3f}"
            pbar.set_postfix(postfix)

        end_time = time.time()

        loss1 = float(np.array(loss1_list).mean()) if len(loss1_list) else float('nan')
        loss2 = float(np.array(loss2_list).mean()) if len(loss2_list) else float('nan')
        self.logger.info(
            '**********Train Epoch {}: averaged Loss1: {:.6f}, Loss2: {:.6f}, train_time: {:.3f}s'.format(
                epoch, loss1, loss2, end_time - start_time
            )
        )
        return loss1, loss2

    def train(self):
        """训练主循环。

        关键改动：
        - 使用 val_pred_loss + val_ae_loss 作为稳定的 checkpoint 指标（避免 loss2 为负或对抗项主导导致误选）。
        - 每个 epoch 若指标改善则保存 best_model 到 self.best_path。
        - 支持 early_stop（与原仓库参数兼容）。
        """

        best_metric = float('inf')
        best_epoch = 0
        not_improved_count = 0

        train_history = []
        val_history = []
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            # ---- train ----
            train_loss1, train_loss2 = self.train_epoch(epoch)
            train_result = {'train_loss1': train_loss1, 'train_loss2': train_loss2}
            train_history.append(train_result)

            # ---- val ----
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            val_pred_loss, val_ae_loss, val_adv_loss, val_loss1, val_loss2 = self.val_epoch(epoch, val_dataloader)
            val_result = {
                'val_pred_loss': val_pred_loss,
                'val_ae_loss': val_ae_loss,
                'val_adv_loss': val_adv_loss,
                'val_loss1': val_loss1,
                'val_loss2': val_loss2,
            }
            val_history.append(val_result)

            # ---- lr scheduler (optional) ----
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except Exception:
                    # some schedulers expect val loss; keep backward compatible
                    pass

            # ---- checkpoint metric (stable) ----
            val_metric = float(val_pred_loss + val_ae_loss)
            if np.isnan(val_metric) or np.isinf(val_metric):
                self.logger.warning(
                    f"Validation metric is NaN/Inf at epoch {epoch} (val_pred_loss={val_pred_loss}, val_ae_loss={val_ae_loss}). "
                    "Stop training to avoid saving a broken checkpoint."
                )
                break

            improved = val_metric < best_metric - 1e-12
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                not_improved_count = 0
                self.save_checkpoint(self.best_path)
                self.logger.info(
                    f"********** New best checkpoint at epoch {epoch}: val_metric(pred+ae)={val_metric:.6f}"
                )
            else:
                not_improved_count += 1

            # ---- divergence guard ----
            if (abs(train_loss1) > 1e6) or (abs(train_loss2) > 1e6):
                self.logger.warning(
                    f"Gradient explosion detected (train_loss1={train_loss1:.3e}, train_loss2={train_loss2:.3e}). Stop."
                )
                break

            # ---- early stop ----
            if bool(getattr(self.args, 'early_stop', False)):
                patience = int(getattr(self.args, 'early_stop_patience', 10))
                if not_improved_count >= patience:
                    self.logger.info(
                        f"Validation metric didn't improve for {patience} epochs. Training stops."
                    )
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))
        if best_epoch > 0:
            self.logger.info(
                f"Best epoch: {best_epoch}, best val_metric(pred+ae)={best_metric:.6f}. Saved to {self.best_path}"
            )
        else:
            # as a fallback, always save something
            self.logger.warning("No valid best checkpoint found during training; saving last epoch model.")
            self.save_checkpoint(self.best_path)

        return train_history, val_history

    def save_checkpoint(self, path=None):
        """保存 checkpoint。

        参数
        ----
        path: str | None
            保存路径。若为 None，则默认保存到 self.best_path。

        说明
        ----
        为避免训练被中断/并发写入导致 .pth 文件损坏，这里使用“临时文件 + 原子替换”。
        """
        if path is None:
            path = self.best_path

        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),
            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),
            'config': self.args,
        }

        # atomic save: write to temp then replace
        tmp_path = str(path) + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        self.logger.info("Saving model checkpoint to " + str(path))

    def save_checkpoint_transfer(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),

            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),

            'config': self.args
        }
        tmp_path = str(self.transfer_path) + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.transfer_path)
        self.logger.info("Saving current best transfer model to " + self.transfer_path)


class Tester(object):
    def __init__(self, pred_model, ae, args, scaler, logger, path=None, alpha=.5, beta=.5, gamma=0.):
        self.pred_model = pred_model
        self.ae = ae
        self.args = args
        # self.data_loader = data_loader
        self.scaler = scaler
        self.logger = logger
        self.path = path
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def testing(self, data_loader, map_location=None):
        if self.path != None:
            print("load model: ", self.path)
            # PyTorch >=2.6 defaults weights_only=True; our checkpoint contains argparse.Namespace in 'config'
            check_point = torch.load(self.path, map_location=map_location, weights_only=False)
            pred_state_dict = check_point['pred_state_dict']
            ae_state_dict = check_point['ae_state_dict']
            # args = check_point['config']
            self.pred_model.load_state_dict(pred_state_dict)
            self.ae.load_state_dict(ae_state_dict)
            self.pred_model.to(self.args.device)
            self.ae.to(self.args.device)

        self.pred_model.eval()
        self.ae.eval()

        scores = []
        loss1_list = []
        loss2_list = []
        gt_list = []
        pred_list = []
        origin_list = []
        construct_list = []
        generate_list = []
        generate_construct_list = []
        pred_channels = self.args.n_pred * self.args.nnodes * self.args.out_channels
        ae_channels = self.args.window_size * self.args.nnodes * self.args.out_channels
        with torch.no_grad():
            pbar = _progress(
                data_loader,
                desc="Test",
                total=len(data_loader) if hasattr(data_loader, '__len__') else None,
                leave=False,
                disable=bool(getattr(self.args, 'debug', False)),
            )
            for batch_m in pbar:
                if self.args.is_mas:
                    batch, mas = batch_m
                    batch = batch.to(self.args.device, non_blocking=True)
                    mas = mas.to(self.args.device, non_blocking=True)
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                    batch = batch.to(self.args.device, non_blocking=True)

                x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
                output = self.pred_model(x, mas=mas)

                target1 = batch.reshape(-1, ae_channels)
                output1 = self.ae(target1)

                if self.args.real_value:
                    target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred,
                                                                          self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,
                                                                   self.args.out_channels).to(batch.device)

                    x = self.scaler.inverse_transform(x.reshape(-1, (self.args.window_size - self.args.n_pred),
                                                                self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    x = torch.from_numpy(x).float().view(-1, self.args.window_size - self.args.n_pred, self.args.nnodes,
                                                         self.args.out_channels).to(batch.device)

                    target1 = self.scaler.inverse_transform(target1.reshape(-1, self.args.window_size,
                                                                            self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target1 = torch.from_numpy(target1).float().view(-1, ae_channels).to(batch.device)

                generate_batch = torch.cat([x, output], dim=1)
                generate_batch = generate_batch.reshape(-1, ae_channels)
                generate_batch_out = self.ae(generate_batch)

                loss1 = torch.mean((output.reshape(-1, pred_channels) - target.reshape(-1, pred_channels)) ** 2,
                                   axis=1)
                loss2 = torch.mean((output1 - target1) ** 2, axis=1)
                loss3 = torch.mean((generate_batch - generate_batch_out) ** 2, axis=1)

                score = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3
                scores.append(score.detach().cpu().numpy())

                loss1_list.append(loss1.detach().cpu().numpy())
                loss2_list.append(loss2.detach().cpu().numpy())

                gt_list.append(target.reshape(-1, self.args.n_pred,
                                              self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1, self.args.n_pred,
                                                self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                origin_list.append(target1.reshape(-1, self.args.window_size,
                                                   self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                construct_list.append(output1.reshape(-1, self.args.window_size,
                                                      self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                generate_list.append(generate_batch.reshape(-1, self.args.window_size,
                                                            self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                generate_construct_list.append(generate_batch_out.reshape(-1, self.args.window_size,
                                                                          self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                pbar.set_postfix({
                    'score': f"{float(score.mean().item()):.4f}"
                })

        return scores, loss1_list, loss2_list, pred_list, gt_list, origin_list, construct_list, generate_list, generate_construct_list


class PredictedModelTrainer(object):
    def __init__(self, pred_model, pred_loss, pred_optimizer, train_loader, val_loader, test_loader, args, scaler,
                 lr_scheduler=None):
        super(PredictedModelTrainer, self).__init__()
        self.pred_model = pred_model
        self.pred_loss = pred_loss
        self.pred_optimizer = pred_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        ckpt_dir = getattr(self.args, 'log_dir_pth', None) or self.args.log_dir
        log_dir = getattr(self.args, 'log_dir_log', None) or self.args.log_dir
        self.best_path = os.path.join(ckpt_dir,
                                      'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(log_dir) == False and not args.debug:
            os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger(log_dir, name=args.model, debug=args.debug, data=args.data, tag='train', model=args.model, run_id=getattr(args, 'run_id', None), console=True)
        self.logger.info('Experiment log path in: {}'.format(log_dir))

    def pred_model_batch(self, batch, training=True, mas=None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform(
                target.reshape(-1, self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
            target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,
                                                           self.args.out_channels).to(batch.device)
        generate_batch = torch.cat([x, output], dim=1)
        return output, target, generate_batch

    def val_epoch(self, epoch, val_dataloader):
        self.pred_model.eval()

        total_val_pred_loss = 0

        loss1_list = []

        start_val = time.time()
        with torch.no_grad():
            pbar = _progress(
                val_dataloader,
                desc=f"Val {epoch}/{self.args.epochs}",
                total=len(val_dataloader) if hasattr(val_dataloader, '__len__') else None,
                leave=False,
                disable=bool(getattr(self.args, 'debug', False)),
            )
            for batch_m in pbar:
                if self.args.is_mas:
                    batch, mas = batch_m
                    batch = batch.to(self.args.device, non_blocking=True)
                    mas = mas.to(self.args.device, non_blocking=True)
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                    batch = batch.to(self.args.device, non_blocking=True)
                ## batch: [B,T,N,C]

                ## eval pred_Model
                output, target, generate_batch = self.pred_model_batch(batch, training=False, mas=mas)
                pred_loss = self.pred_loss(output, target)
                if not torch.isnan(pred_loss):
                    total_val_pred_loss += pred_loss.item()

                loss1_list.append(pred_loss.item())

                pbar.set_postfix({
                    'pred': f"{pred_loss.item():.4f}",
                })

        val_pred_loss = total_val_pred_loss / len(val_dataloader)

        end_val = time.time()
        self.logger.info('**********Val Epoch {}: average predModel Loss: {:.6f}, inference time: {:.3f}s'.
                         format(epoch, val_pred_loss, end_val - start_val))

        loss1 = np.array(loss1_list).mean()

        self.logger.info('**********Val Epoch {}: average Loos1: {:.6f}'.format(epoch, loss1))

        return val_pred_loss, loss1

    def train_epoch(self, epoch):
        self.pred_model.train()

        loss1_list = []
        start_time = time.time()

        for batch_m in self.train_loader:
            if self.args.is_mas:
                batch, mas = batch_m
                mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
            else:
                batch, mas = batch_m[0], None

            # data and target shape: B, T, N, F; output shape: B, T, N, F
            self.pred_optimizer.zero_grad()
            ## train predModel
            output, target, generate_batch = self.pred_model_batch(batch, training=True, mas=mas)

            pred_loss = self.pred_loss(output, target)
            loss1 = pred_loss
            loss1.backward()
            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), self.args.max_grad_norm)

            self.pred_optimizer.step()

            loss1_list.append(loss1.item())

        end_time = time.time()
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        loss1 = np.array(loss1_list).mean()

        self.logger.info('**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, loss1,
                                                                                                        end_time - start_time))
        return loss1

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0

        history = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            ## train
            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            ## val
            val_pred_loss, val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            result = {'train_loss': train_epoch_loss, "val_loss": val_epoch_loss}
            history.append(result)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            ## early_stop
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))

        # save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return history

    def save_checkpoint(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': (self.pred_optimizer.state_dict() if self.pred_optimizer is not None else None),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)


class AEModelTrainer(object):
    def __init__(self, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, scaler,
                 lr_scheduler=None):
        super(AEModelTrainer, self).__init__()
        self.ae_model = ae_model
        self.ae_loss = ae_loss
        self.ae_optimizer = ae_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        ckpt_dir = getattr(self.args, 'log_dir_pth', None) or self.args.log_dir
        log_dir = getattr(self.args, 'log_dir_log', None) or self.args.log_dir
        self.best_path = os.path.join(ckpt_dir,
                                      'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(log_dir) == False and not args.debug:
            os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger(log_dir, name=args.model, debug=args.debug, data=args.data, tag='train', model=args.model, run_id=getattr(args, 'run_id', None), console=True)
        self.logger.info('Experiment log path in: {}'.format(log_dir))

        # Ensure ae_channels exists (flattened [B, T, N, C])
        self.ae_channels = self.args.window_size * self.args.nnodes * self.args.in_channels

    def ae_model_batch(self, batch, training=True):
        self.ae_model.train(training)

        ## [B, T, N, C] --> [B, T * N * C]
        batch = batch.reshape(-1, self.ae_channels)

        output = self.ae_model(batch)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size,
                                                                self.args.nnodes * self.args.in_channels).detach().cpu().numpy())
            batch = torch.from_numpy(batch).view(-1, self.ae_channels).float().to(output.device)
        return output, batch

    def val_epoch(self, epoch, val_dataloader):
        self.ae_model.eval()

        ae_loss_list = []

        start_val = time.time()
        with torch.no_grad():
            for [batch] in val_dataloader:
                ## batch: [B,T,N,C]

                ## eval pred_Model
                output, batch = self.ae_model_batch(batch, training=False)
                ae_loss = self.ae_loss(output, batch)
                if not torch.isnan(ae_loss):
                    ae_loss_list.append(ae_loss.item())

                ae_loss_list.append(ae_loss.item())

        end_val = time.time()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info(
            '**********Val Epoch {}: average AEModel Loss: {:.6f}, inference time: {:.3f}s'.format(epoch, ae_loss,
                                                                                                   end_val - start_val))

        return ae_loss

    def train_epoch(self, epoch):
        self.ae_model.train()

        ae_loss_list = []
        start_time = time.time()

        for [batch] in self.train_loader:
            # data and target shape: B, T, N, F; output shape: B, T, N, F
            self.ae_optimizer.zero_grad()
            ## train AE
            output, batch = self.ae_model_batch(batch, training=True)
            ae_loss = self.ae_loss(output, batch)

            ae_loss.backward()

            self.ae_optimizer.step()

            ae_loss_list.append(ae_loss.item())

        end_time = time.time()
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info(
            '**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, ae_loss,
                                                                                           end_time - start_time))
        return ae_loss

    def train(self):

        best_loss = float('inf')
        not_improved_count = 0

        history = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            ## train
            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            ## val
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            result = {'train_loss': train_epoch_loss, "val_loss": val_epoch_loss}
            history.append(result)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            ## early_stop
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return history

    def save_checkpoint(self):
        state = {
            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)


class AETester(object):
    def __init__(self, ae_model, args, scaler, logger, path=None):
        self.ae_model = ae_model
        self.args = args
        self.scaler = scaler
        self.logger = logger
        self.path = path

    def testing(self, data_loader, map_location=None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location=map_location, weights_only=False)
            ae_state_dict = check_point['ae_state_dict']
            # args = check_point['config']
            self.ae_model.load_state_dict(ae_state_dict)
            self.ae_model.to(self.args.device)
        self.ae_model.eval()

        scores = []
        gt_list = []
        pred_list = []

        ae_channels = self.args.window_size * self.args.nnodes * self.args.out_channels

        with torch.no_grad():
            for [batch] in data_loader:
                ## [B, T, N, C] --> [B, T * N * C]
                batch = batch.reshape(-1, ae_channels)

                output = self.ae_model(batch)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size,
                                                                        self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    batch = torch.from_numpy(batch).float().view(-1, self.args.window_size, self.args.nnodes,
                                                                 self.args.out_channels).to(batch.device)

                score = torch.mean((output.reshape(-1, ae_channels) - batch.reshape(-1, ae_channels)) ** 2,
                                   axis=1)  ##axis=(1,2,3)

                scores.append(score.detach().cpu().numpy())

                gt_list.append(batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

        return scores, pred_list, gt_list

