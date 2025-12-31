import os
import torch
from lib.logger import get_logger
import time
import math
import copy
import numpy as np
import pandas as pd


def _set_requires_grad(model: torch.nn.Module, flag: bool) -> None:
    """Optionally freeze a branch to reduce useless gradients/memory during coupled updates."""
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

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        # self.best_path = os.path.join(self.args.log_dir, 'best_model_unsup_weights_init_' + self.args.data + "_" + self.args.model + '.pth') # 用该版本读取无监督节点权重

        # 兼容 transfer path：若未提供 log_dir_transfer，则回退到 log_dir
        log_dir_transfer = getattr(self.args, 'log_dir_transfer', None) or self.args.log_dir
        self.transfer_path = os.path.join(log_dir_transfer, 'best_model_' + self.args.data + "_" + self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        # self.logger = get_logger(args.log_dir, name=args.model + '_unsup', debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        self.ae_channels = self.args.window_size*self.args.nnodes*self.args.in_channels

    def pred_model_batch(self, batch, training=True, mas = None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        # if self.args.is_mas:
        #     mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
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
def _adv_lambda(self, epoch: int, which: str) -> float:
    """Adversarial weight schedule (warmup + ramp).

    which: 'pred' or 'ae'
    """
    warmup = int(getattr(self.args, "adv_warmup_epochs", 1))
    ramp = int(getattr(self.args, "adv_ramp_epochs", 5))
    if epoch <= warmup:
        return 0.0
    progress = (epoch - warmup) / float(max(1, ramp))
    progress = max(0.0, min(1.0, progress))
    if which == "pred":
        max_l = float(getattr(self.args, "adv_lambda_pred", 0.5))
    elif which == "ae":
        max_l = float(getattr(self.args, "adv_lambda_ae", 1.0))
    else:
        raise ValueError(f"Unknown which={which}")
    return max_l * progress

def _ae_adv_penalty(self, ae_loss: torch.Tensor, adv_loss: torch.Tensor) -> torch.Tensor:
    """A bounded penalty that encourages adv_loss to be large (for the AE),
    without making the objective unbounded below.
    """
    mode = str(getattr(self.args, "adv_mode", "hinge")).lower()
    if mode == "legacy":
        # Caller is expected to use: loss2 = ae_loss - lambda * adv_loss
        return adv_loss
    if mode == "hinge":
        margin = float(getattr(self.args, "adv_margin", 0.1))
        # Encourage adv_loss >= ae_loss + margin; stop pushing once satisfied.
        return torch.relu((ae_loss.detach() + margin) - adv_loss)
    if mode == "exp":
        tau = float(getattr(self.args, "adv_tau", 1.0))
        tau = max(tau, 1e-6)
        # Minimize exp(-adv/tau) -> encourages larger adv, but saturates to 0.
        return torch.exp(-adv_loss / tau)
    raise ValueError(f"Unknown adv_mode: {mode}")


def val_epoch(self, epoch, dataloader=None):
    """Validation over normal windows only (no gradients)."""
    self.pred_model.eval()
    self.ae_model.eval()

    val_loader = self.val_loader if dataloader is None else dataloader

    total_pred, total_ae, total_adv = 0.0, 0.0, 0.0
    total_loss1, total_loss2 = 0.0, 0.0
    num = 0

    lam_pred = self._adv_lambda(epoch, "pred")
    lam_ae = self._adv_lambda(epoch, "ae")
    adv_mode = str(getattr(self.args, "adv_mode", "hinge")).lower()

    with torch.no_grad():
        for batch_m in val_loader:
            batch = batch_m[0] if isinstance(batch_m, (list, tuple)) else batch_m
            mas = batch_m[1] if (isinstance(batch_m, (list, tuple)) and len(batch_m) > 1 and self.args.is_mas) else None

            # pred loss + adv loss on generated
            pred_output, pred_target, gen = self.pred_model_batch(batch, mas=mas)
            pred_loss = self.pred_criterion(pred_output, pred_target).float()

            recon_gen, gen_for_loss = self.ae_model_batch(gen, mas=mas)
            adv_loss = self.ae_criterion(recon_gen, gen_for_loss).float()

            # AE loss on real
            recon_real, real_for_loss = self.ae_model_batch(batch, mas=mas)
            ae_loss = self.ae_criterion(recon_real, real_for_loss).float()

            loss1 = pred_loss + lam_pred * adv_loss
            if adv_mode == "legacy":
                loss2 = ae_loss - lam_ae * adv_loss
            else:
                loss2 = ae_loss + lam_ae * self._ae_adv_penalty(ae_loss, adv_loss)

            bs = batch.shape[0]
            total_pred += pred_loss.item() * bs
            total_adv += adv_loss.item() * bs
            total_ae += ae_loss.item() * bs
            total_loss1 += loss1.item() * bs
            total_loss2 += loss2.item() * bs
            num += bs

    if num == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (total_loss1 / num,
            total_loss2 / num,
            total_pred / num,
            total_ae / num,
            total_adv / num)


def train_epoch(self, epoch, dataloader=None):
    self.pred_model.train()
    self.ae_model.train()

    train_loader = self.train_loader if dataloader is None else dataloader

    total_pred, total_ae, total_adv = 0.0, 0.0, 0.0
    total_loss1, total_loss2 = 0.0, 0.0
    num = 0

    lam_pred = self._adv_lambda(epoch, "pred")
    lam_ae = self._adv_lambda(epoch, "ae")
    adv_mode = str(getattr(self.args, "adv_mode", "hinge")).lower()

    clip_pred = float(getattr(self.args, "clip_grad_norm_pred", 0.0))
    clip_ae = float(getattr(self.args, "clip_grad_norm_ae", 0.0))

    for batch_m in train_loader:
        batch = batch_m[0] if isinstance(batch_m, (list, tuple)) else batch_m
        mas = batch_m[1] if (isinstance(batch_m, (list, tuple)) and len(batch_m) > 1 and self.args.is_mas) else None

        # ------------------ (1) Update prediction model ------------------
        self.pred_optimizer.zero_grad(set_to_none=True)

        if self.args.adv_freeze_other:
            _set_requires_grad(self.ae_model, False)

        pred_output, pred_target, gen = self.pred_model_batch(batch, mas=mas)
        pred_loss = self.pred_criterion(pred_output, pred_target).float()

        recon_gen, gen_for_loss = self.ae_model_batch(gen, mas=mas)
        adv_loss = self.ae_criterion(recon_gen, gen_for_loss).float()

        if self.args.adv_freeze_other:
            _set_requires_grad(self.ae_model, True)

        loss1 = pred_loss + lam_pred * adv_loss

        if not torch.isfinite(loss1):
            self.logger.info(f"[WARN] Non-finite pred loss at epoch {epoch}. Skipping batch.")
        else:
            loss1.backward()
            if clip_pred > 0:
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), clip_pred)
            self.pred_optimizer.step()

        # ------------------ (2) Update reconstruction (AE) model ------------------
        self.ae_optimizer.zero_grad(set_to_none=True)

        # Freeze pred model to avoid gradients flowing into it
        if self.args.adv_freeze_other:
            _set_requires_grad(self.pred_model, False)

        # Re-generate with current pred model (no grad into pred)
        _, _, gen2 = self.pred_model_batch(batch, mas=mas)
        gen2 = gen2.detach()

        if self.args.adv_freeze_other:
            _set_requires_grad(self.pred_model, True)

        recon_real, real_for_loss = self.ae_model_batch(batch, mas=mas)
        ae_loss = self.ae_criterion(recon_real, real_for_loss).float()

        recon_gen2, gen2_for_loss = self.ae_model_batch(gen2, mas=mas)
        adv_loss2 = self.ae_criterion(recon_gen2, gen2_for_loss).float()

        if adv_mode == "legacy":
            loss2 = ae_loss - lam_ae * adv_loss2
        else:
            loss2 = ae_loss + lam_ae * self._ae_adv_penalty(ae_loss, adv_loss2)

        if not torch.isfinite(loss2):
            self.logger.info(f"[WARN] Non-finite AE loss at epoch {epoch}. Skipping batch.")
        else:
            loss2.backward()
            if clip_ae > 0:
                torch.nn.utils.clip_grad_norm_(self.ae_model.parameters(), clip_ae)
            self.ae_optimizer.step()

        bs = batch.shape[0]
        total_pred += pred_loss.item() * bs
        total_ae += ae_loss.item() * bs
        total_adv += adv_loss2.item() * bs
        total_loss1 += loss1.item() * bs
        total_loss2 += loss2.item() * bs
        num += bs

    if num == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (total_loss1 / num,
            total_loss2 / num,
            total_pred / num,
            total_ae / num,
            total_adv / num)


def train(self):
    """Train with best-checkpoint selection on validation (normal-only) loss."""
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = int(getattr(self.args, "early_stop_patience", 10))

    for epoch in range(1, self.args.epochs + 1):
        start_time = time.time()
        tr_loss1, tr_loss2, tr_pred, tr_ae, tr_adv = self.train_epoch(epoch)
        train_time = time.time() - start_time

        self.logger.info(
            f"**********Train Epoch {epoch}: averaged Loss1: {tr_loss1:.6f}, "
            f"Loss2: {tr_loss2:.6f}, predLoss: {tr_pred:.6f}, aeLoss: {tr_ae:.6f}, "
            f"advLoss: {tr_adv:.6f}, train_time: {train_time:.3f}s"
        )

        if self.val_loader is None:
            continue

        start_time = time.time()
        va_loss1, va_loss2, va_pred, va_ae, va_adv = self.val_epoch(epoch)
        infer_time = time.time() - start_time

        self.logger.info(
            f"**********Val Epoch {epoch}: average Loss1: {va_loss1:.6f}, average Loss2: {va_loss2:.6f}, "
            f"while average predModel Loss: {va_pred:.6f}, average AE Loss: {va_ae:.6f}, "
            f"average AE Generate Loss: {va_adv:.6f}, inference time: {infer_time:.3f}s"
        )

        # Model selection metric: only use *meaningful, bounded* losses on normal data
        metric = va_pred + va_ae

        if (not math.isfinite(metric)) or (not math.isfinite(va_pred)) or (not math.isfinite(va_ae)):
            self.logger.info("[WARN] Non-finite validation metric; stopping early.")
            break

        if metric + 1e-12 < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            self.save_checkpoint()
            self.logger.info(f"Saving new best model at epoch {epoch} (metric={best_metric:.6f}) to {self.best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                self.logger.info(
                    f"Early stopping: no improvement for {patience} epochs. "
                    f"Best epoch={best_epoch}, best metric={best_metric:.6f}"
                )
                break

    self.logger.info(f"Training finished. Best epoch={best_epoch}, best metric={best_metric:.6f}")




    def save_checkpoint(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),

            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),

            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    def save_checkpoint_transfer(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),

            'ae_state_dict': self.ae_model.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),

            'config': self.args
        }
        torch.save(state, self.transfer_path)
        self.logger.info("Saving current best transfer model to " + self.transfer_path)



class Tester(object):
    def __init__(self, pred_model, ae, args, scaler, logger, path=None, alpha=.5, beta=.5, gamma = 0.):
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

    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            # PyTorch >=2.6 defaults weights_only=True; we need full checkpoint (includes argparse.Namespace).
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
            for batch_m in data_loader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None

                x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
                output = self.pred_model(x, mas=mas)

                target1 = batch.reshape(-1, ae_channels)
                output1 = self.ae(target1)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                    x = self.scaler.inverse_transform(x.reshape(-1, (self.args.window_size - self.args.n_pred),self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    x = torch.from_numpy(x).float().view(-1, self.args.window_size - self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                    target1 = self.scaler.inverse_transform(target1.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target1 = torch.from_numpy(target1).float().view(-1, ae_channels).to(batch.device)

                generate_batch = torch.cat([x, output], dim=1)
                generate_batch = generate_batch.reshape(-1, ae_channels)
                generate_batch_out = self.ae(generate_batch)


                loss1 = torch.mean((output.reshape(-1,pred_channels) - target.reshape(-1,pred_channels)) ** 2, dim=1)  ## (batch_size,)
                loss2 = torch.mean((output1 - target1) ** 2, dim=1)
                loss3 = torch.mean((generate_batch - generate_batch_out) ** 2, dim=1)

                score = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3#(batch_size)
                scores.append(score.detach().cpu().numpy())


                loss1_list.append(loss1.detach().cpu().numpy())#(batch_size)
                loss2_list.append(loss2.detach().cpu().numpy())#(batch_size)


                gt_list.append(target.reshape(-1,self.args.n_pred , self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                origin_list.append(target1.reshape(-1,self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                construct_list.append(output1.reshape(-1,self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                generate_list.append(generate_batch.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                generate_construct_list.append(generate_batch_out.reshape(-1, self.args.window_size, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
        return scores, loss1_list, loss2_list, pred_list, gt_list, origin_list, construct_list, generate_list, generate_construct_list




class PredictedModelTrainer(object):
    def __init__(self, pred_model, pred_loss, pred_optimizer,  train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
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

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data +"_" +self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def pred_model_batch(self, batch, training=True, mas = None):
        self.pred_model.train(training)
        x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
        output = self.pred_model(x, mas=mas)

        if self.args.real_value:
        #     outputs = self.scaler.inverse_transform(outputs)
            target = self.scaler.inverse_transform(target.reshape(-1,self.args.n_pred, self.args.nnodes*self.args.out_channels).detach().cpu().numpy())
            target = torch.from_numpy(target).float().view(-1,self.args.n_pred,self.args.nnodes,self.args.out_channels).to(batch.device)
        generate_batch = torch.cat([x, output], dim=1)
        return output, target, generate_batch


    def val_epoch(self, epoch, val_dataloader):
        self.pred_model.eval()

        total_val_pred_loss = 0

        loss1_list = []

        start_val = time.time()
        with torch.no_grad():
            for batch_m in val_dataloader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None
                ## batch: [B,T,N,C]

                ## eval pred_Model
                output, target, generate_batch = self.pred_model_batch(batch, training=False, mas=mas)
                pred_loss = self.pred_loss(output, target)
                if not torch.isnan(pred_loss):
                    total_val_pred_loss += pred_loss.item()

                loss1_list.append(pred_loss.item() )

        val_pred_loss = total_val_pred_loss / len(val_dataloader)

        end_val = time.time()
        self.logger.info('**********Val Epoch {}: average predModel Loss: {:.6f}, inference time: {:.3f}s'.
                        format(epoch, val_pred_loss, end_val-start_val))

        loss1 = np.array(loss1_list).mean()

        self.logger.info( '**********Val Epoch {}: average Loos1: {:.6f}'.format(epoch, loss1))

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

            #data and target shape: B, T, N, F; output shape: B, T, N, F
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
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        loss1 = np.array(loss1_list).mean()

        self.logger.info('**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, loss1, end_time - start_time))
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

        #save the best model to file
        self.save_checkpoint()
        self.logger.info("Saving current best model to " + self.best_path)
        return history

    def save_checkpoint(self):
        state = {
            'pred_state_dict': self.pred_model.state_dict(),
            'pred_optimizer': self.pred_optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)




class PredTester(object):
    def __init__(self, pred_model, args, scaler, logger, path=None):
        self.pred_model = pred_model
        self.args = args
        self.scaler = scaler
        self.logger = logger
        self.path = path


    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location = map_location)
            pred_state_dict = check_point['pred_state_dict']
            # args = check_point['config']
            self.pred_model.load_state_dict(pred_state_dict)

            self.pred_model.to(self.args.device)

        self.pred_model.eval()

        scores = []
        gt_list = []
        pred_list = []

        pred_channels = self.args.n_pred * self.args.nnodes * self.args.out_channels

        with torch.no_grad():
            for batch_m in data_loader:
                if self.args.is_mas:
                    batch, mas = batch_m
                    mas = mas[:, :self.args.window_size - self.args.n_pred, ...]
                else:
                    batch, mas = batch_m[0], None

                x, target = batch[:, :self.args.window_size - self.args.n_pred, ...], batch[:, -self.args.n_pred:, ...]
                output = self.pred_model(x, mas=mas)

                if self.args.real_value:
                    #     outputs = self.scaler.inverse_transform(outputs)
                    target = self.scaler.inverse_transform(target.reshape(-1, self.args.n_pred,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    target = torch.from_numpy(target).float().view(-1, self.args.n_pred, self.args.nnodes,self.args.out_channels).to(batch.device)

                score = torch.mean((output.reshape(-1,pred_channels) - target.reshape(-1,pred_channels)) ** 2, dim=1)  ## (batch_size)

                scores.append(score.detach().cpu().numpy())

                # pred_list.append(output.reshape(-1, self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

                gt_list.append(target.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1,self.args.n_pred, self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

        return scores, pred_list, gt_list


class AEModelTrainer(object):
    def __init__(self, ae_model, ae_loss, ae_optimizer,  train_loader, val_loader, test_loader, args, scaler, lr_scheduler=None):
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

        self.best_path = os.path.join(self.args.log_dir, 'best_model_' + self.args.data +"_" +self.args.model + '.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        ## log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        self.ae_channels = self.args.window_size*self.args.nnodes*self.args.in_channels

    def ae_model_batch(self, batch, training=True):
        self.ae_model.train(training)

        ## [B, T, N, C] --> [B, T * N * C]
        batch = batch.reshape(-1, self.ae_channels)

        output = self.ae_model(batch)

        if self.args.real_value:
            #     outputs = self.scaler.inverse_transform(outputs)
            batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy())
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

                ae_loss_list.append(ae_loss.item() )

        end_val = time.time()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info('**********Val Epoch {}: average AEModel Loss: {:.6f}, inference time: {:.3f}s'.format(epoch, ae_loss, end_val - start_val))

        return ae_loss

    def train_epoch(self, epoch):
        self.ae_model.train()

        ae_loss_list = []
        start_time = time.time()

        for [batch] in self.train_loader:
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            self.ae_optimizer.zero_grad()
            ## train AE
            output, batch = self.ae_model_batch(batch, training=True)
            ae_loss = self.ae_loss(output, batch)

            ae_loss.backward()

            self.ae_optimizer.step()

            ae_loss_list.append(ae_loss.item())

        end_time = time.time()
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()

        ae_loss = np.array(ae_loss_list).mean()

        self.logger.info(
            '**********Train Epoch {}: averaged Loss1: {:.6f}, train_time: {:.3f}s'.format(epoch, ae_loss,end_time - start_time))
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

        #save the best model to file
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

    def testing(self, data_loader, map_location = None):
        if self.path != None:
            print("load model: ", self.path)
            check_point = torch.load(self.path, map_location = map_location)
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
                    batch = self.scaler.inverse_transform(batch.reshape(-1, self.args.window_size,self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                    batch = torch.from_numpy(batch).float().view(-1, self.args.window_size, self.args.nnodes,self.args.out_channels).to(batch.device)

                score = torch.mean((output.reshape(-1, ae_channels) - batch.reshape(-1, ae_channels)) ** 2, dim=1)  ## (batch_size)

                scores.append(score.detach().cpu().numpy())

                # gt_list.append(batch.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy()[:,-1,:])
                # pred_list.append(output.reshape(-1, self.args.window_size, self.args.nnodes*self.args.in_channels).detach().cpu().numpy()[:,-1,:])

                gt_list.append(batch.reshape(-1, self.args.window_size,
                                              self.args.nnodes * self.args.out_channels).detach().cpu().numpy())
                pred_list.append(output.reshape(-1, self.args.window_size,
                                                self.args.nnodes * self.args.out_channels).detach().cpu().numpy())

        return scores, pred_list, gt_list
