"""
Train and eval functions used in main.py
"""
import math
from itertools import islice
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


class _BatchLimit:
    """Expose an exact prefix without fetching or training an extra batch."""

    def __init__(self, loader, max_batches):
        if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
            raise ValueError('max_batches must be a positive integer or None')
        self.loader = loader
        self.count = len(loader) if max_batches is None else min(len(loader), max_batches)
        if self.count == 0:
            raise ValueError('training loader must not be empty')

    def __len__(self):
        return self.count

    def __iter__(self):
        return islice(self.loader, self.count)

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False, *, checkpoint_writer, max_batches=None):
    data_loader = _BatchLimit(data_loader, max_batches)
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for batch_index, (samples, targets) in enumerate(metric_logger.log_every(
            data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        nonfinite = torch.tensor(int(not math.isfinite(loss_value)), device=device)
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(nonfinite, op=torch.distributed.ReduceOp.MAX)
        if nonfinite.item():
            rank_losses = [{'rank': utils.get_rank(), 'loss': str(loss_value)}]
            if utils.is_dist_avail_and_initialized():
                gathered = [None] * utils.get_world_size()
                torch.distributed.all_gather_object(gathered, rank_losses[0])
                rank_losses = gathered
            checkpoint_writer.save_error(epoch, batch_index, rank_losses)
            # Do not let another rank exit and cause the launcher to kill the writer early.
            if utils.is_dist_avail_and_initialized():
                torch.distributed.barrier()
            raise FloatingPointError(
                f'Non-finite training loss at epoch={epoch}, batch_index={batch_index}: {rank_losses}')

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            'batches': len(data_loader)}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
