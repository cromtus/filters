import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import Literal
from .metrics import ALL_METRICS
from .data import DataModule
from .utils import MetricsLogger, CheckpointManager, get_device, normalize_hsi
from .pipelines import Pipeline


def vanilla_train(pipeline: Pipeline,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler | None,
                  checkpoint_manager: CheckpointManager | None,
                  metrics_logger: MetricsLogger,
                  data_module: DataModule,
                  start_epoch: int,
                  n_epochs: int,
                  save_every_n_epochs: int,
                  sched_step_each_iter: bool):
    pipeline.fit_camera_gains(data_module.fit_dataloader())

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    metrics_bar = tqdm(position=2, bar_format='val metrics{postfix}')
    for epoch in trange(start_epoch, n_epochs,
                        initial=start_epoch, total=n_epochs,
                        desc='epoch', position=0):
        pipeline.train()
        vanilla_train_loop(pipeline, train_loader, optimizer, scheduler, sched_step_each_iter, metrics_logger, epoch)
        if scheduler and not sched_step_each_iter:
            scheduler.step()

        if (epoch + 1) % save_every_n_epochs == 0:
            pipeline.eval()
            val_metrics = vanilla_val_loop(pipeline, val_loader)
            metrics_bar.set_postfix(**val_metrics)
            metrics_logger.log_dict(epoch, val_metrics)
            if checkpoint_manager:
                checkpoint_manager.update(epoch, val_metrics)


def vanilla_train_loop(pipeline: Pipeline,
                       dataloader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       scheduler,
                       sched_step_each_iter: bool,
                       metrics_logger, epoch):
    device = get_device()
    n_batches = len(dataloader)
    for i, entry in enumerate(tqdm(dataloader, desc='train batch', position=1, leave=False)):
        rgbs = entry['rgbs']
        gt = entry['hsi']
        mask = entry.get('mask')
        if mask is not None:
            mask = mask.to(device)

        rgbs = rgbs.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        loss = pipeline.forward_train(rgbs, gt, mask)
        log_dict = {'Loss': loss.item()}
        metrics_logger.log_dict(epoch * n_batches + i, log_dict)
        loss.backward()
        optimizer.step()
        if sched_step_each_iter and scheduler:
            scheduler.step()


@torch.no_grad()
def _vanilla_val_loop(pipeline: Pipeline,
                      dataloader: DataLoader,
                      stage: Literal['val', 'test']):
    device = get_device()
    metrics = ALL_METRICS.clone().to(device)

    for entry in tqdm(dataloader, desc=f'{stage} batch', position=1, leave=False):
        rgbs = entry['rgbs']
        hsi_gt = entry['hsi']
        mask = entry.get('mask')

        rgbs = rgbs.to(device).to(torch.float32)
        if mask is not None:
            mask = mask.to(device)
        hsi_gt = hsi_gt.to(device)
        # Match training targets (and fix older warped GT saved with mean≈1/C).
        hsi_gt = normalize_hsi(hsi_gt, mask)

        hsi_pred = pipeline.forward_test(rgbs)
        hsi_pred = hsi_pred.contiguous()  # metrics expect contiguous tensors
        hsi_pred = torch.clip(hsi_pred, 0, None)

        metrics.update(hsi_pred, hsi_gt, mask)
    computed = metrics.compute()
    return {k: v.item() for k, v in computed.items()}


def vanilla_val_loop(pipeline: Pipeline, dataloader: DataLoader):
    return _vanilla_val_loop(pipeline, dataloader, 'val')


def vanilla_test_loop(pipeline: Pipeline, dataloader: DataLoader):
    return _vanilla_val_loop(pipeline, dataloader, 'test')
