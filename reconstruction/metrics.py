import torch
from torch import Tensor
from typing import Callable
from torchmetrics import MetricCollection, Metric


class ErrorMapBasedMetric(Metric):
    sum: Tensor
    total: Tensor

    def __init__(
            self,
            pixelwise_fn: Callable[[Tensor, Tensor], Tensor],
            final_fn: Callable[[Tensor], Tensor] | None = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.pixelwise_fn = pixelwise_fn
        self.final_fn = final_fn
        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor, mask: Tensor | None = None):
        error_map = self.pixelwise_fn(preds, target)
        if mask is not None:
            error_map *= mask[:, 0, ...]
        self.sum += error_map.sum()
        self.total += mask.sum() if mask is not None else error_map.numel()

    def compute(self):
        if self.final_fn is not None:
            return self.final_fn(self.sum / self.total)
        return self.sum / self.total


def pixelwise_spectral_angle_mapper(preds: Tensor, target: Tensor, eps=1e-8):
    products = (preds * target).sum(dim=-3)
    magnitudes = preds.norm(dim=-3) * target.norm(dim=-3) + eps
    return torch.rad2deg(torch.acos(products / magnitudes))


def pixelwise_normalized_spectral_error(preds: Tensor, target: Tensor, eps=1e-8):
    diff = (preds - target).abs().sum(dim=-3)
    scale = target.abs().sum(dim=-3) + eps
    return diff / scale


def pixelwise_mean_absolute_error(preds: Tensor, target: Tensor):
    return (preds - target).abs().mean(dim=-3)


class PeakSignalNoiseRatio(Metric):
    total: Tensor
    sum: Tensor

    def __init__(self):
        super().__init__()
        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor, mask: Tensor | None = None):
        pixelwise_mse = (preds - target).pow(2)
        if mask is not None:
            pixelwise_mse = pixelwise_mse * mask
            denom = mask.expand_as(preds).sum(dim=(1, 2, 3))
            mse = pixelwise_mse.sum(dim=(1, 2, 3)) / denom
            max_val = (target * mask).amax(dim=(1, 2, 3))
        else:
            mse = pixelwise_mse.mean(dim=(1, 2, 3))
            max_val = target.amax(dim=(1, 2, 3))
        self.sum += (10 * torch.log10(max_val ** 2 / mse)).sum()
        self.total += mse.numel()

    def compute(self):
        return self.sum / self.total


ALL_METRICS = MetricCollection({
    'PSNR': PeakSignalNoiseRatio(),
    'SAM': ErrorMapBasedMetric(pixelwise_spectral_angle_mapper),
    'MAE': ErrorMapBasedMetric(pixelwise_mean_absolute_error),
    'NSE': ErrorMapBasedMetric(pixelwise_normalized_spectral_error),
})


def pixelwise_metrics(preds: torch.Tensor, target: torch.Tensor):
    return {
        'SAM': pixelwise_spectral_angle_mapper(preds, target),
        'NSE': pixelwise_normalized_spectral_error(preds, target),
        'MAE': pixelwise_mean_absolute_error(preds, target),
    }
