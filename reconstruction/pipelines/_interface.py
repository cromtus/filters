import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.nn.functional import l1_loss
from abc import ABC, abstractmethod
from tqdm import tqdm
from ..utils import normalize_input, normalize_hsi


class Pipeline(ABC, nn.Module):
    def __init__(self, n_cameras: int):
        super().__init__()
        self.register_buffer('camera_gains', torch.ones(n_cameras))
    
    def fit_camera_gains(self, loader: DataLoader) -> Tensor:
        # rgbs.shape = (B, N, C, H, W)
        total_relative_means = torch.zeros_like(self.camera_gains)
        total_samples = 0
        device = self.camera_gains.device
        for batch in tqdm(loader, desc='Fitting camera gains'):
            rgbs = batch['rgbs'].to(device)
            per_camera_means = rgbs.mean(dim=(2, 3, 4))
            reference_mean = per_camera_means[:, :1]
            relative_means = per_camera_means / reference_mean
            total_relative_means += relative_means.sum(dim=0)
            total_samples += rgbs.shape[0]
        self.camera_gains.copy_(total_samples / total_relative_means)

    def forward_train(self, rgbs: Tensor, hsi: Tensor, mask: Tensor | None) -> Tensor:
        rgbs = rgbs * self.camera_gains[None, :, None, None, None]
        rgbs = normalize_input(rgbs)
        hsi = normalize_hsi(hsi, mask)
        return self._forward_train(rgbs, hsi, mask)
    
    def forward_test(self, rgbs: Tensor) -> Tensor:
        rgbs = rgbs * self.camera_gains[None, :, None, None, None]
        rgbs = normalize_input(rgbs)
        return self._forward_test(rgbs)

    @abstractmethod
    def _forward_train(self, rgbs: Tensor, hsi_gt: Tensor, mask: Tensor | None) -> Tensor:
        """
        Inputs:
        - normalized RGB images (B, N, C, H, W)
        - ground truth hyperspectral image (B, K, H, W)
        - mask of GT's valid pixels (B, 1, H, W)
        Outputs: loss
        """
        ...

    @abstractmethod
    def _forward_test(self, rgbs: Tensor) -> Tensor:
        """
        Inputs: normalized RGB images (B, N, C, H, W)
        Outputs: predicted hyperspectral image (B, K, H, W)
        """
        ...

    def forward(self, rgbs: Tensor):
        return self.forward_test(rgbs)

    def masked_l1_loss(self, pred: Tensor, gt: Tensor, mask: Tensor | None):
        if mask is not None:
            pred = pred * mask
            gt = gt * mask
        return l1_loss(pred, gt)
