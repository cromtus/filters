from torch import nn, Tensor
from ._interface import Pipeline


class Vanilla(Pipeline):
    def __init__(self, spectral_rec_model: nn.Module, n_cameras: int, **kwargs):
        super().__init__(n_cameras)
        self.spectral_rec_model = spectral_rec_model

    def _forward_train(self, rgbs: Tensor, hsi: Tensor, mask: Tensor | None) -> Tensor:
        rgbs = rgbs.flatten(1, 2)  # (B, N, C, H, W) -> (B, N*C, H, W)
        pred = self.spectral_rec_model(rgbs)
        loss = self.masked_l1_loss(pred, hsi, mask)
        return loss

    def _forward_test(self, rgbs: Tensor) -> Tensor:
        rgbs = rgbs.flatten(1, 2)  # (B, N, C, H, W) -> (B, N*C, H, W)
        return self.spectral_rec_model(rgbs)

    def _get_name(self):
        return self.spectral_rec_model._get_name()
