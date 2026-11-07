import torch
from torch import nn, Tensor
from torchvision.ops import deform_conv2d
from ._interface import Pipeline

from ..models.optical_flow import OpticalFlow


class DCAM(Pipeline):
    def __init__(
            self,
            spectral_rec_model: nn.Module,
            optical_flow_model: OpticalFlow,
            n_cameras: int,
            in_channels: int,
            conv_channels: int,
    ):
        super().__init__(n_cameras)
        self.optical_flow_model = optical_flow_model
        self.spectral_rec_model = spectral_rec_model

        self.convs = nn.ModuleList([
            nn.Conv2d(3, conv_channels - conv_channels // 3 * 2, kernel_size=3, padding=1)
        ] + [
            DeformConv2d(3, conv_channels // 3, kernel_size=3)
            for _ in range(n_cameras - 1)
        ])

    
    @torch.no_grad()
    def get_flows(self, rgbs: Tensor) -> list[Tensor]:
        """
        Get the optical flows between the reference camera and the other cameras.
        Args:
            rgbs: Tensor of shape (B, N, 3, H, W)
        Returns:
            list of N-1 tensors of shape (B, 2, H, W)
        """
        n_cameras = rgbs.shape[1]
        flows = []
        for i in range(1, n_cameras):
            source, target = rgbs[:, 0], rgbs[:, i]
            source, target = self.optical_flow_model.match_colors(source, target)
            flow = self.optical_flow_model(source, target)
            flows.append(flow)
        return flows


    def forward_conv(self, x: Tensor) -> Tensor:
        """
        Forward the input through the deformable convolution alignment module.
        Args:
            x: Tensor of shape (B, N, 3, H, W)
        Returns:
            Tensor of shape (B, N*3, H, W)
        """
        flows = self.get_flows(x)
        n_cameras = x.shape[1]
        outputs = []
        for i in range(n_cameras):
            conv = self.convs[i]
            if type(conv) == DeformConv2d:
                outputs.append(conv(x[:, i], flows[i - 1]))
            else:
                outputs.append(conv(x[:, i]))
        return torch.cat(outputs, dim=1)

    def _forward_train(self, rgbs: Tensor, hsi_gt: Tensor, mask: Tensor) -> Tensor:
        x = self.forward_conv(rgbs)
        pred = self.spectral_rec_model(x)
        loss = self.masked_l1_loss(pred, hsi_gt, mask)
        return loss

    def _forward_test(self, rgbs: Tensor) -> Tensor:
        x = self.forward_conv(rgbs)
        return self.spectral_rec_model(x)

    def _get_name(self):
        return super()._get_name() + '_' + self.spectral_rec_model._get_name()


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DeformConv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        center = kernel_size // 2
        self.weight.data[:, :, center, center] = torch.eye(out_channels, in_channels)
        self.kernel_size = kernel_size

    def forward(self, x, flow):
        offset = flow.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        out = deform_conv2d(x, offset, self.weight, self.bias, padding=(1, 1))
        return out
