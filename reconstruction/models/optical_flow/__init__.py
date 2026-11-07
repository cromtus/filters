import torch
import torch.nn.functional as F
from torch import Tensor, nn
from abc import ABC, abstractmethod

from ._pwc.pwc import Network as _PWCNet
from ...utils import get_device
from torchvision.models.optical_flow import raft_small


class OpticalFlow(ABC, nn.Module):
    @abstractmethod
    def _compute_flow(self, source: Tensor, target: Tensor) -> Tensor:
        pass

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        flow = self._compute_flow(source, target)
        b, c, h, w = source.shape
        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
        return flow

    def warp_hsi_to_main_view(self, hsi: Tensor, main: Tensor):
        hsi_normalized, main_normalized = self.match_colors(hsi, main)
        hsi_warped, mask = self.warp(hsi_normalized, main_normalized, hsi)
        return hsi_warped, mask

    def warp_rgbs_to_main_view(self, rgbs: Tensor):
        b, n, c, h, w = rgbs.shape
        mask = torch.ones(b, 1, h, w, device=rgbs.device, dtype=torch.bool)
        rgbs_warped = rgbs.clone()
        for i in range(n):
            main_norm, aux_norm = self.match_colors(rgbs[:, 0], rgbs[:, i])
            aux_warped, aux_mask = self.warp(aux_norm, main_norm, rgbs[:, i], mode='bilinear')
            rgbs_warped[:, i] = aux_warped
            mask &= aux_mask
        return rgbs, mask

    @staticmethod
    def match_colors(source: Tensor, target: Tensor):
        b, l, h, w = source.shape
        source_features = source.view(b, -1, h * w).transpose(1, 2) # (b, n, l)
        target_features = target.view(b, -1, h * w).transpose(1, 2) # (b, n, c)
        eye = torch.eye(l, device=source.device) * source_features.mean(dim=(1, 2), keepdim=True) * 1e3
        sens = torch.linalg.inv(source_features.mT @ source_features + eye) @ source_features.mT @ target_features  # (b, l, c)
        # sens = target_features @ torch.linalg.pinv(source_features)  # (b, l, c)
        source = torch.einsum('blhw, blc -> bchw', source, sens)
        source = source / (source.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        target = target / (target.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return source, target

    def warp(self, source: Tensor, target: Tensor, inp: Tensor, mode='bilinear'):
        device = get_device()  # type: ignore 
        flow = self(target, source)
        b, c, h, w = source.shape
        y = torch.arange(h)
        x = torch.arange(w)
        yy, xx = torch.meshgrid(y, x)
        grid = torch.stack([yy, xx], dim=0).to(device)

        grid = grid + flow
        grid = grid.permute(0, 2, 3, 1)
        grid = grid / torch.tensor([h/2, w/2], device=device) - 1
        bound = 1.0 - 1 / min(h, w)
        mask = grid[..., 0] >= -bound
        mask &= grid[..., 0] <= bound
        mask &= grid[..., 1] >= -bound
        mask &= grid[..., 1] <= bound
        grid = grid.flip(dims=[-1])
        warped = F.grid_sample(inp, grid, mode=mode, align_corners=False)
        return warped, mask[:, None]


class PWCNet(OpticalFlow, _PWCNet):
    def _compute_flow(self, source: Tensor, target: Tensor) -> Tensor:
        flow = super(_PWCNet, self).__call__(source, target)
        return flow.flip(dims=[1])


class RAFT(OpticalFlow):
    def __init__(self):
        super(RAFT, self).__init__()
        self.raft = raft_small(pretrained=True)

    def _compute_flow(self, source: Tensor, target: Tensor) -> Tensor:
        flow = self.raft(source, target)[-1]
        return flow.flip(dims=[1])
