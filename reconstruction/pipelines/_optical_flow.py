from torch import nn, Tensor
from ._interface import Pipeline
from ..models.optical_flow import OpticalFlow as OpticalFlowModel


class OpticalFlow(Pipeline):
    def __init__(
            self,
            spectral_rec_model: nn.Module,
            optical_flow_model: OpticalFlowModel
    ):
        super(OpticalFlow, self).__init__()
        self.optical_flow_model = optical_flow_model
        self.spectral_rec_model = spectral_rec_model

    def forward_train(self, mci: Tensor, hsi_gt: Tensor, mask: Tensor) -> Tensor:
        mci, mask2 = self.optical_flow_model.warp_rgbs_to_main_view(mci)
        mask = mask & mask2
        pred = self.spectral_rec_model(mci)
        loss = self.masked_l1_loss(pred, hsi_gt, mask)
        return loss

    def forward_test(self, mci: Tensor) -> Tensor:
        mci, _ = self.optical_flow_model.warp_rgbs_to_main_view(mci)
        return self.spectral_rec_model(mci)

    def _get_name(self):
        return super()._get_name() + '_' + self.spectral_rec_model._get_name()
