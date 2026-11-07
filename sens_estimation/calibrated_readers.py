import numpy as np
from scipy.interpolate import interp1d
from common.readers import RawSpecim
from common import WAVELENGTHS
from .config import Paths


def get_calibration(bands):
    calibration = np.load(Paths.calibration_ours) 
    calibration = interp1d(WAVELENGTHS, calibration)(bands)
    return calibration


class CalibratedSpecim(RawSpecim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_wavelengths = self.wavelengths
        self.wavelengths = WAVELENGTHS


    def get_calibrated_hsi(self):
        hsi = self.get_exposure_normalized_img()
        calibration = get_calibration(self.wavelengths)
        hsi = interp1d(self._original_wavelengths, hsi)(self.wavelengths)
        return hsi * calibration
