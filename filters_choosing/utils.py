import json
import numpy as np
import pandas as pd
import torch
from common.readers import KaustDataset
from scipy.interpolate import interp1d
from common import CAMERA_TYPES
from sens_estimation.config import Paths as SensPaths
from .config import Paths
from .noise import Noise


class Camera:
    def __init__(self,
                 wavelengths: np.ndarray,
                 sensitivities: torch.Tensor | None = None,
                 filters: torch.Tensor | None = None):
        self.wavelengths = wavelengths
        if filters is None:
            self.filters = self.read_filters()
        else:
            self.filters = filters
        if sensitivities is None:
            sensitivities_np = []
            for camera_type in CAMERA_TYPES:
                sensitivities_np.append(self.read_sensitivities(camera_type))
            interval = (wavelengths.max() - wavelengths.min()) / len(wavelengths)
            sensitivities_np = np.array(sensitivities_np)
            sensitivities_np *= interval
            self.sensitivities = torch.tensor(sensitivities_np)
        else:
            self.sensitivities = sensitivities

    def read_filters(self):
        filters_df = pd.read_csv(Paths.filters, index_col="wavelength")
        filters = interp1d(filters_df.index, filters_df.values, axis=0)(self.wavelengths).T
        self.filters_names = list(filters_df.columns)
        return torch.tensor(filters)

    def read_sensitivities(self, camera_type: str):
        camera_path = SensPaths.sensitivities[camera_type]
        sensitivities_df = pd.read_csv(camera_path, index_col='wavelength')
        return interp1d(sensitivities_df.index, sensitivities_df.values, axis=0)(self.wavelengths).T
    
    def to(self, dest):
        return Camera(self.wavelengths.copy(),
                      self.sensitivities.to(dest),
                      self.filters.to(dest))

    def apply_filters(self, tele_filter: int | str, wide_filter: int | str):
        tele_idx = self.filters_names.index(tele_filter) if isinstance(tele_filter, str) else tele_filter
        wide_idx = self.filters_names.index(wide_filter) if isinstance(wide_filter, str) else wide_filter
        filters_pair = self.filters[[tele_idx, wide_idx]]
        # Apply filters to tele and wide cameras (excluding main camera)
        filtered_sensitivities = torch.zeros_like(self.sensitivities)
        filtered_sensitivities[1:] = self.sensitivities[1:] * filters_pair[:, None, :]
        filtered_sensitivities[0] = self.sensitivities[0]
        
        # Flatten the result into a 2D array
        return filtered_sensitivities.reshape(-1, len(self.wavelengths))
    
    def get_main_sensitivity(self):
        return self.sensitivities[0]


def get_noise(camera_types: list[str] = CAMERA_TYPES):
    from common import scales
    with open(Paths.noise_params) as f:
        noise_params = json.load(f)
    noise = Noise.concat([
        Noise.from_dict({
            'a': np.array(noise_params[camera_type]['a']) * scales[camera_type]**2,
            'b': np.array(noise_params[camera_type]['b']) * scales[camera_type]**2
        })
        for camera_type in camera_types
    ])
    return noise


class DaylightKaust(KaustDataset):
    def __init__(self):
        super().__init__()
        df = pd.read_csv(Paths.daylight_spectrum_csv, index_col='wavelength')
        self.illumination = interp1d(df.index, df.values.squeeze())(self.bands)

    def __getitem__(self, index_or_name: int | str):
        hsi = super().__getitem__(index_or_name)
        return hsi * self.illumination
