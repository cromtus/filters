import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm
from pathos.multiprocessing import Pool
from common import CAMERA_TYPES
from common.readers import RawImage
from dataset_composition.config import Paths
from dataset_composition.images import MisalignedDataset


def main():
    precalc_path = Paths.local_data_root / 'exposure_iso_model.npz'
    if not precalc_path.exists():
        exposure_times = []
        isos = []
        means = []
        resolutions = []

        dataset = MisalignedDataset()
        def extract_features(path):
            raw = RawImage.from_file(path)
            return raw.exposure_time, raw.iso, raw.get_exposure_normalized_img().mean(), raw._img_int.shape[:2]

        with Pool(8) as pool:
            paths = [
                dataset._get_image_path(i, camera_type)
                for i in range(len(dataset))
                for camera_type in CAMERA_TYPES
            ]
            for features in tqdm(pool.imap(extract_features, paths), total=len(paths)):
                exposure_times.append(features[0])
                isos.append(features[1])
                means.append(features[2])
                resolutions.append(features[3])

        means = np.array(means)
        exposure_times = np.array(exposure_times)
        isos = np.array(isos)
        np.savez(precalc_path, means=means, exposure_times=exposure_times, isos=isos)
    else:
        data = np.load(precalc_path)
        means = data['means']
        exposure_times = data['exposure_times']
        isos = data['isos']

    print(f'{len(means)} images')
    plt.rcParams.update({'font.size': 16})
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(6, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.01}
    )
    sep_fig, sep_ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for ax in [sep_ax, ax1]:
        ax.scatter(means, exposure_times, alpha=0.2)
        indices = np.argsort(means)
        ax.plot(means[indices], ExposureIsoModel.get_exposure_time(means[indices]), color='red')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect(1)
        ax.grid()
    sep_fig.savefig(Paths.local_data_root / 'exposure_model.pdf')

    sep_fig, sep_ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    for ax in [sep_ax, ax2]:
        ax.scatter(means, isos, alpha=0.2)
        ax.plot(means[indices], ExposureIsoModel.get_iso(means[indices]), color='red')
        ax.set_xlabel('$\\bar{c}_i$')
        ax.set_ylabel('$g$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect(1)
        ax.grid()
    sep_fig.savefig(Paths.local_data_root / 'iso_model.pdf')

    fig.savefig(Paths.local_data_root / 'exposure_iso_model.pdf')


@dataclass
class PiecewisePowerFunction:
    start_x: float
    start_y: float
    end_x: float
    x_nodes: list[float]  # Intermediate x values that split [start_x, end_x]
    powers: list[float]   # Power for each segment
    
    def __post_init__(self):
        # Validate inputs
        if len(self.x_nodes) + 1 != len(self.powers):
            raise ValueError("Number of powers must be equal to number of segments (len(x_nodes) + 1)")
        if not (self.start_x < min(self.x_nodes) and max(self.x_nodes) < self.end_x):
            raise ValueError("x_nodes must be within (start_x, end_x)")
        
        # Create sorted x points including start and end
        self.x_points = [self.start_x] + sorted(self.x_nodes) + [self.end_x]
        
        # Calculate y values at each node
        self.y_points = [self.start_y]
        for i in range(len(self.x_points) - 1):
            left_x, right_x = self.x_points[i], self.x_points[i + 1]
            left_y = self.y_points[-1]
            power = self.powers[i]
            
            # Calculate right_y using power law
            right_y = left_y * (right_x / left_x) ** power
            self.y_points.append(right_y)
            
        # Store segment information for efficient __call__
        self.segments = list(zip(
            self.x_points[:-1],    # left_x
            self.x_points[1:],     # right_x
            self.y_points[:-1],    # left_y
            self.y_points[1:],     # right_y
            self.powers            # power
        ))

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            result = torch.zeros_like(x)
        else:
            result = np.zeros_like(x)
        
        # Handle values outside the range
        result[x < self.start_x] = self.start_y
        result[x > self.end_x] = self.y_points[-1]
        
        # Process each segment
        for left_x, right_x, left_y, right_y, power in self.segments:
            mask = (left_x <= x) & (x <= right_x)
            result[mask] = left_y * (x[mask] / left_x) ** power
            
        return result


class ExposureIsoModel:
    x = 1.2, 7, 600, 100_000
    p1 = -0.45
    exposure_model = PiecewisePowerFunction(
        start_x=x[1],
        start_y=1/17,
        end_x=x[3],
        x_nodes=[x[2]],
        powers=[p1, -1]
    )
    iso_model = PiecewisePowerFunction(
        start_x=x[0],
        start_y=3500,
        end_x=x[2],
        x_nodes=[x[1]],
        powers=[-1, -1-p1]
    )
    @classmethod
    def get_exposure_time(cls, intensity):
        return cls.exposure_model(intensity)

    @classmethod
    def get_iso(cls, intensity):
        return cls.iso_model(intensity)


if __name__ == '__main__':
    main()
