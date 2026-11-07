"""Pre-compute synthesized RGBs (+ mean-normalized HSI) for a hyperspectral dataset.

Uses the same hyperspectral_dataset config keys as train.py:
  name, input_cameras, use_filters, camera_mapping, synthetic
"""

from __future__ import annotations

import h5py
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from einops import einsum
from dataset_composition.exposure_iso_model import ExposureIsoModel
from filters_choosing.noise import Noise
from filters_choosing.utils import get_noise
from reconstruction.config import Paths
from reconstruction.train import get_hyperspectral_dataset
from reconstruction.utils import (
    get_config,
    get_device,
    get_sensitivities,
    normalize_hsi,
    seed_everything,
)
from common import CAMERA_TYPES, scales


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    config_dict = get_config(args.config)

    hs_cfg = config_dict
    dataset_name = hs_cfg['hyperspectral_source']
    input_cameras: list[str] = hs_cfg.get('input_cameras', CAMERA_TYPES)
    input_cameras = sorted(input_cameras, key=lambda x: CAMERA_TYPES.index(x))
    only_main_camera = input_cameras == ['main']
    use_filters = hs_cfg.get('use_filters', True)

    dataset = get_hyperspectral_dataset(dataset_name)
    bands = dataset.bands

    from sens_estimation.config import Paths as SensPaths
    camera_mapping = hs_cfg.get('camera_mapping')
    if camera_mapping is not None:
        for src_cam, dst_cam in camera_mapping.items():
            path = SensPaths.sensitivities[src_cam]
            new_path = path.with_stem(path.stem.replace(src_cam, dst_cam))
            SensPaths.sensitivities[src_cam] = new_path

    device = get_device()
    sens = get_sensitivities(bands, only_main_camera, use_filters).to(device=device, dtype=torch.float32)
    noise = get_noise(input_cameras).to(torch.float32).to(device)

    output_dir = Paths.synthetic_dir / config_dict['name']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Synthesizing RGBs to {output_dir}')
    for i, name in enumerate(tqdm(dataset.names)):
        hsi = torch.as_tensor(dataset[i], device=device, dtype=torch.float32)
        rgbs = synthesize_rgbs(hsi, sens, noise)
        hsi = normalize_hsi(hsi)
        with h5py.File(output_dir / f'{name}.h5', 'w') as f:
            f.create_dataset('rgbs', data=rgbs.cpu().numpy())
            f.create_dataset('hsi', data=hsi.cpu().numpy())

    print(f'Wrote {len(dataset)} files to {output_dir}')


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='YAML config with hyperspectral_dataset section (same as train.py)')
    parser.add_argument('--seed', type=int, default=3407,
                        help='RNG seed (noise is stochastic)')
    return parser.parse_args()


def synthesize_rgbs(hsi: torch.Tensor, sens: torch.Tensor, noise: Noise) -> torch.Tensor:
    """
    Synthesize multi-camera RGBs from an HSI cube.

    Args:
        hsi: (L, H, W) hyperspectral image
        sens: (C, L) camera sensitivities (C = 3 * n_cameras)
        noise: per-channel noise model
    Returns:
        (N, 3, H, W) RGB views
    """
    img = einsum(hsi, sens, 'l h w, c l -> c h w')
    n_cameras = sens.shape[0] // 3
    exposure_times = torch.ones(sens.shape[0], 1, 1, device=hsi.device, dtype=hsi.dtype)
    for i in range(n_cameras):
        subimg = img[i * 3: i * 3 + 3]
        exposure_time = ExposureIsoModel.get_exposure_time(subimg.mean())
        exposure_time *= (max(scales.values()) / scales[CAMERA_TYPES[i]]) ** 2
        exposure_times[i * 3: i * 3 + 3] = exposure_time
    img = img * exposure_times
    img = noise(img)
    img = img / exposure_times
    c, h, w = img.shape
    return img.reshape(c // 3, 3, h, w)


if __name__ == '__main__':
    main()
