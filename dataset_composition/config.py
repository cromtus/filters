from common import CAMERA_TYPES
from dataclasses import dataclass
from pathlib import Path
from envparse import env


env.read_envfile()

camera_types = ['hsi', *CAMERA_TYPES]


@dataclass
class Paths:
    local_data_root = Path(__file__).parent / 'data'
    dataset_root = Path(env('DOOMER_DATASET'))  # type: ignore

    undistortion_dir = dataset_root / 'undistortion'
    chessboard_photos = {
        'hsi': sorted((undistortion_dir / 'hsi').glob('*')),
        'main': sorted((undistortion_dir / 'main').glob('*.dng')),
    }
    distortion_params = local_data_root / 'distortion_params.json'
    tele_vign_map = dataset_root / 'vignetting' / 'tele-vign-ntc.npy'
    rendered_dir = dataset_root / 'rendered'

    markup_files = sorted((dataset_root / 'markup').glob('*.json'))

    preprocessed_dir = dataset_root / 'preprocessed'
    grayscale_dir = preprocessed_dir / 'grayscale'
    alignment_animation_dir = preprocessed_dir / 'alignment_animation'

    homographies_dir = local_data_root / 'homographies'
    approved_indices_path = local_data_root / 'approved_indices.json'
    manually_aligned_path = local_data_root / 'manually_aligned.json'

    aligned_dir = dataset_root / 'aligned'
    rendered_original_dir = rendered_dir / 'original'
    rendered_aligned_dir = rendered_dir / 'aligned'

    ball_reflectance_path = local_data_root / 'ball_reflectance.npy'

    @classmethod
    def get_grayscale_path(cls, idx: int, camera: str) -> Path:
        return cls.grayscale_dir / f'{idx:03}_{camera}.jpg'

    @classmethod
    def get_homographies_path(cls, idx: int) -> Path:
        return cls.homographies_dir / f'{idx:03}.json'

    @classmethod
    def get_alignment_animation_path(cls, idx: int) -> Path:
        return cls.alignment_animation_dir / f'aligned{idx:03}.png'

    @classmethod
    def get_rendered_original_path(cls, idx: int, camera: str, small=False) -> Path:
        path = cls.rendered_original_dir / f'{idx:03}_{camera}.jpg'
        if small:
            path = path.with_suffix('.small.jpg')
        return path

    @classmethod
    def get_rendered_aligned_path(cls, idx: int, camera: str) -> Path:
        return cls.rendered_aligned_dir / f'{idx:03}_{camera}.jpg'
