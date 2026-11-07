import json
import numpy as np
import cv2
from typing import TypedDict
from functools import lru_cache
from common.readers import RawImage
from functools import cached_property
from .config import Paths, camera_types
from common.readers import H5Dataset
from sens_estimation.calibrated_readers import CalibratedSpecim


class MisalignedDataset:
    Entry = TypedDict('Entry', {
        'hsi': CalibratedSpecim,
        'main': RawImage,
        'wide': RawImage,
        'tele': RawImage,
    })

    def __getitem__(self, index_or_name: int | str) -> Entry:
        images = {}
        for camera_type in camera_types:
            if camera_type == 'hsi':
                filepath = self._get_hsi_path(index_or_name)
                image = CalibratedSpecim.from_directory(filepath)
            else:
                filepath = self._get_image_path(index_or_name, camera_type)
                image = RawImage.from_file(filepath)
            images[camera_type] = image
        return images  # type: ignore

    @staticmethod
    def photometric_normalization(entry: Entry) -> dict[str, np.ndarray]:
        arrays_dict = {}
        for camera in entry:
            if camera == 'hsi':
                image = entry['hsi'].get_calibrated_hsi()
            else:
                image = entry[camera].get_exposure_normalized_img()[::-1, ::-1]
                if camera == 'tele':
                    image /= get_tele_vign_map()
            arrays_dict[camera] = image
        return arrays_dict
    
    @staticmethod
    def oversaturated_masks(entry: Entry) -> dict[str, np.ndarray]:
        return {
            camera: (
                (entry['hsi']._img_int == entry['hsi'].WHITE_LVL).any(axis=2)
                if camera == 'hsi' else
                (entry[camera]._img_int[::-1, ::-1] == entry[camera].white_level_per_channel).any(axis=2)
            )
            for camera in entry
        }


    @cached_property
    def wavelengths(self) -> np.ndarray:
        return CalibratedSpecim.from_directory(self._get_hsi_path(0)).wavelengths

    def _get_hsi_path(self, index_or_name: int | str):
        original_paths = self._get_original_paths(index_or_name)
        key = self.camera_to_path_key['hsi']
        return Paths.dataset_root / original_paths[key]

    def _get_image_path(self, index_or_name: int | str, camera_type: str):
        original_paths = self._get_original_paths(index_or_name)
        key = self.camera_to_path_key[camera_type]
        return Paths.dataset_root / original_paths[key].lstrip('/')

    def _get_original_paths(self, index_or_name: int | str):
        if isinstance(index_or_name, int):
            return self._get_original_paths_by_index(index_or_name)
        for i, file in enumerate(Paths.markup_files):
            if index_or_name == file.stem:
                return self._get_original_paths_by_index(i)
        raise ValueError(f"{index_or_name} not found")

    def _get_original_paths_by_index(self, index: int):
        metadata_file = Paths.markup_files[index]
        with open(metadata_file) as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(Paths.markup_files)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    camera_to_path_key = {
        'hsi': 'HSI_path',
        'main': 'main_path',
        'wide': 'wide_path',
        'tele': 'tele_path',
    }


@lru_cache(maxsize=1)
def get_tele_vign_map():
    img = np.load(Paths.tele_vign_map)
    img = cv2.blur(img, (10, 10))
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    img /= img.max(axis=(0, 1))
    return img


class AlignedDataset(H5Dataset):
    bands = np.linspace(400, 730, 34)

    def __init__(self, names: list[str] | None = None):
        super().__init__(Paths.aligned_dir, names)
