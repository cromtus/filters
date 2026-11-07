import torch
import torchvision.transforms.v2 as transforms
from pathlib import Path
from torchvision.tv_tensors import Image as tv_Image
from typing import Sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from envparse import env
from typing import Callable
from common.readers import H5Dataset
from dataset_composition.images import AlignedDataset
from .utils import logger
from common import CAMERA_TYPES

env.read_envfile()


class DataModule:
    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            patch_size: int,
            split: dict[str, Sequence[str]],
            input_cameras: list[str],
    ):
        self.train_names = split["train"]
        self.val_names = split.get("val")
        self.test_names = split["test"]
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.dataset = dataset
        self.cameras_num = len(input_cameras)
        self.patch_size = patch_size

        self.input_idxs = []
        for camera_name in CAMERA_TYPES:
            self.input_idxs.append(camera_name in input_cameras)

    def leave_certain_cameras(self, entry: dict[str, tv_Image]):
        return {**entry, 'rgbs': tv_Image(entry['rgbs'][self.input_idxs])}

    def get_transform(self, augment: bool):
        transform: list = []
        transform.append(self.leave_certain_cameras)
        if augment:
            transform += [
                transforms.RandomCrop(self.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        return transforms.Compose(transform)

    def fit_dataloader(self):
        return self.get_dataloader(self.train_names, augment=False)

    def train_dataloader(self):
        return self.get_dataloader(self.train_names, augment=True)

    def val_dataloader(self):
        if self.val_names is None:
            return self.test_dataloader()
        return self.get_dataloader(self.val_names, augment=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_names, augment=False)

    def get_dataloader(self, names: Sequence[str], augment: bool):
        transform = self.get_transform(augment)
        dataset = TransformedDataset(self.dataset, names, transform)
        env_var = 'TRAIN_WORKERS' if augment else 'TEST_WORKERS'
        n_workers: int = env.int(env_var, default=8)
        logger.debug(f'{env_var}={n_workers}')
        if augment:
            return DataLoader(dataset, self.batch_size, num_workers=n_workers, shuffle=True, pin_memory=True)
        else:
            batch_size: int = env.int('TEST_BATCH_SIZE', default=self.batch_size)
            return DataLoader(dataset, batch_size, num_workers=n_workers, shuffle=False)

    @property
    def color_channels(self) -> int:
        return 3 * self.cameras_num

    @property
    def hyperspectral_channels(self) -> int:
        return self.dataset[0]['hsi'].shape[0]
    

class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, names: Sequence[str], transform: Callable):
        self.dataset = dataset
        self.transform = transform
        self.names = names

    def __getitem__(self, index: int):
        return self.transform(self.dataset[self.names[index]])

    def __len__(self):
        return len(self.names)


class RealDataset:
    def __init__(self, warped_gt_dir: Path):
        self.aligned_dataset = AlignedDataset()
        self.warped_hyperspectral = H5Dataset(warped_gt_dir)
        self.preloaded_entries = {}
        self.names = self.aligned_dataset.names

        if env.bool('PRELOAD_DATASETS', default=False):
            self.preload()
            print('Dataset has been cashed to RAM')
    
    def __getitem__(self, name: str):
        if name in self.preloaded_entries:
            return dict(self.preloaded_entries[name])
        rgbs = torch.from_numpy(self.aligned_dataset[name]['rgbs'])
        warped_entry = self.warped_hyperspectral[name]
        warped_hsi = torch.from_numpy(warped_entry['hsi'])
        mask = torch.from_numpy(warped_entry['mask'])
        return {
            'rgbs': tv_Image(rgbs),
            'hsi': tv_Image(warped_hsi),
            'mask': tv_Image(mask),
        }
    
    def preload(self):
        for name in self.names:
            self.preloaded_entries[name] = self[name]

    def __len__(self):
        return len(self.names)


class H5DatasetWrapper(Dataset):
    def __init__(self, h5_dataset: H5Dataset):
        self.h5_dataset = h5_dataset

    def __getitem__(self, index: int):
        return {
            key: tv_Image(torch.from_numpy(value)) if hasattr(value, 'shape') else value
            for key, value in self.h5_dataset[index].items()
        }

    def __len__(self):
        return len(self.h5_dataset)