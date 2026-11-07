import logging
import sys
import yaml
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from envparse import env
from functools import lru_cache
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from filters_choosing.config import Paths as FiltersPaths
from filters_choosing.utils import Camera
from .config import Paths
from .metrics import ALL_METRICS

env.read_envfile()

logger = logging.getLogger('IITP')
level = env.str('LOGGING_LEVEL', default='WARNING')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class MetricsLogger:
    def __init__(self, name: str):
        logdir = Paths.logs_dir / name
        logdir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def log_dict(self, epoch: int, metrics_dict: dict):
        for metric, value in metrics_dict.items():
            self.writer.add_scalar(metric, value, epoch)

@lru_cache
def get_device():
    import torch
    n = torch.cuda.device_count()
    i = env('GPU', cast=int, default=0)  # type: ignore
    logger.debug(f'Using GPU #{i} of {n}')
    device = torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


class CheckpointManager:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler | None,
                 monitor_metric: str,
                 config: dict,
                 name: str | None = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.config = config
        self.best_metric = None
        self.name = name if name else model._get_name()
        self.epoch = 0

    def update(self, epoch: int, val_metrics_computed: dict):
        self.epoch = epoch
        last_path = self.last_checkpoint_path(self.name)
        best_path = self.best_checkpoint_path(self.name)
        self.save_state_dict(last_path)
        metric_name = self.monitor_metric.split(':')[-1]
        higher_is_better = ALL_METRICS[metric_name].higher_is_better
        val_metric_value = val_metrics_computed[self.monitor_metric]
        if self.best_metric is None or (
            (val_metric_value > self.best_metric) == bool(higher_is_better)
        ):
            self.best_metric = val_metric_value
            self.save_state_dict(best_path)

    def save_state_dict(self, save_path: Path | str):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "n_epochs": self.epoch + 1,
            "config": self.config,
        }
        torch.save(state_dict, save_path)

    def resume_from(self, checkpoint_path: Path | str):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        epoch = state_dict['n_epochs']
        self.name = self.checkpoint_name_from_path(checkpoint_path)
        return epoch
    
    @staticmethod
    def get_config_and_weights(checkpoint_path: Path | str):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return state_dict['config'], state_dict['model']

    def set_nonexistent_name(self, suffix: str | None = None):
        initial_name = self.name
        if suffix:
            initial_name += '_' + suffix
            self.name = initial_name
        index = 1
        while CheckpointManager.last_checkpoint_path(self.name).exists():
            index += 1
            self.name = f'{initial_name}_{index}'
        logger.info(f'New checkpoint name: {self.name}')

    @staticmethod
    def last_checkpoint_path(name: str):
        return Paths.checkpoints_dir / f'{name}_last.pth'

    @staticmethod
    def best_checkpoint_path(name: str):
        return Paths.checkpoints_dir / f'{name}_best.pth'
    
    @staticmethod
    def checkpoint_name_from_path(checkpoint_path: Path | str):
        return '_'.join(Path(checkpoint_path).stem.split('_')[:-1])


def print_metrics(metrics: dict):
    df = pd.DataFrame(metrics, index=[''])
    print(df)
    print()


@torch.no_grad()
def plot_hsi(hsi: Tensor, ax, n_spectra=1337, title: str | None = None):
    hsi_np = hsi.cpu().numpy()
    n_bands, h, w = hsi_np.shape
    step = int(np.sqrt(h * w / n_spectra))
    spectra = hsi_np[:, ::step, ::step].reshape(n_bands, -1)
    ax.plot(spectra, c='k', alpha=.05)
    ax.grid(True)
    ax.set_ylim(0, None)
    if title:
        ax.set_title(title)


def get_config(config_or_checkpoint_path: Path | str | None = None):
    path = Path(config_or_checkpoint_path or Paths.default_config_path)
    if path.suffix == '.pth':
        return CheckpointManager.get_config_and_weights(path)[0]
    else:
        with open(path) as f:
            return yaml.safe_load(f)


def print_config(config: dict):
    print(yaml.dump(config))


def get_sensitivities(bands, only_main_camera = False,
                      use_filters = True):
    if use_filters:
        filters_path = FiltersPaths.chosen_filters
    else:
        filters_path = FiltersPaths.no_filters 
    with open(filters_path, encoding='utf-8') as f:
        chosen_filters = json.load(f)
        tele_filter = chosen_filters['tele']
        wide_filter = chosen_filters['wide']
    if only_main_camera:
        return Camera(bands).get_main_sensitivity()
    else:
        return Camera(bands).apply_filters(tele_filter, wide_filter)


def seed_everything(seed: int):
    # https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_input(mci: Tensor):
    """
    Normalizes input images per sample mean of 1st image.
    Used solely for training.
    Args:
        mci: (..., K, C, H, W) tensor of input images
    Returns:
        (..., K, C, H, W) tensor of normalized input images
    """
    return mci / mci[:, :1].mean(dim=(-3, -2, -1), keepdim=True)


def normalize_hsi(hsi: Tensor, mask: Tensor | None = None):
    """
    Normalizes HSIs per sample mean.
    Used both for dataset preprocessing and training.
    Args:
        hsi: (..., C, H, W) tensor of HSIs
        mask: (..., 1, H, W) tensor of masks
    Returns:
        (..., C, H, W) tensor of normalized HSIs
    """
    if mask is None:
        mask = torch.ones_like(hsi[..., :1, :, :])
    n_bands = hsi.shape[-3]
    valid_pixels_sum = (hsi * mask).sum(dim=(-3, -2, -1), keepdim=True)
    valid_pixels_count = mask.sum(dim=(-3, -2, -1), keepdim=True) * n_bands
    per_sample_mean = valid_pixels_sum / valid_pixels_count
    return hsi / per_sample_mean
