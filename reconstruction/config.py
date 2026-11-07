from pathlib import Path
from envparse import env
from dataset_composition.config import Paths as DataPaths


def create_on_demand(path: Path):
    def getter(cls):
        path.mkdir(parents=True, exist_ok=True)
        return path
    return property(getter)


env.read_envfile()
username = env.str('USER', default='unknown_user')


class PathsMeta(type):
    local_unlimited_data_root = Path(__file__).parent.parent / 'artifacts'
    checkpoints_dir = create_on_demand(local_unlimited_data_root / 'checkpoints')
    logs_dir = create_on_demand(local_unlimited_data_root / 'logs')
    results_dir = create_on_demand(Path(__file__).parent / 'results' / username)
    warped_gt_root = DataPaths.dataset_root / 'warped'
    synthetic_dir = create_on_demand(local_unlimited_data_root / 'synthetic')
    default_config_path = Path(__file__).parent / 'config.yaml'
    splits_dir = Path(__file__).parent / 'splits'


class Paths(metaclass=PathsMeta):
    pass
