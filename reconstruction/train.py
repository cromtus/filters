import json
import numpy as np
import torch
import torch.optim as opt
from envparse import env
from copy import deepcopy
from argparse import ArgumentParser
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from thop import profile, clever_format
from common.readers import H5Dataset
from reconstruction import pipelines
from reconstruction.models import spectral_rec
from reconstruction.config import Paths
from reconstruction.data import DataModule, H5DatasetWrapper, RealDataset
from reconstruction.loops import vanilla_train, vanilla_test_loop
from reconstruction.utils import get_device, CheckpointManager, logger, print_metrics, MetricsLogger, get_config, print_config, seed_everything
from reconstruction.pipelines import Pipeline
from reconstruction.models import optical_flow
from common import CAMERA_TYPES


def main():
    args = parse_args()
    seed_everything(args.seed)
    config_dict = get_config(args.config)
    for arg in args.override:
        key, value = parse_override_args(arg)
        update_nested_dict(config_dict, key, value)
    print_config(config_dict)

    data_module = create_data_module(config_dict)
    in_channels = data_module.color_channels
    out_channels = data_module.hyperspectral_channels
    pipeline = create_pipeline(config_dict, in_channels, out_channels)
    input_size = get_test_input_size(data_module.test_dataloader())
    print(input_size)
    model_summary = get_model_summary(deepcopy(pipeline), input_size)
    print_model_summary(model_summary)

    optimizer, scheduler = create_optimizer(pipeline, config_dict, data_module)
    training_params = get_training_params(config_dict)

    checkpoint_manager = CheckpointManager(
        pipeline,
        optimizer,
        scheduler,
        training_params['monitor_metric'],
        config_dict,
        name=get_possible_checkpoint_name(pipeline, config_dict)
    )
    if args.resume_from:
        start_epoch = checkpoint_manager.resume_from(args.resume_from)
        logger.info(f'Resuming from checkpoint {checkpoint_manager.name}, epoch {start_epoch}. '
                    f'The checkpoint will be overwritten')
    else:
        checkpoint_manager.set_nonexistent_name(args.suffix)
        start_epoch = 0
    metrics_logger = MetricsLogger(checkpoint_manager.name)
    sched_step_each_iter = training_params['sched_step_each_iter']

    try:
        vanilla_train(
            pipeline=pipeline,
            optimizer=optimizer,
            scheduler=scheduler,
            data_module=data_module,
            start_epoch=start_epoch,
            n_epochs=training_params['n_epochs'],
            save_every_n_epochs=training_params['save_every_n_epochs'],
            checkpoint_manager=checkpoint_manager,
            metrics_logger=metrics_logger,
            sched_step_each_iter=sched_step_each_iter
        )
    except KeyboardInterrupt:
        logger.info('Training interrupted by user, gracefully shutting down...')
        if checkpoint_manager.epoch == 0:
            logger.info('No validation performed, skipping testing and saving')
            return
    if training_params['checkpoint_selection'] == 'validation_best':
        checkpoint_path = checkpoint_manager.best_checkpoint_path(checkpoint_manager.name)
    else:
        checkpoint_path = checkpoint_manager.last_checkpoint_path(checkpoint_manager.name)
    if checkpoint_path.exists():
        logger.info(f'Testing on the chosen checkpoint {checkpoint_path}')
        checkpoint_manager.resume_from(checkpoint_path)
    test_metrics = vanilla_test_loop(pipeline, data_module.test_dataloader())
    print_metrics(test_metrics)
    save_results(test_metrics,
                 model_summary,
                 checkpoint_path,
                 config_dict,
                 checkpoint_manager.name)


OPTIMIZERS = {
    "Adam": opt.Adam,
}


SCHEDULERS = {
    "PolyLR": opt.lr_scheduler.PolynomialLR,
    "CosineAnnealingLR": opt.lr_scheduler.CosineAnnealingLR,
    "CyclicLR": opt.lr_scheduler.CyclicLR,
    "ExponentialLR": opt.lr_scheduler.ExponentialLR
}


def create_data_module(config_dict: dict) -> DataModule:
    patch_size: int = config_dict['training'].get('patch_size', None)
    batch_size: int = config_dict['training']['batch_size']
    synthetic: bool = config_dict['hyperspectral_dataset'].get('synthetic', False)
    input_cameras: list[str] = config_dict['hyperspectral_dataset'].get('input_cameras', CAMERA_TYPES)
    input_cameras = sorted(input_cameras, key=lambda x: CAMERA_TYPES.index(x))  # "main" will be first
    with open(Paths.splits_dir / config_dict['hyperspectral_dataset']['split']) as f:
        split = json.load(f)

    if synthetic:
        dataset = H5DatasetWrapper(H5Dataset(Paths.synthetic_dir / config_dict['hyperspectral_dataset']['root']))
    else:
        warped_gt_dir = config_dict['warped_gt_dir']
        dataset = RealDataset(Paths.warped_gt_root / warped_gt_dir)
    return DataModule(
        dataset=dataset,
        batch_size=batch_size,
        patch_size=patch_size,
        input_cameras=input_cameras,
        split=split,
    )


def create_spectral_rec_model(config_subdict: dict, in_channels: int, out_channels: int) -> nn.Module:
    model_name = config_subdict['name']
    model_class = getattr(spectral_rec, model_name)
    params = config_subdict.get('params', {})
    model = model_class(in_channels=in_channels, out_channels=out_channels, **params)
    model.to(get_device())
    return model


def create_pipeline(config_dict: dict, in_channels: int, out_channels: int) -> Pipeline:
    config_subdict = config_dict.get('pipeline', {})
    pipeline_name = config_subdict.get('name', 'Vanilla')
    pipeline_name = 'Vanilla' if pipeline_name == 'OpticalFlow' else pipeline_name

    params = {**config_subdict.get('params', {})}
    if pipeline_name in ['Vanilla', 'OpticalFlow', 'DCAM']:
        if 'spectral_rec_model' not in params:
            raise ValueError('spectral_rec_model is required for pipeline ' + pipeline_name)
        spectral_rec_model = create_spectral_rec_model(params['spectral_rec_model'], in_channels, out_channels)
        params['spectral_rec_model'] = spectral_rec_model
    if pipeline_name in ['DCAM', 'OpticalFlow']:
        params['optical_flow_model'] = create_optical_flow_model(params['optical_flow_model'])
    if pipeline_name in ['DCAM']:
        params['in_channels'] = in_channels
        params['conv_channels'] = in_channels
    pipeline_class = getattr(pipelines, pipeline_name)
    pipeline = pipeline_class(n_cameras=in_channels // 3, **params)
    pipeline.to(get_device())
    return pipeline


def create_optical_flow_model(config_subdict: dict) -> optical_flow.OpticalFlow:
    model_name = config_subdict['name']
    model_class = getattr(optical_flow, model_name)
    params = config_subdict.get('params', {})
    model = model_class(**params)
    model.to(get_device())
    model.eval()
    model.requires_grad_(False)
    return model


def create_optimizer(pipeline: Pipeline, config_dict: dict, datamodule: DataModule):
    optimizer_name = config_dict['optimizer']['name']
    optimizer_class = OPTIMIZERS[optimizer_name]
    params = config_dict['optimizer'].get('params', {})
    optimizer = optimizer_class(pipeline.parameters(), **params)
    scheduler = None
    if 'scheduler' in config_dict:
        scheduler_name = config_dict['scheduler']['name']
        scheduler_class = SCHEDULERS[scheduler_name]
        params = config_dict['scheduler'].get('params', {})
        if scheduler_name == 'CosineAnnealingLR':
            params['T_max'] = len(datamodule.train_dataloader()) // 2
        scheduler = scheduler_class(optimizer, **params)
    return optimizer, scheduler



def get_training_params(config_dict: dict):
    n_epochs = config_dict['training']['n_epochs']
    monitor_metric = config_dict['training']['monitor']
    save_every_n_epochs = config_dict['training']['save_every_n_epochs']
    sched_step_each_iter = config_dict['training'].get('sched_step_each_iter', False)
    checkpoint_selection = config_dict['training']['checkpoint_selection']
    return {
        'n_epochs': n_epochs,
        'monitor_metric': monitor_metric,
        'save_every_n_epochs': save_every_n_epochs,
        'sched_step_each_iter': sched_step_each_iter,
        'checkpoint_selection': checkpoint_selection
    }


def get_possible_checkpoint_name(pipeline: Pipeline, config_dict: dict):
    experiment_name = config_dict.get('name')
    if experiment_name is not None:
        return experiment_name

    dataset_name = config_dict['hyperspectral_dataset']['name']
    pipeline_name = pipeline._get_name()
    camera_names = config_dict['hyperspectral_dataset'].get('input_cameras', CAMERA_TYPES)
    camera_abbreviation = ''.join([name[0] for name in camera_names])
    return f'{pipeline_name}_{dataset_name}_{camera_abbreviation}'


def get_hyperspectral_dataset(name: str):
    if name == 'Arad1K':
        return Arad1KDataset()
    raise ValueError(f'Unknown hyperspectral dataset: {name}')


class Arad1KDataset(H5Dataset):
    def __init__(self):
        super().__init__(env('ARAD1K_DATASET', cast=Path), extension='.mat')
        self.bands = np.linspace(400, 700, 31)

    def __getitem__(self, index: int):
        entry = super().__getitem__(index)
        return entry['cube']


def get_model_summary(pipeline: Pipeline, input_size: tuple[int, int, int]):
    device = get_device()
    inp = torch.randn(1, *input_size, device=device)
    macs, _ = profile(pipeline, (inp,), verbose=False)  # type: ignore
    params = 0
    for param in pipeline.parameters():
        if param.requires_grad:
            params += param.numel()
    return {'macs': int(macs), 'params': int(params)}


def print_model_summary(summary: dict):
    macs, params = summary['macs'], summary['params']
    macs, params = clever_format([macs, params], "%.2f")
    print(f'Macs: {macs}, Params: {params}')


def get_test_input_size(data_loader: DataLoader):
    sample = next(iter(data_loader))
    return sample['rgbs'].shape[1:]


def save_results(
        test_metrics: dict,
        model_summary: dict,
        best_checkpoint_path: Path,
        config_dict: dict,
        name: str
):
    path = Paths.results_dir / f'{name}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'model_summary': model_summary,
            'best_checkpoint_path': str(best_checkpoint_path),
            'config': config_dict,
        }, f, indent=4)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--config', type=Path, default=Paths.default_config_path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--override', '-o', action='append', default=[],
                       help='Override config values. Format: key=value or section.key=value')
    return parser.parse_args()


def parse_override_args(arg: str) -> tuple:
    """Parse override argument in format 'key=value' or 'section.key=value'."""
    if '=' not in arg:
        raise ValueError("Override arguments must be in format 'key=value' or 'section.key=value'")
    key, value = arg.split('=', 1)
    return key, value


def update_nested_dict(d: dict, key_path: str, value: str):
    """Update nested dictionary using dot notation (e.g., 'section.key')."""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    # Try to convert value to appropriate type (int, float, bool)
    try:
        value = eval(value)  # This will convert strings to appropriate types
    except:
        pass  # Keep as string if conversion fails
    current[keys[-1]] = value


def dict_to_str(d: dict) -> str:
    return ','.join([
        f'{k}={",".join(v)}' if isinstance(v, list) else (
            f'{k}=({dict_to_str(v)})' if isinstance(v, dict) else (
                f'{k}={v}'
            )
        )
        for k, v in d.items()
        if not (isinstance(v, dict) and len(v) == 0)
    ])


if __name__ == '__main__':
    main()
