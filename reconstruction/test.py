import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from common.colour import srgb_cmfs
from reconstruction.utils import CheckpointManager, print_metrics, print_config as print_config_impl, get_device
from reconstruction.train import create_data_module, create_pipeline, get_model_summary, print_model_summary, get_test_input_size, save_results
from reconstruction.loops import vanilla_test_loop


def main():
    args = parse_args()
    pipeline, data_module = create_pipeline_and_data_module(args.checkpoint, print_config=True, swap=args.swap_data)
    pipeline.eval()

    test_dataloader = data_module.test_dataloader()
    model_summary = get_model_summary(pipeline, get_test_input_size(test_dataloader))
    print_model_summary(model_summary)
    test_metrics = vanilla_test_loop(pipeline, test_dataloader)
    print_metrics(test_metrics)
    if args.save:
        config_dict, _ = CheckpointManager.get_config_and_weights(args.checkpoint)
        name = CheckpointManager.checkpoint_name_from_path(args.checkpoint)
        save_results(test_metrics,
                     model_summary,
                     args.checkpoint.resolve(),
                     config_dict,
                     name)


def create_pipeline_and_data_module(checkpoint_path: Path | str, print_config: bool = False, swap: bool = False):
    config, weights = CheckpointManager.get_config_and_weights(checkpoint_path)
    if print_config:
        print_config_impl(config)
    data_module = create_data_module(config)
    pipeline = create_pipeline(config, data_module.color_channels, data_module.hyperspectral_channels)
    pipeline.load_state_dict(weights)
    return pipeline, data_module


def get_rgb_sens(bands: np.ndarray) -> torch.Tensor:
    device = get_device()
    return torch.tensor(srgb_cmfs(bands), device=device, dtype=torch.float32)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--swap-data', action='store_true',
                        help='To validate synthetic train on real test and vice versa')
    parser.add_argument('--save-rec', action='store_true',
                        help='Whether to save a reconstructed HSIs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
