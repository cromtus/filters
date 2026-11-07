import torch
import h5py
from argparse import ArgumentParser
from tqdm import tqdm
from dataset_composition.images import AlignedDataset
from reconstruction.config import Paths
from reconstruction.utils import get_device, normalize_hsi
from reconstruction.models.optical_flow import OpticalFlow
from reconstruction.models import optical_flow


@torch.no_grad()
def main():
    args = parse_args()
    optical_flow_name: str = args.optical_flow
    batch_size: int = args.batch_size
    device = get_device()
    optical_flow_model = create_optical_flow_model(optical_flow_name)
    output_dir = Paths.warped_gt_root / optical_flow_model._get_name()
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset = AlignedDataset()

    optical_flow_model = optical_flow_model.to(device)
    n = len(dataset)
    rgbs, hsis, names = [], [], []
    with tqdm(total=n, desc='Warping and normalizing the hyperspectral images...') as pbar:
        for i, name in enumerate(dataset.names):
            entry = dataset[name]
            rgbs.append(torch.from_numpy(entry['rgbs']))
            hsis.append(torch.from_numpy(entry['hsi']))
            names.append(name)
            if len(rgbs) == batch_size or i == n - 1:
                rgbs_batch = torch.stack(rgbs).to(device)
                hsi_batch = torch.stack(hsis).to(device)
                hsi_warped_batch, mask_batch = optical_flow_model.warp_hsi_to_main_view(hsi_batch, rgbs_batch[:, 0])
                hsi_warped_batch = normalize_hsi(hsi_warped_batch, mask_batch)
                for hsi_warped, mask, name in zip(hsi_warped_batch, mask_batch, names):
                    output_path = output_dir / f'{name}.h5'
                    image = hsi_warped.cpu().numpy()
                    mask = mask.cpu().numpy()
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('hsi', data=image)
                        f.create_dataset('mask', data=mask)
                    pbar.update()
                rgbs, hsis, names = [], [], []
    print(f'Wrote warped hyperspectral images and masks to {output_dir}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--optical_flow', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def create_optical_flow_model(model_name: str) -> OpticalFlow:
    model_class = getattr(optical_flow, model_name)
    model = model_class()
    model.to(get_device())
    model.eval()
    return model


if __name__ == '__main__':
    main()