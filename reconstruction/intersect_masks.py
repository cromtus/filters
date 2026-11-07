import numpy as np
import h5py
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from common.readers import H5Dataset


def main():
    args = parse_args()
    warped_gt_dirs: list[Path] = args.warped_gt_dirs
    readers = [H5Dataset(warped_gt_dir) for warped_gt_dir in warped_gt_dirs]
    output_dirs = [warped_gt_dir.with_suffix('.shared_mask') for warped_gt_dir in warped_gt_dirs]

    print('Creating output directories...')
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(output_dir)
    print('Done')
    print('Intersecting masks...')

    for entries in tqdm(zip(*readers), total=len(readers[0])):
        masks = [entry['mask'] for entry in entries]
        intersected_mask = np.logical_and.reduce(masks)
        for entry, output_dir in zip(entries, output_dirs):
            entry['mask'] = intersected_mask
            name = entry['name'] + '.h5'
            with h5py.File(output_dir / name, 'w') as f:
                for key, value in entry.items():
                    f[key] = value


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--warped-gt-dirs', type=Path, nargs='+', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main()
