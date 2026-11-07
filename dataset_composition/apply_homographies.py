import json
from tqdm import tqdm
from pathos.multiprocessing import Pool
from argparse import ArgumentParser
from dataset_composition.config import Paths
from dataset_composition.images import MisalignedDataset
from dataset_composition.align import Aligner


def main():
    args = parse_args()
    aligner = Aligner()
    misaligned_dataset = MisalignedDataset()
    with open(Paths.approved_indices_path) as f:
        indices = json.load(f)
    indices = indices[:1]

    def match(i: int):
        images = misaligned_dataset[i]
        homographies = aligner.load_homographies(i)
        manual = Aligner.is_manually_aligned(i)
        aligner.align(i, images, homographies, manual)

    with Pool(args.workers) as p:
        for _ in tqdm(p.imap(match, indices), total=len(indices)):
            pass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    main()

