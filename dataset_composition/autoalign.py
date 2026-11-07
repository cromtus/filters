import numpy as np
import cv2
import json
import imageio.v3 as imageio
from tqdm import tqdm
from pathos.multiprocessing import Pool
from argparse import ArgumentParser
from common import scales
from dataset_composition.render import get_smallest_size, center_crop
from dataset_composition.images import MisalignedDataset
from dataset_composition.align import Aligner, ALIGNMENT_DEPENDENCY
from dataset_composition.config import Paths


def main():
    args = parse_args()
    aligner = Aligner()
    misaligned_dataset = MisalignedDataset()
    with open(Paths.approved_indices_path) as f:
        approved_indices = json.load(f)
    if args.all:
        indices_to_match = range(len(misaligned_dataset))
    else:
        indices_to_match = [i for i in range(len(misaligned_dataset))
                            if i not in approved_indices]

    def align(i: int):
        images = misaligned_dataset[i]
        grayscale_images = {camera: imageio.imread(Paths.get_grayscale_path(i, camera))
                            for camera in images}
        grayscale_images = {
            camera: cv2.resize(
                grayscale_images[camera],
                None,
                fx=scales[camera] / max(scales.values()),
                fy=scales[camera] / max(scales.values()),
                interpolation=cv2.INTER_AREA)
            for camera in grayscale_images
        }
        homographies = {}
        for source, target in ALIGNMENT_DEPENDENCY:
            # Resize & crop images for consistency
            scale_source = scales[source] / max(scales.values())
            scale_target = scales[target] / max(scales.values())
            source_base = grayscale_images[source]
            target_base = grayscale_images[target]
            h, w = get_smallest_size([source_base.shape, target_base.shape])
            source_base, (sly, slx) = center_crop(source_base, h, w)
            target_base, (tly, tlx) = center_crop(target_base, h, w)

            # Compute homography between base images
            try:
                H = compute_homography_sift(source_base, target_base)
            except:
                H = compute_homography_fft(source_base, target_base)

            # Compute homography between original images
            Ss = transform_scale(scale_source)
            St = transform_scale(scale_target)
            Ts = transform_translate(-slx, -sly)
            Tt = transform_translate(-tlx, -tly)
            H = np.linalg.inv(Tt @ St) @ H @ Ts @ Ss
            homographies[source, target] = H
        aligner.align(i, images, homographies)

    with Pool(args.workers) as p:
        n = len(indices_to_match)
        for _ in tqdm(p.imap(align, indices_to_match), total=n):
            pass


def transform_scale(scale: float):
    return np.diag([scale, scale, 1])


def transform_translate(dx: int, dy: int):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])


def compute_homography_sift(source: np.ndarray, target: np.ndarray):
    sift = cv2.SIFT_create()  # type: ignore
    kp1, des1 = sift.detectAndCompute(source, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Ensure we have enough good matches
    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found")

    matches = good_matches

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # type: ignore
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # type: ignore

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    assert H.shape == (3, 3)

    # Here I abuse the fact that source and target are nearly identically scaled
    assert np.abs(np.sqrt(np.clip(np.linalg.det(H[:2, :2]), 0, None)) - 1) < 0.1  # type: ignore
    
    return H


def compute_homography_fft(source: np.ndarray, target: np.ndarray):
    source = source - source.mean()
    target = target - target.mean()
    source = source / source.std()
    target = target / target.std()
    fft_source = np.fft.fft2(source)
    fft_target = np.fft.fft2(target)
    fft_correlation = np.conj(fft_source) * fft_target
    correlation = np.fft.ifft2(fft_correlation).real
    dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)
    if dx > source.shape[1] / 2:
        dx -= source.shape[1]
    if dy > source.shape[0] / 2:
        dy -= source.shape[0]
    return np.array([[1., 0, dx], [0, 1, dy], [0, 0, 1]])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--all', action='store_true', help='Align all images, not just the ones that are not approved')
    return parser.parse_args()


if __name__ == '__main__':
    main()

