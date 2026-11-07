import numpy as np
import cv2
import json
import imageio.v3 as imageio
import rawpy
from typing import Iterable, Mapping
from tqdm import tqdm
from argparse import ArgumentParser
from pathos.multiprocessing import Pool
from common import scales
from common.colour import srgb_cmfs, render as render_rgb
from dataset_composition.config import Paths
from dataset_composition.images import MisalignedDataset

misaligned_dataset = MisalignedDataset()


def main():
    args = parse_args()
    n = len(misaligned_dataset)
    with Pool(args.workers) as p:
        for _ in tqdm(p.imap(render_rgb_and_grayscale, range(n)), total=n):
            pass


def render_rgb_and_grayscale(i: int):
    # RGB
    images = misaligned_dataset[i]
    cmfs = srgb_cmfs(misaligned_dataset.wavelengths)
    for camera in images:
        if camera == 'hsi':
            hsi = images[camera]
            rgb = render_rgb(hsi, cmfs)
        else:
            with rawpy.imread(str(misaligned_dataset._get_image_path(i, camera))) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=False)
        small_rgb = resize_if_big(rgb)
        imageio.imwrite(Paths.get_rendered_original_path(i, camera, small=True), small_rgb)
        imageio.imwrite(Paths.get_rendered_original_path(i, camera), rgb)


    # Grayscale
    images = {camera_type: to_grayscale(image)
              for camera_type, image in misaligned_dataset[i].items()}
    images = undistort(images)
    # We don't crop anything here.
    # We only use the data from the central patch
    # to adjust brightness for the whole image
    resized_sizes = {camera_type: (img.shape[0] * scales[camera_type], img.shape[1] * scales[camera_type])
                     for camera_type, img in images.items()}
    smallest_h, smallest_w = get_smallest_size(resized_sizes.values())

    for camera_type, img in images.items():
        crop_h, crop_w = int(smallest_h / scales[camera_type]), int(smallest_w / scales[camera_type])
        img = img / np.quantile(center_crop(img, crop_h, crop_w)[0], 0.99)
        img = render_rgb(img, normalization='none') # only does gamma correction and uint8
        imageio.imwrite(Paths.get_grayscale_path(i, camera_type), img)


def to_grayscale(img):
    n_channels = img.shape[2]
    if n_channels > 3:
        wavelengths = misaligned_dataset.wavelengths
        cmfs = srgb_cmfs(wavelengths)
        return img @ cmfs[:, 0]
    return img[:, :, 0]


def undistort(images: Mapping[str, np.ndarray]):
    with open(Paths.distortion_params) as f:
        distortion_params = json.load(f)
    undistorted_images = {}
    for camera_type in images:
        if camera_type not in distortion_params:
            undistorted_images[camera_type] = images[camera_type]
        else:
            params = distortion_params[camera_type]
            for key in params:
                params[key] = np.array(params[key])
            undistorted_images[camera_type] = undistort_one(images[camera_type], **params)
    return undistorted_images


def undistort_one(img: np.ndarray, 
              mtx: np.ndarray,
              dist: np.ndarray,
              newcameramtx: np.ndarray,
              roi: tuple[int, int, int, int]):
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


def get_smallest_size(shapes: Iterable[tuple[int, int]]) -> tuple[int, int]:
    crop_h = min([shape[0] for shape in shapes]) // 2 * 2
    crop_w = min([shape[1] for shape in shapes]) // 2 * 2
    return crop_h, crop_w


def center_crop(img: np.ndarray, crop_h: int, crop_w: int):
    h, w = img.shape[:2]
    ly, lx = h//2-crop_h//2, w//2-crop_w//2
    return img[ly:ly+crop_h, lx:lx+crop_w], (ly, lx)


def resize_if_big(img: np.ndarray, max_width: int = 512):
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    main()
