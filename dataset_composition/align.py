import cv2
import numpy as np
import imageio.v3 as imageio
import json
import h5py
from scipy.interpolate import interp1d
from common import scales
from common.colour import gamma_correction, render, srgb_cmfs
from .render import to_grayscale, undistort
from .config import camera_types, Paths
from .images import MisalignedDataset, AlignedDataset


class Aligner:
    def align(self,
              idx: int,
              entry: MisalignedDataset.Entry,
              homographies: dict[tuple[str, str], np.ndarray],
              manual: bool = False):
        color_matrices = {
            camera: (
                entry[camera].rgb2srgb
                if camera != 'hsi'
                else srgb_cmfs(entry['hsi'].wavelengths)
            )
            for camera in entry
        }
        images = MisalignedDataset.photometric_normalization(entry)

        # attach oversaturated masks to images to warp them similarly
        oversaturated_masks = MisalignedDataset.oversaturated_masks(entry)
        images = {
            camera: np.concatenate([
                images[camera],
                oversaturated_masks[camera][:, :, None] * 1.
            ], axis=2)
            for camera in images
        }

        images = undistort(images)
        aligned_images = {}
        compose = HomographyComposer(homographies)
        target_camera = 'tele'
        for source_camera in images:
            source_img = images[source_camera]
            target_img = images[target_camera]
            H = compose.get_homography(source_camera, target_camera)

            # make all images have the same (small) angular resolution
            max_scale = max(scales.values())
            scale = scales[target_camera] / max_scale
            crop_h, crop_w = (int(target_img.shape[0] * scale), int(target_img.shape[1] * scale))
            crop_h = crop_h // 64 * 64
            crop_w = crop_w // 64 * 64
            H = np.diag([scale, scale, 1]) @ H
            scale, H = self.decompose_homography(H)

            result = cv2.resize(
                cv2.warpPerspective(source_img, H, (source_img.shape[1], source_img.shape[0])),
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )[:crop_h, :crop_w]
            # it's bad it happened but we don't need uncaught exceptions
            if result.shape[:2] != (crop_h, crop_w):
                t = np.zeros((crop_h, crop_w, result.shape[2]), dtype=np.float32)
                t[:result.shape[0], :result.shape[1]] = result
                result = t
            aligned_images[source_camera] = result
        
        oversaturated = {
            camera: (aligned_images[camera][:, :, -1] > 0).mean()
            for camera in aligned_images
        }

        # detach oversaturated masks
        aligned_images = {
            camera: aligned_images[camera][:, :, :-1]
            for camera in aligned_images
        }

        self.save_aligned_images(idx, aligned_images, color_matrices)
        self.save_preview_animation(idx, list(aligned_images.values()))
        self.save_homographies(idx, homographies, manual)
        self.save_renders(idx, aligned_images, color_matrices, oversaturated)
    
    @staticmethod
    def decompose_homography(H: np.ndarray):
        '''
        Decompose homography into scale and another homography.
        '''
        scale = np.abs(np.linalg.det(H[:2, :2])) ** 0.5
        scale = np.clip(scale, 0, 1)
        scale = 1 / np.round(1 / scale)
        S = np.diag([scale, scale, 1])
        H = np.linalg.inv(S) @ H
        return scale, H

    @staticmethod
    def save_aligned_images(i: int, aligned_images: dict[str, np.ndarray], color_matrices: dict[str, np.ndarray]):
        rgbs = np.stack([
            aligned_images[camera_type]
            for camera_type in camera_types
            if camera_type in aligned_images and camera_type != 'hsi'
        ])
        color_matrices_data = np.stack([
            color_matrices[camera] 
            for camera in camera_types if camera != 'hsi'
        ]).astype(np.float32)
        hsi = aligned_images['hsi']
        input_wavelengths = MisalignedDataset().wavelengths
        output_wavelengths = AlignedDataset.bands
        hsi = interp1d(input_wavelengths, hsi)(output_wavelengths)
        rgbs = rgbs.transpose(0, 3, 1, 2).astype(np.float32)
        hsi = hsi.transpose(2, 0, 1).astype(np.float32)
        with h5py.File(Paths.aligned_dir / f'{i:03}.h5', 'w') as f:
            f.create_dataset('rgbs', data=rgbs)
            f.create_dataset('color_matrices', data=color_matrices_data)
            f.create_dataset('hsi', data=hsi)

    @staticmethod
    def save_preview_animation(i: int, aligned_images: list[np.ndarray]):
        min_size = min(map(np.shape, aligned_images))[:2]
        def normalize(img):
            img = cv2.resize(img, min_size[::-1])
            q = np.quantile(img, 0.99)
            if q == 0:
                q = max(1e-6, np.max(img))
            img = img / q
            img = np.clip(img, 0, 1)
            img = gamma_correction(img)
            return img
        aligned_grayscale = list(map(normalize, map(to_grayscale, aligned_images)))
        imageio.imwrite(
            Paths.get_alignment_animation_path(i),
            (np.array(aligned_grayscale) * 255).astype(np.uint8)
        )
    
    @staticmethod
    def save_renders(i: int, aligned_images: dict[str, np.ndarray], color_matrices: dict[str, np.ndarray], oversaturated: dict[str, float]):
        for camera, img in aligned_images.items():
            img = render(img, color_matrices[camera])
            path = Paths.get_rendered_aligned_path(i, camera)
            imageio.imwrite(path, img)

    @staticmethod
    def save_homographies(i: int, homographies: dict[tuple[str, str], np.ndarray], manual: bool = False):
        with open(Paths.get_homographies_path(i), 'w') as f:
            output = {
                f'{source}->{target}': H.tolist()
                for (source, target), H in homographies.items()
            }
            if manual:
                output['manual'] = True
            json.dump(output, f, indent=4)

    @staticmethod
    def load_homographies(idx: int):
        with open(Paths.get_homographies_path(idx)) as f:
            return {
                tuple(pair.split('->')): np.array(H)
                for pair, H in json.load(f).items()
                if '->' in pair
            }
    
    @staticmethod
    def is_manually_aligned(idx: int):
        with open(Paths.get_homographies_path(idx)) as f:
            return 'manual' in json.load(f)


ALIGNMENT_DEPENDENCY = [
    # (source, target)
    ('hsi', 'tele'),
    ('main', 'hsi'),
    ('wide', 'hsi'),
]


class HomographyComposer:
    def __init__(self, homographies: dict[tuple[str, str], np.ndarray]):
        self.homographies = homographies

    def get_homography(self, source: str, target: str):
        if source == target:
            return np.eye(3)
        if (source, target) in self.homographies:
            return self.homographies[(source, target)]
        if (target, source) in self.homographies:
            return np.linalg.inv(self.homographies[(target, source)])
        # other homographies are performed through hsi
        return self.get_homography('hsi', target) @ self.get_homography(source, 'hsi')
