import numpy as np
import rawpy
import h5py
try:
    import pyexifinfo
except ImportError:
    pyexifinfo = None
import pandas as pd
import cv2
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawImage:
    """
    DNG images reader.

    Attributes
    ----------
    `img_int` : numpy array
        Raw data from a sensor. Black current extracted
    `rgb2srgb` : numpy array
        Color transform matrix from camera RGB to Standard RGB
    `white_level_per_channel` : numpy array
        Maximal possible values in `img_int` per channel.
        Pixels of these values are considered saturated.
    """
    _img_int: np.ndarray
    rgb2srgb: np.ndarray
    white_level_per_channel: np.ndarray
    exposure_time: float
    iso: int

    @classmethod
    def from_file(cls, filepath: Path | str, clip=True, quant=1):
        with rawpy.imread(str(filepath)) as raw:
            # debayering
            if (raw.raw_pattern[0] == [0, 1]).all():
                flag = cv2.COLOR_BayerRGGB2RGB
            if (raw.raw_pattern[0] == [1, 0]).all():
                flag = cv2.COLOR_BayerGRBG2RGB
            if (raw.raw_pattern[0] == [2, 3]).all():
                flag = cv2.COLOR_BayerBGGR2RGB
            img_uint8 = cv2.cvtColor(raw.raw_image_visible, flag)

            # black level, saturation level normalization
            black_level = np.array(raw.black_level_per_channel)[:3]
            img_int = img_uint8.astype(np.int32)
            img_int -= black_level
            img_int //= quant
            white_level_per_channel = raw.white_level - black_level
            white_level_per_channel //= quant
            white_level_per_channel -= 4
            color_matrix = (raw.color_matrix * raw.camera_whitebalance).T

            # leave only 3 channels
            color_matrix = color_matrix[:3]
            white_level_per_channel = white_level_per_channel[:3]

            # finalize
            if clip:
                img_int = np.clip(img_int, 0, white_level_per_channel)
            iso, exposure_time = cls.get_iso_and_exposure(filepath)
            return cls(img_int, color_matrix, white_level_per_channel, exposure_time, iso)

    @staticmethod
    def get_iso_and_exposure(filepath: Path | str):
        exif = pyexifinfo.get_json(filepath)
        iso = exif[0]["EXIF:ISO"]
        ss = exif[0]["Composite:ShutterSpeed"]
        if isinstance(ss, str):
            if "/" in ss:
                up, down = ss.split("/")
                return iso, float(up) / float(down)
            else:
                return iso, float(ss)
        else:
            return iso, float(ss)

    
    def get_exposure_normalized_img(self, exposure=True, iso=True):
        img = self._img_int
        if exposure:
            img = img / self.exposure_time
        if iso:
            img = img / self.iso
        return img


@dataclass
class RawSpecim:
    _img_int: np.ndarray
    wavelengths: np.ndarray
    exposure_time: float
    BLACK_LVL = 239
    WHITE_LVL = 4095 - BLACK_LVL

    @classmethod
    def from_directory(cls, directory: Path | str):
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        manifest = pd.read_xml(directory / "manifest.xml", parser="etree")
        header_name = manifest[(manifest["type"] == "raw") & (manifest["extension"] == "hdr")]["file"].values[0]
        path2header = directory / header_name
        header = cls.read_header(path2header)
        wavelengths = np.array(header["wavelength"])

        img_int = np.fromfile(path2header.with_suffix(".raw"), dtype=np.uint16)
        img_int = img_int.reshape(512, 204, 512).transpose(2, 0, 1)[:, ::-1, :]
        img_int = np.clip(img_int - cls.BLACK_LVL, 0, None)
        return cls(img_int, wavelengths, float(header["tint"]))

    @staticmethod
    def read_header(path2header):
        with open(path2header) as header_file:
            lines = header_file.readlines()

        header = {}
        while lines:
            line = lines.pop(0)
            key, sep, value = line.partition("=")

            key = key.strip()
            value = value.strip()
            if not value:
                continue

            if value:
                value = value.strip()
                if value[0] == "{" and value[-1] == "}":
                    value = list(value[1:-1].split(","))
                if key == "wavelength":
                    wavelengths = []
                    wavelength = lines.pop(0)
                    wavelength = wavelength.strip()
                    while wavelength[-1] != "}":
                        wavelengths.append(float(wavelength[:-1]))
                        wavelength = lines.pop(0)
                        wavelength = wavelength.strip()
                    value = wavelengths
            header[key] = value
        return header

    def get_exposure_normalized_img(self):
        return self._img_int / self.exposure_time


@dataclass
class Xrite:
    wavelengths: np.ndarray
    values: np.ndarray

    @classmethod
    def from_file(cls, path: Path):
        data = {'bands': []}
        with open(path) as f:
            lines = f.readlines()

        while lines:
            line = lines.pop(0)
            if "BEGIN_DATA_FORMAT" in line:
                bands = lines.pop(0).split()
                for str_band in bands:
                    _, band = str_band.split("_")
                    data["bands"].append(int(band))
                continue
            if "BEGIN_DATA" in line:
                values = lines.pop(0)
                values = values.split()
                values = list(map(float, values))
                data["values"] = np.clip(values, 0, None)
        return cls(np.array(data['bands']), data['values'])


class H5Dataset:
    def __init__(
            self,
            root_dir: Path,
            names: list[str] | None = None,
            extension: str = '.h5',
    ):
        self.root_dir = root_dir
        if names is not None:
            self.names = names
        else:
            names = [path.stem for path in self.root_dir.glob(f'*{extension}')]
            self.names = sorted(names)
        self.extension = extension

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index_or_name: int | str) -> dict[str, np.ndarray]:
        if isinstance(index_or_name, int):
            name = self.names[index_or_name]
        elif isinstance(index_or_name, str):
            name = index_or_name
        path = self.root_dir / f'{name}{self.extension}'
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist')
        with h5py.File(path, 'r') as f:
            entry = {k: v[()] for k, v in f.items()}
            entry['name'] = name
            return entry

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class KaustDataset:
    def __init__(self):
        raise NotImplementedError
