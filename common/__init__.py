import numpy as np


CAMERA_TYPES = ["main", "tele", "wide"]
WAVELENGTHS = np.arange(400, 731, 3)
scales = {
    'hsi': 1.45*2,
    'main': 0.5,
    'tele': 0.188,
    'wide': 1
}
