from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    local_data_root = Path(__file__).parent / 'data'
    sensitivities = {
        'main': local_data_root / 'sensitivities-main.csv',
        'tele': local_data_root / 'sensitivities-tele.csv',
        'wide': local_data_root / 'sensitivities-wide.csv',
    }
    calibration_ours = local_data_root / 'xrite-to-specim2.npy'
