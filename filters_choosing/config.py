from pathlib import Path
from envparse import env

env.read_envfile()

 
class Paths:
    local_data_root = Path(__file__).parent / 'data'
    radiances = local_data_root / 'radiances.npz'
    filters = local_data_root / 'filters.csv'
    conditional_variance_list = local_data_root / 'conditional_variance_list.csv'
    daylight_spectrum_csv = local_data_root / 'daylight.csv'
    noise_plot = local_data_root / 'noise_plot.png'
    noise_params = local_data_root / 'noise_params.json'
    chosen_filters = local_data_root / 'chosen_filters.json'
    no_filters = local_data_root / 'no_filters.json'
