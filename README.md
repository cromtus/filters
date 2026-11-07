# Filter-Enhanced Smartphone Hyperspectral Imaging
## Setup
The project requires Python 3.10.

Clone this repository and run the following in its root:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

[Download the Doomer dataset](https://drive.google.com/drive/folders/1CuurbUYKTRTS4H1mNL2ex2qUKkGaL3Yp?usp=sharing) anywhere in your system.
Replace "`/path/to/Doomer`" in the `.env`-file with your actual root directory of Doomer.

## Inference
To run inference in the Real-world setting, [pick a checkpoint from Google Drive](https://drive.google.com/drive/folders/15oF-hawKMtupRRSoz3KMCF0nx9vfhaqW?usp=sharing) and pass it to the `--checkpoint` argument of the `reconstruction/test.py`.
Note that each method has 10 checkpoints, since we needed that for statistical significance.

For example, if you put `DCAM_GMSTpp_Doomer_mtw_2.pth` into the root of the repo,
```bash
cd reconstruction
python test.py --checkpoint ../DCAM_GMSTpp_Doomer_mtw_2.pth
```
evaluates our DCAM + GMST++ model on the test set and writes a message to stdout ending with
```
Macs: 25.61G, Params: 656.50K
       MAE       NSE       PSNR       SAM
  0.070023  0.080963  34.860725  3.673643
```

## Training
To train a model, pick a config from `reconstruction/configs` and pass it to the `--config` argument of the `reconstruction/train.py`.

For example,
```bash
cd reconstruction
python train.py --config configs/real/3cam_dcam_gmstpp.yaml
```
will train our DCAM + GMST++ model using multi-camera inputs.

## Doomer
Structure of the dataset:
```
Doomer
├── aligned
│   ├── 000.h5
│   ├── 001.h5
│   ...
└── warped
    └── PWCNet
        ├── 000.h5
        ├── 001.h5
        ...
```

Train/validation/test split is specified in `reconstruction/splits/Doomer.json`

Structure of an .h5-file in the `aligned` directory:
- `rgbs` _(3, 3, 192, 256):_ Main, Telephoto and Wide input views;
- `hsi` _(34, 192, 256):_ Hyperspectral image, aligned using SIFT + RANSAC.

Structure of an .h5-file in the `warped/PWCNet` directory:
- `hsi` _(34, 192, 256):_ Hyperspectral image under optical-flow warping towards Main view.
- `mask` _(1, 192, 256):_ Mask of valid pixels.

As ground truth we use HSIs from the `warped/PWCNet` directory.
To generate ground truth using optical flow of RAFT, run `python reconstruction/prepare_benchmark_gt.py RAFT` and do not forget to update the `warped_gt_dir` parameter at a training YAML-config.

## Coming Soon
- Web-page of the project.
- Full version of the Doomer dataset, including raw images.
- Code for choice of filters, training on synthetic data and transfer to other devices.
