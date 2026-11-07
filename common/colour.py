import numpy as np
import torch
import colour
from scipy.interpolate import interp1d
from scipy.linalg import null_space
from sklearn.decomposition import TruncatedSVD
from typing import Literal


def srgb_cmfs(wavelengths: np.ndarray):
    cmfs = colour.MSDS_CMFS['Wright & Guild 1931 2 Degree RGB CMFs']
    return interp1d(cmfs.wavelengths, cmfs.values, axis=0, fill_value=0, bounds_error=False)(wavelengths)


def gamma_correction(rgb: np.ndarray):
    return np.where(rgb <= 0.0031308,
                    12.92 * rgb,
                    1.055 * rgb**(1/2.4) - 0.055)


def render(
        img: np.ndarray,
        color_matrix: np.ndarray | None = None,
        normalization: Literal['q99', 'max', 'reinhardt', 'none'] = 'q99',
        clip: bool = True,
        gamma_corr: bool = True,
        uint8: bool = True,
):
    if color_matrix is not None:
        # print(img.shape, color_matrix.shape)  
        if len(color_matrix.shape) == 3:
            h, w  = img.shape[:2]
            img = img.reshape(h, w, 3, 3)

            img = np.einsum('hwic, icj->hwij', img, color_matrix)
            img = img.reshape(h, w, 9)
        
        else:
            img = img @ color_matrix
        
    eps = 1e-6
    match normalization:
        case 'q99':
            img = img / np.maximum(np.quantile(img, 0.99), eps)
        case 'max':
            img = img / np.maximum(img.max(), eps)
        case 'reinhardt':
            img = img / np.maximum(img.mean(), eps)
            img = img / (1 + img)
    
    if clip:
        img = np.clip(img, 0, 1)
    if gamma_corr:
        img = gamma_correction(img)
    if uint8:
        img = (img * 255).astype(np.uint8)
    return img


def prerender_invisible(hsi: np.ndarray, cmfs: np.ndarray):
    spectra = hsi.reshape(-1, hsi.shape[2])
    cov = spectra.T @ spectra / len(spectra)
    cov_black = null_space(cmfs.T @ cov)
    black_hsi = hsi @ cov_black
    black_spectra = black_hsi.reshape(-1, black_hsi.shape[2])
    tsvd = TruncatedSVD(n_components=3).fit(black_spectra)
    black_cmfs = tsvd.components_.T
    img = black_hsi @ black_cmfs
    img = img[..., [1, 0, 2]]
    return img


def gamma_correction_torch(rgb: torch.Tensor):
    return torch.where(rgb <= 0.0031308,
                       12.92 * rgb,
                       1.055 * rgb**(1/2.4) - 0.055)


def render_torch(
        img: torch.Tensor,
        color_matrix: torch.Tensor | None = None,
        normalization: Literal['q99', 'max', 'reinhardt', 'none'] = 'q99',
        clip: bool = True,
        gamma_corr: bool = True,
        uint8: bool = True,
):
    '''
    img: CxHxW
    '''
    if color_matrix is not None:
        # print(img.shape, color_matrix.shape)  
        if len(color_matrix.shape) == 3:
            h, w  = img.shape[-2:]
            img = img.reshape(3, 3, h, w)

            img = torch.einsum('ichw, cij->jchw', img, color_matrix)
            img = img.reshape(9, h, w)
        
        else:
            img = torch.einsum('chw, ci->ihw', img, color_matrix)
        

    eps = torch.tensor([1e-6]).to(img.get_device())
    match normalization:
        case 'q99':
            img = img / torch.maximum(torch.quantile(img, 0.99), eps)
        case 'max':
            img = img / torch.maximum(img.max(), eps)
        case 'reinhardt':
            img = img / torch.maximum(img.mean(), eps)
            img = img / (1 + img)
    if clip:
        img = torch.clamp(img, 0, 1)
    if gamma_corr:
        img = gamma_correction_torch(img)
    if uint8:
        img = (img * 255).to(torch.uint8)
    return img