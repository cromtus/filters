import torch
import numpy as np
from dataclasses import dataclass
from typing import overload


@dataclass
class Noise:
    a: np.ndarray
    b: np.ndarray
    a_torch: torch.Tensor
    b_torch: torch.Tensor

    @classmethod
    def from_numpy_params(cls, a: np.ndarray, b: np.ndarray):
        return cls(a, b, torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))

    def to(self, dest):
        a_torch = self.a_torch.to(dest)
        b_torch = self.b_torch.to(dest)
        return Noise(self.a.copy(), self.b.copy(), a_torch, b_torch)

    @classmethod
    def from_fit(cls, signal: np.ndarray, std: np.ndarray):
        """
        signal (n_samples, n_channels)
        std (n_samples, n_channels)
        """
        n_channels = signal.shape[1]
        a = np.zeros(n_channels)
        b = np.zeros(n_channels)
        for i in range(n_channels):
            x, y = signal[:, i], std[:, i]**2
            a[i], b[i] = np.polyfit(x, y, 1)
            b[i] = np.clip(b[i], 0, None)
        return cls.from_numpy_params(a, b)
    
    @classmethod
    def from_dict(cls, params: dict):
        return cls.from_numpy_params(np.array(params['a']),
                                     np.array(params['b']))
    
    def to_dict(self):
        return {'a': self.a.tolist(), 'b': self.b.tolist()}
    
    def get_std(self, signal: np.ndarray | torch.Tensor):
        if isinstance(signal, np.ndarray):
            return np.sqrt(self.a * signal + self.b)
        else:
            return torch.sqrt(self.a_torch * signal + self.b_torch)
    
    def transform(self, signal: np.ndarray):
        normal_noise = np.random.normal(0, 1, size=signal.shape)
        signal_noise = self.get_std(signal) * normal_noise
        return np.clip(signal + signal_noise, 0, None)
    
    def transform_torch(self, signal: torch.Tensor):
        normal_noise = torch.normal(0, 1, size=signal.shape, device=signal.device)
        if signal.ndim > 2:
            a = self.a_torch[:, None, None]
            b = self.b_torch[:, None, None]
        else:
            a, b = self.a_torch, self.b_torch
        signal_noise = torch.sqrt(a * signal + b) * normal_noise
        return torch.clip(signal + signal_noise, 0, None)

    @overload
    def __call__(self, signal: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, signal: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, signal: np.ndarray | torch.Tensor):
        if isinstance(signal, np.ndarray):
            return self.transform(signal)
        else:
            return self.transform_torch(signal)
    
    @classmethod
    def concat(cls, noises: list['Noise']):
        return cls.from_numpy_params(
            np.concatenate([noise.a for noise in noises]),
            np.concatenate([noise.b for noise in noises])
        )
