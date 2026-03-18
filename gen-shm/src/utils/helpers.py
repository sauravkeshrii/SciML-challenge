"""
Utility functions for Gen-SHM project.
"""

import torch
import numpy as np
from typing import Tuple, Union, List
import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device (CUDA if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_meshgrid(
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    x_points: int,
    t_points: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create meshgrid for spatial and temporal coordinates.
    
    Args:
        x_range: (min_x, max_x)
        t_range: (min_t, max_t)
        x_points: Number of spatial points
        t_points: Number of temporal points
        
    Returns:
        x_grid, t_grid tensors
    """
    x = torch.linspace(x_range[0], x_range[1], x_points)
    t = torch.linspace(t_range[0], t_range[1], t_points)
    x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
    return x_grid, t_grid


def sample_collocation_points(
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    num_points: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Sample collocation points uniformly in space-time domain.
    
    Args:
        x_range: (min_x, max_x)
        t_range: (min_t, max_t)
        num_points: Number of points to sample
        device: Target device
        
    Returns:
        Tensor of shape (num_points, 2) with [x, t] coordinates
    """
    if device is None:
        device = get_device()
        
    x_samples = torch.rand(num_points, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t_samples = torch.rand(num_points, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    return torch.stack([x_samples, t_samples], dim=1)


def compute_derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Compute derivative of y with respect to x using automatic differentiation.
    
    Args:
        y: Function values
        x: Variable to differentiate with respect to
        order: Order of derivative (1 or 2)
        
    Returns:
        Derivative tensor
    """
    if order == 1:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    elif order == 2:
        first_deriv = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
        return torch.autograd.grad(
            first_deriv, x, grad_outputs=torch.ones_like(first_deriv),
            create_graph=True, retain_graph=True
        )[0]
    else:
        raise ValueError("Only first and second order derivatives supported")


def normalize_tensor(tensor: torch.Tensor, min_val: float = None, max_val: float = None) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value (computed if None)
        max_val: Maximum value (computed if None)
        
    Returns:
        Normalized tensor
    """
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def denormalize_tensor(normalized_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Denormalize tensor from [0, 1] range.
    
    Args:
        normalized_tensor: Normalized input tensor
        min_val: Original minimum value
        max_val: Original maximum value
        
    Returns:
        Denormalized tensor
    """
    return normalized_tensor * (max_val - min_val) + min_val


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute moving average of data.
    
    Args:
        data: Input array
        window_size: Size of averaging window
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)