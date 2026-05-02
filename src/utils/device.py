"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Preferred device ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        torch.mps.manual_seed(seed)


def setup_device(device: Optional[str] = None, mixed_precision: bool = True) -> torch.device:
    """Setup device with optimal settings.
    
    Args:
        device: Preferred device
        mixed_precision: Whether to enable mixed precision
        
    Returns:
        torch.device: The configured device
    """
    device = get_device(device)
    
    if device.type == "cuda":
        # Enable optimized cuDNN
        cudnn.benchmark = True
        cudnn.enabled = True
        
        # Set memory allocation strategy
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    elif device.type == "mps":
        # MPS-specific optimizations
        torch.mps.empty_cache()
    
    return device


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
    
    return info


def clear_cache(device: Optional[torch.device] = None) -> None:
    """Clear device memory cache.
    
    Args:
        device: Device to clear cache for
    """
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
