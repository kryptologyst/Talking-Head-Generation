"""Audio processing utilities for talking head generation."""

import librosa
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional, Union
import soundfile as sf


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file and return waveform and sample rate.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Whether to convert to mono
        normalize: Whether to normalize audio
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=mono,
        dtype=np.float32
    )
    
    if normalize:
        waveform = librosa.util.normalize(waveform)
    
    return waveform, sr


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None
) -> np.ndarray:
    """Extract mel spectrogram from waveform.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate of audio
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        
    Returns:
        Mel spectrogram
    """
    if win_length is None:
        win_length = n_fft
    
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=0,
        fmax=sample_rate // 2
    )
    
    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def extract_mfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 26,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """Extract MFCC features from waveform.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate of audio
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        MFCC features
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfcc


def extract_spectral_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512
) -> dict:
    """Extract various spectral features from waveform.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate of audio
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    # Spectral centroid
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=waveform, sr=sample_rate, hop_length=hop_length
    )[0]
    
    # Spectral rolloff
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=waveform, sr=sample_rate, hop_length=hop_length
    )[0]
    
    # Zero crossing rate
    features['zcr'] = librosa.feature.zero_crossing_rate(
        waveform, frame_length=frame_length, hop_length=hop_length
    )[0]
    
    # RMS energy
    features['rms'] = librosa.feature.rms(
        y=waveform, frame_length=frame_length, hop_length=hop_length
    )[0]
    
    return features


def pad_or_truncate_audio(
    waveform: np.ndarray,
    target_length: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """Pad or truncate audio to target length.
    
    Args:
        waveform: Input waveform
        target_length: Target length in samples
        pad_value: Value to use for padding
        
    Returns:
        Padded or truncated waveform
    """
    current_length = len(waveform)
    
    if current_length > target_length:
        # Truncate
        return waveform[:target_length]
    elif current_length < target_length:
        # Pad
        padding = np.full(target_length - current_length, pad_value, dtype=waveform.dtype)
        return np.concatenate([waveform, padding])
    else:
        return waveform


def audio_to_tensor(
    waveform: np.ndarray,
    add_batch_dim: bool = True
) -> torch.Tensor:
    """Convert numpy audio to PyTorch tensor.
    
    Args:
        waveform: Input waveform as numpy array
        add_batch_dim: Whether to add batch dimension
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(waveform).float()
    
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy audio.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Numpy array
    """
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    
    return tensor.detach().cpu().numpy()


def save_audio(
    waveform: Union[np.ndarray, torch.Tensor],
    file_path: str,
    sample_rate: int = 16000
) -> None:
    """Save audio waveform to file.
    
    Args:
        waveform: Audio waveform
        file_path: Output file path
        sample_rate: Sample rate
    """
    if isinstance(waveform, torch.Tensor):
        waveform = tensor_to_audio(waveform)
    
    sf.write(file_path, waveform, sample_rate)
