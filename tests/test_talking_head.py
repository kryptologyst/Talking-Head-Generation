"""Tests for talking head generation."""

import pytest
import torch
import numpy as np
from src.models.wav2lip import Wav2LipModel
from src.utils.device import get_device, set_seed
from src.utils.audio import audio_to_tensor
from src.utils.video import frames_to_tensor


class TestWav2LipModel:
    """Test cases for Wav2Lip model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        # Model configuration
        self.config = {
            "audio_encoder": {
                "model_name": "facebook/wav2vec2-base-960h",
                "freeze_encoder": True,
                "output_dim": 768
            },
            "video_encoder": {
                "input_channels": 3,
                "output_dim": 512,
                "pretrained": False
            },
            "fusion": {
                "hidden_dim": 512,
                "num_heads": 8,
                "num_layers": 2
            },
            "decoder": {
                "input_channels": 3,
                "output_channels": 3,
                "base_channels": 32,
                "num_layers": 2
            }
        }
        
        self.model = Wav2LipModel(**self.config).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert isinstance(self.model, Wav2LipModel)
        assert hasattr(self.model, 'audio_encoder')
        assert hasattr(self.model, 'video_encoder')
        assert hasattr(self.model, 'fusion')
        assert hasattr(self.model, 'decoder')
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        video_frames = 25  # 1 second at 25fps
        video_size = (64, 64)  # Smaller for testing
        
        # Create dummy inputs
        audio = torch.randn(batch_size, audio_length).to(self.device)
        video = torch.randn(batch_size, video_frames, 3, *video_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(audio, video)
        
        # Check output shape
        expected_shape = (batch_size, video_frames, 3, *video_size)
        assert output.shape == expected_shape
    
    def test_generate_method(self):
        """Test generation method."""
        batch_size = 1
        audio_length = 8000  # 0.5 second
        video_frames = 12  # 0.5 second
        video_size = (64, 64)
        
        # Create dummy inputs
        audio = torch.randn(batch_size, audio_length).to(self.device)
        reference_video = torch.randn(batch_size, video_frames, 3, *video_size).to(self.device)
        
        # Generate video
        with torch.no_grad():
            generated = self.model.generate(audio, reference_video)
        
        # Check output shape
        expected_shape = (batch_size, video_frames, 3, *video_size)
        assert generated.shape == expected_shape


class TestAudioUtils:
    """Test cases for audio utilities."""
    
    def test_audio_to_tensor(self):
        """Test audio to tensor conversion."""
        # Create dummy audio
        audio = np.random.randn(16000).astype(np.float32)
        
        # Convert to tensor
        tensor = audio_to_tensor(audio, add_batch_dim=True)
        
        # Check shape and type
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 16000)
        assert tensor.dtype == torch.float32
    
    def test_audio_to_tensor_no_batch(self):
        """Test audio to tensor conversion without batch dimension."""
        audio = np.random.randn(16000).astype(np.float32)
        tensor = audio_to_tensor(audio, add_batch_dim=False)
        
        assert tensor.shape == (16000,)


class TestVideoUtils:
    """Test cases for video utilities."""
    
    def test_frames_to_tensor(self):
        """Test frames to tensor conversion."""
        # Create dummy frames
        frames = np.random.randn(25, 64, 64, 3).astype(np.float32)
        
        # Convert to tensor
        tensor = frames_to_tensor(frames, add_batch_dim=True)
        
        # Check shape and type
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 25, 3, 64, 64)
        assert tensor.dtype == torch.float32


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device_auto = get_device("auto")
        assert device_auto.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        rand1 = torch.randn(10)
        set_seed(42)
        rand2 = torch.randn(10)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
