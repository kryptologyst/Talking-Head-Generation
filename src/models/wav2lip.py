"""Wav2Lip-style model for talking head generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Dict, Optional, Tuple
import math


class AudioEncoder(nn.Module):
    """Audio encoder using Wav2Vec2."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_encoder: bool = False,
        output_dim: int = 768
    ):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Projection layer to match desired output dimension
        self.projection = nn.Linear(self.wav2vec2.config.hidden_size, output_dim)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through audio encoder.
        
        Args:
            audio: Input audio tensor of shape (B, T)
            
        Returns:
            Audio features of shape (B, T', D)
        """
        outputs = self.wav2vec2(audio)
        features = outputs.last_hidden_state
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


class VideoEncoder(nn.Module):
    """Video encoder using ResNet backbone."""
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Use ResNet18 as backbone
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify first layer if needed
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection layer
        self.projection = nn.Linear(512, output_dim)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass through video encoder.
        
        Args:
            video: Input video tensor of shape (B, T, C, H, W)
            
        Returns:
            Video features of shape (B, T, D)
        """
        B, T, C, H, W = video.shape
        
        # Reshape to process all frames at once
        video_flat = video.view(B * T, C, H, W)
        
        # Extract features
        features = self.backbone(video_flat)  # (B*T, 512)
        features = features.view(B, T, -1)
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


class CrossModalAttention(nn.Module):
    """Cross-modal attention between audio and video features."""
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through cross-modal attention.
        
        Args:
            audio_features: Audio features of shape (B, T_a, D_a)
            video_features: Video features of shape (B, T_v, D_v)
            
        Returns:
            Fused features of shape (B, T_v, D)
        """
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)
        
        # Cross-modal attention
        x = video_proj
        
        for attention, norm, ff in zip(
            self.attention_layers, self.norm_layers, self.feedforward
        ):
            # Self-attention on video features
            attn_out, _ = attention(x, x, x)
            x = norm(x + attn_out)
            
            # Cross-attention with audio
            cross_attn_out, _ = attention(x, audio_proj, audio_proj)
            x = norm(x + cross_attn_out)
            
            # Feedforward
            ff_out = ff(x)
            x = norm(x + ff_out)
        
        return x


class VideoDecoder(nn.Module):
    """Video decoder using UNet architecture."""
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        in_channels = input_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            
            # Encoder
            encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.encoder_layers.append(encoder)
            
            in_channels = out_channels
        
        # Decoder layers
        for i in range(num_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** i) if i > 0 else output_channels
            
            decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 2, stride=2
                ),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.decoder_layers.append(decoder)
            
            in_channels = out_channels
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Decoded video of shape (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Process each frame independently
        outputs = []
        
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            
            # Encoder
            encoder_outputs = []
            for encoder in self.encoder_layers:
                frame = encoder(frame)
                encoder_outputs.append(frame)
                frame = F.max_pool2d(frame, 2)
            
            # Decoder
            for i, decoder in enumerate(self.decoder_layers):
                if i > 0:
                    # Skip connection
                    skip_idx = len(encoder_outputs) - 1 - i
                    frame = torch.cat([frame, encoder_outputs[skip_idx]], dim=1)
                
                frame = decoder(frame)
            
            frame = self.final_activation(frame)
            outputs.append(frame)
        
        return torch.stack(outputs, dim=1)  # (B, T, C, H, W)


class Wav2LipModel(nn.Module):
    """Main Wav2Lip-style model for talking head generation."""
    
    def __init__(
        self,
        audio_encoder_config: Dict,
        video_encoder_config: Dict,
        fusion_config: Dict,
        decoder_config: Dict
    ):
        super().__init__()
        
        self.audio_encoder = AudioEncoder(**audio_encoder_config)
        self.video_encoder = VideoEncoder(**video_encoder_config)
        
        self.fusion = CrossModalAttention(
            audio_dim=audio_encoder_config["output_dim"],
            video_dim=video_encoder_config["output_dim"],
            **fusion_config
        )
        
        self.decoder = VideoDecoder(**decoder_config)
        
        # Feature projection for decoder input
        self.feature_projection = nn.Linear(
            fusion_config["hidden_dim"],
            decoder_config["input_channels"] * 64 * 64  # Assuming 64x64 feature maps
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio: Input audio of shape (B, T_a)
            video: Input video of shape (B, T_v, C, H, W)
            
        Returns:
            Generated video of shape (B, T_v, C, H, W)
        """
        # Encode audio and video
        audio_features = self.audio_encoder(audio)  # (B, T_a', D_a)
        video_features = self.video_encoder(video)  # (B, T_v, D_v)
        
        # Cross-modal fusion
        fused_features = self.fusion(audio_features, video_features)  # (B, T_v, D)
        
        # Project features for decoder
        B, T_v, D = fused_features.shape
        projected_features = self.feature_projection(fused_features)  # (B, T_v, C*H*W)
        
        # Reshape for decoder
        projected_features = projected_features.view(B, T_v, 3, 64, 64)
        
        # Decode to video
        generated_video = self.decoder(projected_features)
        
        return generated_video
    
    def generate(
        self,
        audio: torch.Tensor,
        reference_video: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate talking head video from audio and reference video.
        
        Args:
            audio: Input audio
            reference_video: Reference video frame(s)
            max_length: Maximum length of generated video
            
        Returns:
            Generated talking head video
        """
        self.eval()
        
        with torch.no_grad():
            return self.forward(audio, reference_video)
