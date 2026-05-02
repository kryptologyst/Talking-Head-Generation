"""Loss functions for talking head generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class SyncLoss(nn.Module):
    """Synchronization loss between audio and video features."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute synchronization loss.
        
        Args:
            audio_features: Audio features of shape (B, T_a, D)
            video_features: Video features of shape (B, T_v, D)
            
        Returns:
            Synchronization loss
        """
        # Normalize features
        audio_features = F.normalize(audio_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(
            audio_features.transpose(1, 2),  # (B, D, T_a)
            video_features  # (B, T_v, D)
        )  # (B, T_a, T_v)
        
        # Apply temperature
        similarity = similarity / self.temperature
        
        # Create positive pairs (diagonal elements)
        B, T_a, T_v = similarity.shape
        min_length = min(T_a, T_v)
        
        # Extract diagonal elements
        diagonal_indices = torch.arange(min_length, device=similarity.device)
        positive_similarities = similarity[:, diagonal_indices, diagonal_indices]
        
        # Compute contrastive loss
        loss = -torch.mean(positive_similarities)
        
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features."""
    
    def __init__(self, feature_layers: Optional[list] = None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]  # VGG16 feature layers
        
        self.feature_layers = feature_layers
        
        # Load pre-trained VGG16
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            generated: Generated video frames of shape (B, T, C, H, W)
            target: Target video frames of shape (B, T, C, H, W)
            
        Returns:
            Perceptual loss
        """
        B, T, C, H, W = generated.shape
        
        # Reshape to process all frames
        generated_flat = generated.view(B * T, C, H, W)
        target_flat = target.view(B * T, C, H, W)
        
        # Extract features
        gen_features = []
        target_features = []
        
        x_gen = generated_flat
        x_target = target_flat
        
        for i, layer in enumerate(self.features):
            x_gen = layer(x_gen)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                gen_features.append(x_gen)
                target_features.append(x_target)
        
        # Compute L2 loss for each feature layer
        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += F.mse_loss(gen_feat, target_feat)
        
        return loss / len(self.feature_layers)


class AdversarialLoss(nn.Module):
    """Adversarial loss for discriminator."""
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        is_real: bool = True
    ) -> torch.Tensor:
        """Compute adversarial loss.
        
        Args:
            generated: Generated video frames
            target: Target video frames
            is_real: Whether the input is real or fake
            
        Returns:
            Adversarial loss
        """
        if is_real:
            # Real samples
            logits = self.discriminator(target)
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits)
            )
        else:
            # Generated samples
            logits = self.discriminator(generated)
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits)
            )
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for smooth video generation."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss.
        
        Args:
            video: Video tensor of shape (B, T, C, H, W)
            
        Returns:
            Temporal consistency loss
        """
        # Compute difference between consecutive frames
        frame_diff = video[:, 1:] - video[:, :-1]
        
        # L2 loss on frame differences
        loss = torch.mean(frame_diff ** 2)
        
        return self.weight * loss


class CombinedLoss(nn.Module):
    """Combined loss function for talking head generation."""
    
    def __init__(
        self,
        sync_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.1,
        temporal_weight: float = 0.1,
        discriminator: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.sync_weight = sync_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.temporal_weight = temporal_weight
        
        # Initialize loss functions
        self.sync_loss = SyncLoss()
        self.perceptual_loss = PerceptualLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
        if discriminator is not None:
            self.adversarial_loss = AdversarialLoss(discriminator)
        else:
            self.adversarial_loss = None
        
    def forward(
        self,
        generated_video: torch.Tensor,
        target_video: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            generated_video: Generated video frames
            target_video: Target video frames
            audio_features: Audio features
            video_features: Video features
            is_training: Whether in training mode
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Reconstruction loss (L1)
        reconstruction_loss = F.l1_loss(generated_video, target_video)
        losses["reconstruction"] = reconstruction_loss
        
        # Synchronization loss
        sync_loss = self.sync_loss(audio_features, video_features)
        losses["sync"] = sync_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(generated_video, target_video)
        losses["perceptual"] = perceptual_loss
        
        # Temporal consistency loss
        temporal_loss = self.temporal_loss(generated_video)
        losses["temporal"] = temporal_loss
        
        # Adversarial loss (if discriminator is available)
        if self.adversarial_loss is not None and is_training:
            adversarial_loss = self.adversarial_loss(
                generated_video, target_video, is_real=False
            )
            losses["adversarial"] = adversarial_loss
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.sync_weight * sync_loss +
            self.perceptual_weight * perceptual_loss +
            self.temporal_weight * temporal_loss
        )
        
        if self.adversarial_loss is not None and is_training:
            total_loss += self.adversarial_weight * losses["adversarial"]
        
        losses["total"] = total_loss
        
        return losses
