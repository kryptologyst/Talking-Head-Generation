"""Evaluation metrics for talking head generation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score


class SyncQualityMetrics:
    """Metrics for evaluating audio-video synchronization quality."""
    
    def __init__(self, sample_rate: int = 16000, video_fps: int = 25):
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        
    def compute_sync_score(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> float:
        """Compute synchronization score between audio and video features.
        
        Args:
            audio_features: Audio features of shape (B, T_a, D)
            video_features: Video features of shape (B, T_v, D)
            
        Returns:
            Synchronization score (higher is better)
        """
        # Normalize features
        audio_features = F.normalize(audio_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)
        
        # Compute cross-correlation
        B, T_a, D = audio_features.shape
        _, T_v, _ = video_features.shape
        
        # Align temporal dimensions
        min_length = min(T_a, T_v)
        audio_aligned = audio_features[:, :min_length]
        video_aligned = video_features[:, :min_length]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            audio_aligned, video_aligned, dim=-1
        )  # (B, T)
        
        # Average over time and batch
        sync_score = torch.mean(similarity).item()
        
        return sync_score
    
    def compute_offset_error(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> float:
        """Compute temporal offset error between audio and video.
        
        Args:
            audio_features: Audio features
            video_features: Video features
            
        Returns:
            Offset error in seconds
        """
        # Compute cross-correlation to find optimal alignment
        audio_np = audio_features.detach().cpu().numpy()
        video_np = video_features.detach().cpu().numpy()
        
        # Average over batch and feature dimensions
        audio_avg = np.mean(audio_np, axis=(0, 2))  # (T_a,)
        video_avg = np.mean(video_np, axis=(0, 2))  # (T_v,)
        
        # Compute cross-correlation
        correlation = np.correlate(audio_avg, video_avg, mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(correlation)
        optimal_offset = peak_idx - len(video_avg) + 1
        
        # Convert to seconds
        offset_seconds = optimal_offset / self.video_fps
        
        return abs(offset_seconds)


class VisualQualityMetrics:
    """Metrics for evaluating visual quality of generated videos."""
    
    def __init__(self):
        pass
    
    def compute_psnr(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio.
        
        Args:
            generated: Generated video frames
            target: Target video frames
            
        Returns:
            PSNR value
        """
        mse = F.mse_loss(generated, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """Compute Structural Similarity Index.
        
        Args:
            generated: Generated video frames
            target: Target video frames
            window_size: Window size for SSIM computation
            
        Returns:
            SSIM value
        """
        # Simple SSIM implementation
        B, T, C, H, W = generated.shape
        
        # Reshape to process all frames
        gen_flat = generated.view(B * T, C, H, W)
        target_flat = target.view(B * T, C, H, W)
        
        # Compute means
        mu_gen = torch.mean(gen_flat, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target_flat, dim=[2, 3], keepdim=True)
        
        # Compute variances and covariance
        sigma_gen = torch.var(gen_flat, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target_flat, dim=[2, 3], keepdim=True)
        
        # Compute covariance
        gen_centered = gen_flat - mu_gen
        target_centered = target_flat - mu_target
        sigma_gen_target = torch.mean(gen_centered * target_centered, dim=[2, 3], keepdim=True)
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Compute SSIM
        ssim = (
            (2 * mu_gen * mu_target + c1) * (2 * sigma_gen_target + c2)
        ) / (
            (mu_gen ** 2 + mu_target ** 2 + c1) * (sigma_gen + sigma_target + c2)
        )
        
        return torch.mean(ssim).item()
    
    def compute_lpips(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute Learned Perceptual Image Patch Similarity.
        
        Args:
            generated: Generated video frames
            target: Target video frames
            
        Returns:
            LPIPS value
        """
        # Simplified LPIPS using VGG features
        import torchvision.models as models
        
        vgg = models.vgg16(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        
        B, T, C, H, W = generated.shape
        
        # Reshape to process all frames
        gen_flat = generated.view(B * T, C, H, W)
        target_flat = target.view(B * T, C, H, W)
        
        # Extract features
        gen_features = vgg(gen_flat)
        target_features = vgg(target_flat)
        
        # Compute L2 distance
        lpips = F.mse_loss(gen_features, target_features)
        
        return lpips.item()


class TalkingHeadEvaluator:
    """Main evaluator for talking head generation."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        video_fps: int = 25
    ):
        self.sync_metrics = SyncQualityMetrics(sample_rate, video_fps)
        self.visual_metrics = VisualQualityMetrics()
    
    def evaluate(
        self,
        generated_video: torch.Tensor,
        target_video: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate talking head generation quality.
        
        Args:
            generated_video: Generated video frames
            target_video: Target video frames
            audio_features: Audio features
            video_features: Video features
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Synchronization metrics
        metrics["sync_score"] = self.sync_metrics.compute_sync_score(
            audio_features, video_features
        )
        metrics["offset_error"] = self.sync_metrics.compute_offset_error(
            audio_features, video_features
        )
        
        # Visual quality metrics
        metrics["psnr"] = self.visual_metrics.compute_psnr(
            generated_video, target_video
        )
        metrics["ssim"] = self.visual_metrics.compute_ssim(
            generated_video, target_video
        )
        metrics["lpips"] = self.visual_metrics.compute_lpips(
            generated_video, target_video
        )
        
        return metrics
    
    def evaluate_batch(
        self,
        generated_videos: torch.Tensor,
        target_videos: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate a batch of samples.
        
        Args:
            generated_videos: Generated video frames
            target_videos: Target video frames
            audio_features: Audio features
            video_features: Video features
            
        Returns:
            Dictionary of average evaluation metrics
        """
        batch_metrics = []
        
        for i in range(generated_videos.shape[0]):
            sample_metrics = self.evaluate(
                generated_videos[i:i+1],
                target_videos[i:i+1],
                audio_features[i:i+1],
                video_features[i:i+1]
            )
            batch_metrics.append(sample_metrics)
        
        # Average metrics across batch
        avg_metrics = {}
        for key in batch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
        
        return avg_metrics
