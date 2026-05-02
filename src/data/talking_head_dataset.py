"""Dataset and data loading utilities for talking head generation."""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from ..utils.audio import load_audio, pad_or_truncate_audio, audio_to_tensor
from ..utils.video import load_video, resize_video, normalize_frames, frames_to_tensor


class TalkingHeadDataset(Dataset):
    """Dataset for talking head generation."""
    
    def __init__(
        self,
        audio_dir: str,
        video_dir: str,
        annotations_file: str,
        sample_rate: int = 16000,
        video_fps: int = 25,
        video_size: Tuple[int, int] = (256, 256),
        max_audio_length: float = 10.0,
        max_video_length: float = 10.0,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.video_size = video_size
        self.max_audio_length = max_audio_length
        self.max_video_length = max_video_length
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Split data
        self.data_split = self._split_data(split, train_split, val_split, test_split, random_seed)
        
    def _split_data(
        self,
        split: str,
        train_split: float,
        val_split: float,
        test_split: float,
        random_seed: int
    ) -> List[Dict]:
        """Split data into train/val/test sets."""
        np.random.seed(random_seed)
        
        total_samples = len(self.annotations)
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        
        # Shuffle indices
        indices = np.random.permutation(total_samples)
        
        if split == "train":
            split_indices = indices[:train_size]
        elif split == "val":
            split_indices = indices[train_size:train_size + val_size]
        elif split == "test":
            split_indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return [self.annotations[i] for i in split_indices]
    
    def __len__(self) -> int:
        return len(self.data_split)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        sample = self.data_split[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, sample["audio_file"])
        audio, _ = load_audio(audio_path, self.sample_rate)
        
        # Pad or truncate audio
        max_audio_samples = int(self.max_audio_length * self.sample_rate)
        audio = pad_or_truncate_audio(audio, max_audio_samples)
        audio_tensor = audio_to_tensor(audio, add_batch_dim=False)
        
        # Load video
        video_path = os.path.join(self.video_dir, sample["video_file"])
        video_frames, fps = load_video(video_path, self.video_fps)
        
        # Resize video
        video_frames = resize_video(video_frames, self.video_size)
        
        # Pad or truncate video
        max_video_frames = int(self.max_video_length * self.video_fps)
        if len(video_frames) > max_video_frames:
            video_frames = video_frames[:max_video_frames]
        elif len(video_frames) < max_video_frames:
            # Pad with last frame
            last_frame = video_frames[-1:]
            padding_frames = np.repeat(last_frame, max_video_frames - len(video_frames), axis=0)
            video_frames = np.concatenate([video_frames, padding_frames], axis=0)
        
        # Normalize frames
        video_frames = normalize_frames(video_frames)
        video_tensor = frames_to_tensor(video_frames, add_batch_dim=False)
        
        return {
            "audio": audio_tensor,
            "video": video_tensor,
            "audio_file": sample["audio_file"],
            "video_file": sample["video_file"],
            "duration": sample.get("duration", self.max_audio_length)
        }


class SyntheticTalkingHeadDataset(Dataset):
    """Synthetic dataset for demonstration purposes."""
    
    def __init__(
        self,
        num_samples: int = 100,
        sample_rate: int = 16000,
        video_fps: int = 25,
        video_size: Tuple[int, int] = (256, 256),
        max_audio_length: float = 5.0,
        max_video_length: float = 5.0
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.video_size = video_size
        self.max_audio_length = max_audio_length
        self.max_video_length = max_video_length
        
        # Generate synthetic data
        self.samples = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic audio-video pairs."""
        samples = []
        
        for i in range(self.num_samples):
            # Generate synthetic audio (sine wave with varying frequency)
            duration = np.random.uniform(2.0, self.max_audio_length)
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            frequency = np.random.uniform(200, 800)
            audio = np.sin(2 * np.pi * frequency * t) * 0.5
            
            # Add some noise
            noise = np.random.normal(0, 0.1, audio.shape)
            audio = audio + noise
            
            # Generate synthetic video (colored frames with moving patterns)
            num_frames = int(duration * self.video_fps)
            video_frames = []
            
            for frame_idx in range(num_frames):
                # Create a colored frame with moving patterns
                frame = np.zeros((*self.video_size, 3), dtype=np.uint8)
                
                # Add moving circles
                center_x = int(self.video_size[0] * 0.5 + 50 * np.sin(frame_idx * 0.1))
                center_y = int(self.video_size[1] * 0.5 + 30 * np.cos(frame_idx * 0.1))
                
                cv2.circle(frame, (center_x, center_y), 30, (255, 100, 100), -1)
                
                # Add some random patterns
                for _ in range(5):
                    x = np.random.randint(0, self.video_size[0])
                    y = np.random.randint(0, self.video_size[1])
                    radius = np.random.randint(5, 15)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.circle(frame, (x, y), radius, color, -1)
                
                video_frames.append(frame)
            
            video_frames = np.array(video_frames)
            
            samples.append({
                "audio": audio,
                "video": video_frames,
                "duration": duration
            })
        
        return samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a synthetic sample."""
        sample = self.samples[idx]
        
        # Convert audio to tensor
        audio_tensor = audio_to_tensor(sample["audio"], add_batch_dim=False)
        
        # Convert video to tensor
        video_frames = normalize_frames(sample["video"])
        video_tensor = frames_to_tensor(video_frames, add_batch_dim=False)
        
        return {
            "audio": audio_tensor,
            "video": video_tensor,
            "duration": sample["duration"]
        }


def create_data_loaders(
    config: Dict,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    # Check if real data exists, otherwise use synthetic
    if os.path.exists(config["annotations_file"]):
        train_dataset = TalkingHeadDataset(
            split="train",
            **config
        )
        val_dataset = TalkingHeadDataset(
            split="val",
            **config
        )
        test_dataset = TalkingHeadDataset(
            split="test",
            **config
        )
    else:
        # Use synthetic datasets
        train_dataset = SyntheticTalkingHeadDataset(num_samples=80)
        val_dataset = SyntheticTalkingHeadDataset(num_samples=10)
        test_dataset = SyntheticTalkingHeadDataset(num_samples=10)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader
