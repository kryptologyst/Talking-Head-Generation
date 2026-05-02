#!/usr/bin/env python3
"""Example script demonstrating talking head generation."""

import torch
import numpy as np
from src.models.wav2lip import Wav2LipModel
from src.utils.device import setup_device, set_seed
from src.utils.audio import audio_to_tensor
from src.utils.video import frames_to_tensor, save_video
from src.eval.talking_head_metrics import TalkingHeadEvaluator


def create_dummy_data():
    """Create dummy audio and video data for demonstration."""
    
    # Create dummy audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, audio.shape)
    audio = audio + noise
    
    # Create dummy video (25 frames of colored patterns)
    video_frames = 25
    height, width = 64, 64
    
    frames = []
    for i in range(video_frames):
        # Create a frame with moving colored circles
        frame = np.zeros((height, width, 3), dtype=np.float32)
        
        # Moving circle
        center_x = int(width * 0.5 + 20 * np.sin(i * 0.2))
        center_y = int(height * 0.5 + 15 * np.cos(i * 0.2))
        
        # Draw circle
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 15:
                    frame[y, x] = [0.8, 0.2, 0.2]  # Red circle
        
        frames.append(frame)
    
    frames = np.array(frames)
    
    return audio, frames


def main():
    """Main demonstration function."""
    
    print("🎭 Talking Head Generation Demo")
    print("=" * 40)
    
    # Setup
    device = setup_device("auto")
    set_seed(42)
    
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        "audio_encoder": {
            "model_name": "facebook/wav2vec2-base-960h",
            "freeze_encoder": True,  # Freeze for faster demo
            "output_dim": 768
        },
        "video_encoder": {
            "input_channels": 3,
            "output_dim": 512,
            "pretrained": False  # Use random weights for demo
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
    
    # Initialize model
    print("Initializing model...")
    model = Wav2LipModel(**config).to(device)
    model.eval()
    
    # Create dummy data
    print("Creating dummy data...")
    audio, video_frames = create_dummy_data()
    
    # Convert to tensors
    audio_tensor = audio_to_tensor(audio, add_batch_dim=True).to(device)
    video_tensor = frames_to_tensor(video_frames, add_batch_dim=True).to(device)
    
    print(f"Audio shape: {audio_tensor.shape}")
    print(f"Video shape: {video_tensor.shape}")
    
    # Generate talking head
    print("Generating talking head...")
    with torch.no_grad():
        generated_video = model(audio_tensor, video_tensor)
    
    print(f"Generated video shape: {generated_video.shape}")
    
    # Evaluate quality
    print("Evaluating quality...")
    evaluator = TalkingHeadEvaluator()
    
    with torch.no_grad():
        audio_features = model.audio_encoder(audio_tensor)
        video_features = model.video_encoder(video_tensor)
        
        metrics = evaluator.evaluate(
            generated_video, video_tensor, audio_features, video_features
        )
    
    print("\nEvaluation Results:")
    print(f"  Sync Score: {metrics['sync_score']:.3f}")
    print(f"  Offset Error: {metrics['offset_error']:.3f}s")
    print(f"  PSNR: {metrics['psnr']:.2f}")
    print(f"  SSIM: {metrics['ssim']:.3f}")
    print(f"  LPIPS: {metrics['lpips']:.3f}")
    
    # Save generated video
    print("\nSaving generated video...")
    output_path = "demo_output.mp4"
    
    # Convert tensor to numpy and denormalize
    generated_np = generated_video.squeeze(0).detach().cpu().numpy()
    generated_np = (generated_np * 255).astype(np.uint8)
    
    save_video(generated_np, output_path)
    print(f"Generated video saved to: {output_path}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    main()
