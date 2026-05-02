"""Evaluation script for talking head generation."""

import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.wav2lip import Wav2LipModel
from src.data.talking_head_dataset import create_data_loaders
from src.eval.talking_head_metrics import TalkingHeadEvaluator
from src.utils.device import setup_device, set_seed


def evaluate_model(config, checkpoint_path):
    """Evaluate the model on test data."""
    
    # Setup device
    device = setup_device(config.get("device", "auto"))
    set_seed(config.get("seed", 42))
    
    # Load model
    model = Wav2LipModel(
        audio_encoder_config=config["model"]["audio_encoder"],
        video_encoder_config=config["model"]["video_encoder"],
        fusion_config=config["model"]["fusion"],
        decoder_config=config["model"]["decoder"]
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(
        config["data"],
        batch_size=1,  # Evaluate one sample at a time
        num_workers=0
    )
    
    # Initialize evaluator
    evaluator = TalkingHeadEvaluator()
    
    # Evaluate
    all_metrics = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            
            # Get features
            audio_features = model.audio_encoder(audio)
            video_features = model.video_encoder(video)
            
            # Generate video
            generated_video = model(audio, video)
            
            # Compute metrics
            metrics = evaluator.evaluate(
                generated_video, video, audio_features, video_features
            )
            
            all_metrics.append(metrics)
            
            print(f"Sample {batch_idx + 1}:")
            print(f"  Sync Score: {metrics['sync_score']:.3f}")
            print(f"  Offset Error: {metrics['offset_error']:.3f}s")
            print(f"  PSNR: {metrics['psnr']:.2f}")
            print(f"  SSIM: {metrics['ssim']:.3f}")
            print(f"  LPIPS: {metrics['lpips']:.3f}")
            print()
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    print("Average Metrics:")
    print(f"  Sync Score: {avg_metrics['sync_score']:.3f}")
    print(f"  Offset Error: {avg_metrics['offset_error']:.3f}s")
    print(f"  PSNR: {avg_metrics['psnr']:.2f}")
    print(f"  SSIM: {avg_metrics['ssim']:.3f}")
    print(f"  LPIPS: {avg_metrics['lpips']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate talking head generation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Evaluate model
    evaluate_model(config, args.checkpoint)


if __name__ == "__main__":
    main()
