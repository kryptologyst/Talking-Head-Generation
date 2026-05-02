"""Training script for talking head generation."""

import os
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

from src.models.wav2lip import Wav2LipModel
from src.data.talking_head_dataset import create_data_loaders
from src.losses.talking_head_losses import CombinedLoss
from src.eval.talking_head_metrics import TalkingHeadEvaluator
from src.utils.device import setup_device, set_seed, get_device_info
from src.utils.video import save_video
from src.utils.audio import save_audio


class TalkingHeadTrainer:
    """Trainer for talking head generation model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = setup_device(config.get("device", "auto"))
        
        # Set random seed
        set_seed(config.get("seed", 42))
        
        # Initialize model
        self.model = Wav2LipModel(
            audio_encoder_config=config["model"]["audio_encoder"],
            video_encoder_config=config["model"]["video_encoder"],
            fusion_config=config["model"]["fusion"],
            decoder_config=config["model"]["decoder"]
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            sync_weight=config["model"]["sync_loss_weight"],
            reconstruction_weight=config["model"]["reconstruction_loss_weight"],
            perceptual_weight=config["model"]["perceptual_loss_weight"],
            adversarial_weight=config["model"]["adversarial_loss_weight"]
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["model"]["learning_rate"],
            weight_decay=config["model"]["weight_decay"]
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["model"]["max_epochs"]
        )
        
        # Initialize evaluator
        self.evaluator = TalkingHeadEvaluator()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config["data"],
            batch_size=config["model"]["batch_size"],
            num_workers=config.get("num_workers", 4)
        )
        
        # Initialize logging
        if config.get("use_wandb", False):
            wandb.init(
                project=config["project_name"],
                config=config,
                name=f"talking_head_{config['version']}"
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch["audio"].to(self.device)
            video = batch["video"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get features for loss computation
            audio_features = self.model.audio_encoder(audio)
            video_features = self.model.video_encoder(video)
            
            # Generate video
            generated_video = self.model(audio, video)
            
            # Compute loss
            losses = self.criterion(
                generated_video=generated_video,
                target_video=video,
                audio_features=audio_features,
                video_features=video_features,
                is_training=True
            )
            
            # Backward pass
            losses["total"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "sync": f"{losses['sync'].item():.4f}",
                "recon": f"{losses['reconstruction'].item():.4f}"
            })
            
            epoch_losses.append({k: v.item() for k, v in losses.items()})
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                audio = batch["audio"].to(self.device)
                video = batch["video"].to(self.device)
                
                # Get features
                audio_features = self.model.audio_encoder(audio)
                video_features = self.model.video_encoder(video)
                
                # Generate video
                generated_video = self.model(audio, video)
                
                # Compute loss
                losses = self.criterion(
                    generated_video=generated_video,
                    target_video=video,
                    audio_features=audio_features,
                    video_features=video_features,
                    is_training=False
                )
                
                val_losses.append({k: v.item() for k, v in losses.items()})
                
                # Compute metrics
                metrics = self.evaluator.evaluate_batch(
                    generated_video, video, audio_features, video_features
                )
                val_metrics.append(metrics)
        
        # Average losses and metrics
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])
        
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([metric[key] for metric in val_metrics])
        
        return {**avg_losses, **avg_metrics}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Device info: {get_device_info()}")
        
        for epoch in range(self.config["model"]["max_epochs"]):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_results['total']:.4f}")
            print(f"  Val Sync Score: {val_results.get('sync_score', 0):.4f}")
            print(f"  Val PSNR: {val_results.get('psnr', 0):.4f}")
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                log_dict = {f"train/{k}": v for k, v in train_losses.items()}
                log_dict.update({f"val/{k}": v for k, v in val_results.items()})
                log_dict["epoch"] = epoch
                wandb.log(log_dict)
            
            # Save checkpoint
            is_best = val_results["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results["total"]
            
            self.save_checkpoint(epoch, is_best)
            
            # Save sample outputs
            if epoch % 10 == 0:
                self.save_sample_outputs(epoch)
    
    def save_sample_outputs(self, epoch: int):
        """Save sample generated videos."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a sample batch
            batch = next(iter(self.val_loader))
            audio = batch["audio"][:2].to(self.device)  # Take first 2 samples
            video = batch["video"][:2].to(self.device)
            
            # Generate video
            generated_video = self.model(audio, video)
            
            # Save outputs
            output_dir = self.config.get("output_dir", "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(2):
                # Save generated video
                gen_path = os.path.join(output_dir, f"generated_epoch_{epoch}_sample_{i}.mp4")
                save_video(generated_video[i], gen_path)
                
                # Save target video
                target_path = os.path.join(output_dir, f"target_epoch_{epoch}_sample_{i}.mp4")
                save_video(video[i], target_path)


def main():
    parser = argparse.ArgumentParser(description="Train talking head generation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create trainer
    trainer = TalkingHeadTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
