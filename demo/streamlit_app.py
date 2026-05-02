"""Streamlit demo for talking head generation."""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from typing import Optional, Tuple
import av

from src.models.wav2lip import Wav2LipModel
from src.utils.device import setup_device, set_seed
from src.utils.audio import load_audio, save_audio
from src.utils.video import load_video, save_video, resize_video, normalize_frames, frames_to_tensor
from src.eval.talking_head_metrics import TalkingHeadEvaluator


class TalkingHeadDemo:
    """Demo application for talking head generation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = setup_device(config.get("device", "auto"))
        set_seed(config.get("seed", 42))
        
        # Initialize model
        self.model = Wav2LipModel(
            audio_encoder_config=config["model"]["audio_encoder"],
            video_encoder_config=config["model"]["video_encoder"],
            fusion_config=config["model"]["fusion"],
            decoder_config=config["model"]["decoder"]
        ).to(self.device)
        
        # Initialize evaluator
        self.evaluator = TalkingHeadEvaluator()
        
        # Load model weights if available
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model weights."""
        checkpoint_path = self.config.get("checkpoint_path", "checkpoints/best_model.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            st.success(f"Loaded model from {checkpoint_path}")
        else:
            st.warning("No pre-trained model found. Using randomly initialized model.")
    
    def preprocess_audio(self, audio_file) -> torch.Tensor:
        """Preprocess uploaded audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            audio, sr = load_audio(tmp_path, self.config["data"]["sample_rate"])
            
            # Pad or truncate
            max_length = int(self.config["data"]["max_audio_length"] * sr)
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                padding = np.zeros(max_length - len(audio))
                audio = np.concatenate([audio, padding])
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            return audio_tensor
            
        finally:
            os.unlink(tmp_path)
    
    def preprocess_video(self, video_file) -> torch.Tensor:
        """Preprocess uploaded video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load video
            frames, fps = load_video(tmp_path, self.config["data"]["video_fps"])
            
            # Resize video
            target_size = tuple(self.config["data"]["video_size"])
            frames = resize_video(frames, target_size)
            
            # Pad or truncate
            max_frames = int(self.config["data"]["max_video_length"] * self.config["data"]["video_fps"])
            if len(frames) > max_frames:
                frames = frames[:max_frames]
            elif len(frames) < max_frames:
                # Pad with last frame
                last_frame = frames[-1:]
                padding_frames = np.repeat(last_frame, max_frames - len(frames), axis=0)
                frames = np.concatenate([frames, padding_frames], axis=0)
            
            # Normalize frames
            frames = normalize_frames(frames)
            
            # Convert to tensor
            video_tensor = frames_to_tensor(frames, add_batch_dim=True).to(self.device)
            
            return video_tensor
            
        finally:
            os.unlink(tmp_path)
    
    def generate_talking_head(
        self,
        audio: torch.Tensor,
        reference_video: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Generate talking head video."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate video
            generated_video = self.model(audio, reference_video)
            
            # Compute metrics
            audio_features = self.model.audio_encoder(audio)
            video_features = self.model.video_encoder(reference_video)
            
            metrics = self.evaluator.evaluate(
                generated_video, reference_video, audio_features, video_features
            )
            
            return generated_video, metrics
    
    def tensor_to_video_bytes(self, video_tensor: torch.Tensor) -> bytes:
        """Convert video tensor to bytes for download."""
        # Convert tensor to numpy
        video_np = video_tensor.squeeze(0).detach().cpu().numpy()
        
        # Convert from normalized to uint8
        video_np = (video_np * 255).astype(np.uint8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            save_video(video_np, tmp_file.name)
            
            with open(tmp_file.name, "rb") as f:
                video_bytes = f.read()
            
            os.unlink(tmp_file.name)
            
            return video_bytes


def main():
    st.set_page_config(
        page_title="Talking Head Generation",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Talking Head Generation")
    st.markdown("Generate realistic talking head videos from audio input")
    
    # Safety disclaimer
    st.warning("""
    **Safety Notice**: This is a research/educational tool. Generated content should not be used for:
    - Creating deepfakes or misleading content
    - Impersonating real people without consent
    - Any malicious or harmful purposes
    
    Please use responsibly and ethically.
    """)
    
    # Load config
    config = {
        "device": "auto",
        "seed": 42,
        "model": {
            "audio_encoder": {
                "model_name": "facebook/wav2vec2-base-960h",
                "freeze_encoder": False,
                "output_dim": 768
            },
            "video_encoder": {
                "input_channels": 3,
                "output_dim": 512,
                "pretrained": True
            },
            "fusion": {
                "hidden_dim": 512,
                "num_heads": 8,
                "num_layers": 4
            },
            "decoder": {
                "input_channels": 3,
                "output_channels": 3,
                "base_channels": 64,
                "num_layers": 4
            }
        },
        "data": {
            "sample_rate": 16000,
            "video_fps": 25,
            "video_size": [256, 256],
            "max_audio_length": 10.0,
            "max_video_length": 10.0
        }
    }
    
    # Initialize demo
    demo = TalkingHeadDemo(config)
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # File upload
    st.sidebar.subheader("Upload Files")
    audio_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "m4a"],
        help="Upload an audio file (speech) to generate talking head"
    )
    
    reference_video_file = st.sidebar.file_uploader(
        "Upload Reference Video",
        type=["mp4", "avi", "mov"],
        help="Upload a reference video with a face to animate"
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_duration = st.sidebar.slider(
        "Max Duration (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=1.0
    )
    
    # Update config with user settings
    config["data"]["max_audio_length"] = max_duration
    config["data"]["max_video_length"] = max_duration
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            st.success(f"Audio uploaded: {audio_file.name}")
        else:
            st.info("Please upload an audio file")
        
        if reference_video_file is not None:
            st.video(reference_video_file)
            st.success(f"Reference video uploaded: {reference_video_file.name}")
        else:
            st.info("Please upload a reference video")
    
    with col2:
        st.subheader("Generated Output")
        
        if audio_file is not None and reference_video_file is not None:
            if st.button("Generate Talking Head", type="primary"):
                with st.spinner("Generating talking head..."):
                    try:
                        # Preprocess inputs
                        audio_tensor = demo.preprocess_audio(audio_file)
                        reference_video_tensor = demo.preprocess_video(reference_video_file)
                        
                        # Generate talking head
                        generated_video, metrics = demo.generate_talking_head(
                            audio_tensor, reference_video_tensor
                        )
                        
                        # Convert to video bytes
                        video_bytes = demo.tensor_to_video_bytes(generated_video)
                        
                        # Display generated video
                        st.video(video_bytes)
                        
                        # Display metrics
                        st.subheader("Quality Metrics")
                        col_metrics1, col_metrics2 = st.columns(2)
                        
                        with col_metrics1:
                            st.metric("Sync Score", f"{metrics['sync_score']:.3f}")
                            st.metric("PSNR", f"{metrics['psnr']:.2f}")
                        
                        with col_metrics2:
                            st.metric("Offset Error (s)", f"{metrics['offset_error']:.3f}")
                            st.metric("SSIM", f"{metrics['ssim']:.3f}")
                        
                        # Download button
                        st.download_button(
                            label="Download Generated Video",
                            data=video_bytes,
                            file_name="generated_talking_head.mp4",
                            mime="video/mp4"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating talking head: {str(e)}")
        else:
            st.info("Upload both audio and reference video to generate talking head")
    
    # Additional information
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This demo showcases talking head generation using a Wav2Lip-style model. The model:
    
    - Uses Wav2Vec2 for audio feature extraction
    - Employs ResNet for video feature extraction
    - Applies cross-modal attention for audio-video fusion
    - Generates realistic lip-synced videos using a UNet decoder
    
    **Note**: This is a simplified implementation for educational purposes. 
    Real-world applications would require more sophisticated models and training data.
    """)


if __name__ == "__main__":
    main()
