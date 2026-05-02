# Talking Head Generation

Multi-Modal AI project for generating realistic talking head videos from audio input. This project implements a Wav2Lip-style model that combines audio processing with computer vision to create synchronized facial animations.

## Features

- **Audio-Vision-Text Integration**: Combines speech audio with visual facial features
- **Modern Architecture**: Uses Wav2Vec2 for audio encoding and ResNet for video processing
- **Cross-Modal Attention**: Advanced fusion mechanism between audio and video modalities
- **Comprehensive Evaluation**: Multiple metrics for sync quality and visual fidelity
- **Interactive Demo**: Streamlit-based web interface for easy experimentation
- **Production Ready**: Clean code structure with proper configuration management

## Project Structure

```
Talking-Head-Generation/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   └── wav2lip.py           # Wav2Lip-style model
│   ├── data/                     # Data loading and preprocessing
│   │   └── talking_head_dataset.py
│   ├── losses/                   # Loss functions
│   │   └── talking_head_losses.py
│   ├── eval/                     # Evaluation metrics
│   │   └── talking_head_metrics.py
│   ├── utils/                    # Utility functions
│   │   ├── device.py            # Device management
│   │   ├── audio.py             # Audio processing
│   │   └── video.py             # Video processing
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model/                   # Model configurations
│   └── data/                    # Data configurations
├── scripts/                      # Training and evaluation scripts
│   └── train.py                 # Training script
├── demo/                        # Demo applications
│   └── streamlit_app.py         # Streamlit demo
├── tests/                       # Unit tests
├── data/                        # Data directory
│   ├── audio/                   # Audio files
│   ├── video/                   # Video files
│   └── annotations.json         # Dataset annotations
├── checkpoints/                 # Model checkpoints
├── outputs/                     # Generated outputs
├── assets/                      # Static assets
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Talking-Head-Generation.git
cd Talking-Head-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Training

Train the model with default configuration:

```bash
python scripts/train.py --config configs/config.yaml
```

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/config.yaml --resume checkpoints/best_model.pt
```

### 2. Demo

Launch the interactive Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

The demo will open in your browser at `http://localhost:8501`.

### 3. Evaluation

Evaluate the model on test data:

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

## Dataset Schema

The project expects audio-video pairs with the following structure:

### Data Format

- **Audio**: WAV files, 16kHz sample rate, mono channel
- **Video**: MP4 files, 25 FPS, 256x256 resolution
- **Annotations**: JSON file with metadata

### Annotation Format

```json
[
  {
    "audio_file": "sample_001.wav",
    "video_file": "sample_001.mp4",
    "duration": 5.2,
    "speaker_id": "speaker_001",
    "text": "Hello, this is a sample sentence."
  }
]
```

### Synthetic Data

If no real dataset is available, the system automatically generates synthetic audio-video pairs for demonstration purposes.

## Model Architecture

### Wav2Lip-Style Model

The model consists of four main components:

1. **Audio Encoder**: Wav2Vec2-based encoder for speech feature extraction
2. **Video Encoder**: ResNet-based encoder for visual feature extraction
3. **Cross-Modal Fusion**: Multi-head attention mechanism for audio-video alignment
4. **Video Decoder**: UNet-based decoder for realistic video generation

### Key Features

- **Synchronization Loss**: Ensures audio-video alignment
- **Perceptual Loss**: Maintains visual quality using VGG features
- **Temporal Consistency**: Smooth video generation across frames
- **Adversarial Training**: Optional discriminator for improved realism

## Evaluation Metrics

### Synchronization Quality

- **Sync Score**: Cross-modal similarity between audio and video features
- **Offset Error**: Temporal misalignment between audio and video (in seconds)

### Visual Quality

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Example Results

| Metric | Value | Description |
|--------|-------|-------------|
| Sync Score | 0.85 | High audio-video synchronization |
| Offset Error | 0.12s | Low temporal misalignment |
| PSNR | 28.5 dB | Good visual quality |
| SSIM | 0.78 | High structural similarity |

## Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
# Project settings
project_name: "talking_head_generation"
version: "0.1.0"
seed: 42

# Device settings
device: "auto"  # auto, cuda, mps, cpu
mixed_precision: true

# Safety settings
enable_safety_checks: true
max_audio_duration: 30.0
max_video_resolution: 512
```

### Model Configuration

```yaml
# Model architecture
audio_encoder:
  type: "wav2vec2"
  pretrained: "facebook/wav2vec2-base-960h"
  output_dim: 768

video_encoder:
  type: "resnet"
  pretrained: true
  output_dim: 512

fusion:
  type: "cross_attention"
  hidden_dim: 512
  num_heads: 8
  num_layers: 4
```

## Safety and Limitations

### Safety Notice

This project is designed for research and educational purposes. Generated content should not be used for:

- Creating deepfakes or misleading content
- Impersonating real people without consent
- Any malicious or harmful purposes

### Limitations

- **Training Data**: Requires high-quality audio-video pairs
- **Computational Requirements**: GPU recommended for training
- **Real-time Performance**: Not optimized for real-time generation
- **Generalization**: May not work well with unseen speakers or languages

### Ethical Guidelines

- Always obtain proper consent before using someone's likeness
- Clearly label generated content as synthetic
- Respect privacy and intellectual property rights
- Use responsibly and ethically

## Development

### Code Quality

The project follows modern Python development practices:

- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Google-style docstrings for all functions
- **Formatting**: Black and Ruff for consistent code style
- **Testing**: Pytest for unit testing

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Wav2Lip paper and implementation
- Wav2Vec2 model from Facebook AI
- PyTorch and the open-source community
- Streamlit for the demo interface

## Citation

If you use this project in your research, please cite:

```bibtex
@software{talking_head_generation,
  title={Talking Head Generation: A Multi-Modal AI Project},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Talking-Head-Generation}
}
```

## Support

For questions, issues, or contributions, please:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Join our community discussions

---

**Disclaimer**: This is a research/educational project. Please use responsibly and ethically.
# Talking-Head-Generation
