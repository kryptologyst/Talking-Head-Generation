"""Video processing utilities for talking head generation."""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List, Union
from PIL import Image
import av


def load_video(
    file_path: str,
    fps: Optional[int] = None,
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Load video file and return frames and fps.
    
    Args:
        file_path: Path to video file
        fps: Target fps (None to keep original)
        max_frames: Maximum number of frames to load
        
    Returns:
        Tuple of (frames, fps)
    """
    container = av.open(file_path)
    video_stream = container.streams.video[0]
    
    frames = []
    frame_count = 0
    
    for frame in container.decode(video_stream):
        if max_frames and frame_count >= max_frames:
            break
            
        # Convert to RGB
        rgb_frame = frame.to_rgb().to_ndarray()
        frames.append(rgb_frame)
        frame_count += 1
    
    container.close()
    
    frames = np.array(frames)
    original_fps = video_stream.average_rate
    
    # Resample if needed
    if fps and fps != original_fps:
        frames = resample_video_frames(frames, original_fps, fps)
    
    return frames, fps or original_fps


def resample_video_frames(
    frames: np.ndarray,
    original_fps: float,
    target_fps: float
) -> np.ndarray:
    """Resample video frames to target fps.
    
    Args:
        frames: Input video frames
        original_fps: Original fps
        target_fps: Target fps
        
    Returns:
        Resampled frames
    """
    if original_fps == target_fps:
        return frames
    
    ratio = target_fps / original_fps
    target_length = int(len(frames) * ratio)
    
    indices = np.linspace(0, len(frames) - 1, target_length, dtype=int)
    return frames[indices]


def resize_video(
    frames: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize video frames to target size.
    
    Args:
        frames: Input video frames
        target_size: Target (width, height)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized frames
    """
    resized_frames = []
    
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size, interpolation=interpolation)
        resized_frames.append(resized_frame)
    
    return np.array(resized_frames)


def crop_face_region(
    frame: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
    margin: float = 0.2
) -> np.ndarray:
    """Crop face region from frame.
    
    Args:
        frame: Input frame
        face_box: Face bounding box (x, y, w, h)
        margin: Margin around face box
        
    Returns:
        Cropped frame
    """
    if face_box is None:
        # Use center crop as fallback
        h, w = frame.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        return frame[start_h:start_h + size, start_w:start_w + size]
    
    x, y, w, h = face_box
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(frame.shape[1], x + w + margin_w)
    y2 = min(frame.shape[0], y + h + margin_h)
    
    return frame[y1:y2, x1:x2]


def normalize_frames(
    frames: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize video frames.
    
    Args:
        frames: Input frames
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized frames
    """
    frames = frames.astype(np.float32) / 255.0
    
    for i in range(3):
        frames[:, :, :, i] = (frames[:, :, :, i] - mean[i]) / std[i]
    
    return frames


def frames_to_tensor(
    frames: np.ndarray,
    add_batch_dim: bool = True
) -> torch.Tensor:
    """Convert frames to PyTorch tensor.
    
    Args:
        frames: Input frames
        add_batch_dim: Whether to add batch dimension
        
    Returns:
        PyTorch tensor
    """
    # Convert to CHW format
    if frames.shape[-1] == 3:  # HWC format
        frames = np.transpose(frames, (0, 3, 1, 2))  # TCHW
    
    tensor = torch.from_numpy(frames).float()
    
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)  # BTCHW
    
    return tensor


def tensor_to_frames(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to frames.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Frames as numpy array
    """
    if tensor.dim() == 5:  # BTCHW
        tensor = tensor.squeeze(0)  # TCHW
    
    frames = tensor.detach().cpu().numpy()
    
    # Convert to HWC format
    if frames.shape[1] == 3:  # TCHW format
        frames = np.transpose(frames, (0, 2, 3, 1))  # THWC
    
    return frames


def save_video(
    frames: Union[np.ndarray, torch.Tensor],
    file_path: str,
    fps: int = 25,
    codec: str = 'mp4v'
) -> None:
    """Save frames as video file.
    
    Args:
        frames: Video frames
        file_path: Output file path
        fps: Frames per second
        codec: Video codec
    """
    if isinstance(frames, torch.Tensor):
        frames = tensor_to_frames(frames)
    
    # Ensure frames are in correct format
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    h, w = frames.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()


def extract_optical_flow(
    frames: np.ndarray,
    method: str = 'farneback'
) -> np.ndarray:
    """Extract optical flow between consecutive frames.
    
    Args:
        frames: Input video frames
        method: Optical flow method ('farneback', 'lucas_kanade')
        
    Returns:
        Optical flow vectors
    """
    flows = []
    
    for i in range(len(frames) - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        
        if method == 'farneback':
            flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, None, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        flows.append(flow)
    
    return np.array(flows)


def detect_face_landmarks(
    frame: np.ndarray,
    face_cascade_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """Detect face landmarks in frame.
    
    Args:
        frame: Input frame
        face_cascade_path: Path to face cascade file
        
    Returns:
        Face landmarks or None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Load face cascade
    if face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Return the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    return largest_face
