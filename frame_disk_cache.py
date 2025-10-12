#!/usr/bin/env python3
"""
Frame disk cache using MD5-indexed folders for efficient storage.
"""

import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class FrameDiskCache:
    """
    Disk-based frame cache using MD5 hashing to avoid duplication.

    Structure:
        cache_dir/
            frames/
                <video_md5>/
                    window_0/
                        frame_000.png
                        frame_001.png
                        ...
                    window_1/
                        frame_000.png
                        ...
            emo_frames/
                <video_md5>/
                    window_0/
                        frame_000.png
                        ...
    """

    def __init__(self, cache_dir: Path, frame_type: str = 'frames'):
        """
        Initialize frame disk cache.

        Args:
            cache_dir: Root cache directory
            frame_type: 'frames' or 'emo_frames'
        """
        self.cache_dir = Path(cache_dir)
        self.frame_type = frame_type
        self.root = self.cache_dir / frame_type
        self.root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FrameDiskCache at {self.root}")

    def get_video_hash(self, video_path: str) -> str:
        """
        Compute MD5 hash of video file for indexing.

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string
        """
        video_path = Path(video_path)

        # Use cached hash if available
        cache_file = self.cache_dir / 'video_hashes.txt'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                for line in f:
                    if line.strip():
                        path, hash_val = line.strip().split('\t')
                        if path == str(video_path):
                            return hash_val

        # Compute hash
        md5 = hashlib.md5()
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)

        hash_val = md5.hexdigest()

        # Cache the hash
        with open(cache_file, 'a') as f:
            f.write(f"{video_path}\t{hash_val}\n")

        return hash_val

    def get_window_dir(self, video_path: str, window_idx: int, create: bool = False) -> Path:
        """
        Get directory for a specific window's frames.

        Args:
            video_path: Path to video file
            window_idx: Window index
            create: Whether to create directory if it doesn't exist

        Returns:
            Path to window directory
        """
        video_hash = self.get_video_hash(video_path)
        window_dir = self.root / video_hash / f"window_{window_idx}"

        if create:
            window_dir.mkdir(parents=True, exist_ok=True)

        return window_dir

    def save_frames(
        self,
        video_path: str,
        window_idx: int,
        frames: torch.Tensor,
        format: str = 'png',
        remove_green_background: bool = False,
        green_threshold: float = 0.6
    ):
        """
        Save frames to disk.

        Args:
            video_path: Path to video file
            window_idx: Window index
            frames: Tensor of frames [T, C, H, W] or [T, H, W, C]
            format: Image format ('png', 'jpg')
            remove_green_background: If True, make green chroma key pixels transparent (for EMO frames)
            green_threshold: Threshold for detecting green dominance (0.0-1.0)
        """
        window_dir = self.get_window_dir(video_path, window_idx, create=True)

        # Convert tensor to numpy
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Ensure [T, H, W, C] format
        if frames.shape[1] == 3 or frames.shape[1] == 1:  # [T, C, H, W]
            frames = np.transpose(frames, (0, 2, 3, 1))

        T = frames.shape[0]

        # Save each frame
        for t in range(T):
            frame = frames[t].copy()  # Make a copy to avoid modifying original

            # Convert to uint8 if float (do this before chroma keying)
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_min, frame_max = frame.min(), frame.max()

                # Check range and normalize appropriately
                if frame_min >= 0.0 and frame_max <= 1.0:
                    # [0, 1] range - standard normalization
                    frame = (frame * 255).astype(np.uint8)
                elif frame_min >= -1.0 and frame_max <= 1.0:
                    # [-1, 1] range - shift and scale
                    frame = ((frame + 1.0) * 127.5).astype(np.uint8)
                elif frame_max <= 255.0:
                    # Already in [0, 255] range
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                else:
                    # Unknown range - normalize to [0, 255]
                    logger.warning(f"Unusual frame value range: [{frame_min:.2f}, {frame_max:.2f}], normalizing...")
                    frame = ((frame - frame_min) / (frame_max - frame_min + 1e-8) * 255).astype(np.uint8)

            # Apply green screen removal and composite onto solid background
            # Removes green chroma key and replaces with gray background
            if remove_green_background:
                if frame.shape[-1] == 3:  # RGB image
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Convert to HSV color space for better green detection
                    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                    # Define green color range in HSV (EXACT match to debug_generated_frames)
                    lower_green = np.array([35, 40, 40])   # Lower bound for green
                    upper_green = np.array([85, 255, 255]) # Upper bound for green

                    # Create mask for green pixels (255 where green, 0 elsewhere)
                    green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

                    # Create alpha channel (255 where NOT green, 0 where green)
                    alpha = 255 - green_mask

                    # Convert back to RGB (from BGR)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Create solid gray background (128 = middle gray)
                    gray_background = np.full_like(frame_rgb, 128, dtype=np.uint8)

                    # Composite: blend foreground over gray background using alpha
                    # alpha/255 gives us blend weight [0, 1]
                    alpha_float = alpha.astype(np.float32) / 255.0
                    alpha_3ch = np.stack([alpha_float, alpha_float, alpha_float], axis=-1)

                    # Composite: fg * alpha + bg * (1 - alpha)
                    frame = (frame_rgb * alpha_3ch + gray_background * (1 - alpha_3ch)).astype(np.uint8)

            # Save as image
            frame_path = window_dir / f"frame_{t:03d}.{format}"

            # Use RGBA mode if we have 4 channels
            if frame.shape[-1] == 4:
                img = Image.fromarray(frame, mode='RGBA')
            else:
                img = Image.fromarray(frame)

            img.save(frame_path, format=format.upper())

        logger.debug(f"Saved {T} frames to {window_dir}")

    def load_frames(
        self,
        video_path: str,
        window_idx: int,
        as_tensor: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Load frames from disk.

        Args:
            video_path: Path to video file
            window_idx: Window index
            as_tensor: Return as torch.Tensor

        Returns:
            Frames as tensor [T, C, H, W] or None if not found
        """
        window_dir = self.get_window_dir(video_path, window_idx, create=False)

        if not window_dir.exists():
            logger.debug(f"Window directory not found: {window_dir}")
            return None

        # Find all frame files
        frame_files = sorted(window_dir.glob("frame_*.png")) + sorted(window_dir.glob("frame_*.jpg"))

        if not frame_files:
            logger.warning(f"No frames found in {window_dir}")
            return None

        # Load frames
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            frame = np.array(img)
            frames.append(frame)

        frames = np.stack(frames, axis=0)  # [T, H, W, C]

        if as_tensor:
            # Convert to [T, C, H, W] tensor
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            # Normalize to [0, 1] if uint8
            if frames.max() > 1.0:
                frames = frames / 255.0

        logger.debug(f"Loaded {len(frame_files)} frames from {window_dir}")
        return frames

    def has_frames(self, video_path: str, window_idx: int) -> bool:
        """
        Check if frames exist for this window.

        Args:
            video_path: Path to video file
            window_idx: Window index

        Returns:
            True if frames exist
        """
        window_dir = self.get_window_dir(video_path, window_idx, create=False)
        if not window_dir.exists():
            return False

        # Check if any frame files exist
        frame_files = list(window_dir.glob("frame_*.png")) + list(window_dir.glob("frame_*.jpg"))
        return len(frame_files) > 0

    def delete_window(self, video_path: str, window_idx: int):
        """
        Delete all frames for a window.

        Args:
            video_path: Path to video file
            window_idx: Window index
        """
        window_dir = self.get_window_dir(video_path, window_idx, create=False)
        if window_dir.exists():
            import shutil
            shutil.rmtree(window_dir)
            logger.info(f"Deleted window frames: {window_dir}")

    def delete_video(self, video_path: str):
        """
        Delete all frames for a video.

        Args:
            video_path: Path to video file
        """
        video_hash = self.get_video_hash(video_path)
        video_dir = self.root / video_hash
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            logger.info(f"Deleted all frames for video: {video_path}")

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        video_dirs = list(self.root.glob("*"))
        total_videos = len(video_dirs)
        total_windows = 0
        total_frames = 0
        total_size = 0

        for video_dir in video_dirs:
            window_dirs = list(video_dir.glob("window_*"))
            total_windows += len(window_dirs)

            for window_dir in window_dirs:
                frame_files = list(window_dir.glob("frame_*.*"))
                total_frames += len(frame_files)

                for frame_file in frame_files:
                    total_size += frame_file.stat().st_size

        return {
            'total_videos': total_videos,
            'total_windows': total_windows,
            'total_frames': total_frames,
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024**3)
        }


def main():
    """Test the frame disk cache"""
    import argparse

    parser = argparse.ArgumentParser(description='Frame disk cache utility')
    parser.add_argument('--cache-dir', type=str, default='cache_single_bucket',
                       help='Cache directory')
    parser.add_argument('--stats', action='store_true',
                       help='Show cache statistics')
    parser.add_argument('--clean', action='store_true',
                       help='Clean entire cache')
    args = parser.parse_args()

    cache = FrameDiskCache(Path(args.cache_dir), frame_type='frames')

    if args.stats:
        stats = cache.get_cache_stats()
        print(f"Cache Statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Total windows: {stats['total_windows']}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Total size: {stats['total_size_gb']:.2f} GB")

    if args.clean:
        import shutil
        if cache.root.exists():
            shutil.rmtree(cache.root)
            print(f"Cleaned cache: {cache.root}")


if __name__ == "__main__":
    main()
