#!/usr/bin/env python3
"""
Per-Video Cache for VASA Dataset
=================================
Each video gets its own folder with metadata H5 + frames organized together.

Structure:
    cache_dir/
        <video_md5>/
            metadata.h5              # All windows' metadata for this video
            frames/
                window_0/
                    frame_000.png
                    frame_001.png
                    ...
                window_1/
                    ...
            emo_frames/
                window_0/
                    frame_000.png
                    ...
        cache_index.json             # Fast lookup of all cached videos
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib
import json
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class PerVideoCache:
    """
    Per-video cache with MD5-indexed folders containing metadata + frames.

    Benefits:
    - One corrupted video doesn't break entire cache
    - Easy to identify and delete bad videos
    - Parallel processing support
    - Easy invalidation (delete one folder)
    - All data for a video in one place
    """

    def __init__(
        self,
        cache_dir: Path,
        compression: str = 'gzip',
        compression_level: int = 4
    ):
        """
        Initialize PerVideoCache.

        Args:
            cache_dir: Root cache directory
            compression: HDF5 compression type
            compression_level: Compression level (1-9)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.compression_level = compression_level
        self.index_path = self.cache_dir / 'cache_index.json'

        # Cache for video hashes to avoid recomputing
        self.hash_cache_path = self.cache_dir / 'video_hashes.txt'
        self._hash_cache = self._load_hash_cache()

        logger.info(f"Initialized PerVideoCache at {self.cache_dir}")

    def _load_hash_cache(self) -> Dict[str, str]:
        """Load cached video MD5 hashes."""
        cache = {}
        if self.hash_cache_path.exists():
            with open(self.hash_cache_path, 'r') as f:
                for line in f:
                    if line.strip():
                        path, hash_val = line.strip().split('\t')
                        cache[path] = hash_val
        return cache

    def _save_hash_to_cache(self, video_path: str, hash_val: str):
        """Save video MD5 hash to cache file."""
        self._hash_cache[video_path] = hash_val
        with open(self.hash_cache_path, 'a') as f:
            f.write(f"{video_path}\t{hash_val}\n")

    def get_video_hash(self, video_path: str) -> str:
        """
        Compute MD5 hash of video file for indexing.

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string
        """
        video_path_str = str(video_path)

        # Check cache first
        if video_path_str in self._hash_cache:
            return self._hash_cache[video_path_str]

        # Compute hash
        hash_md5 = hashlib.md5()
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096 * 1024), b''):
                hash_md5.update(chunk)

        hash_val = hash_md5.hexdigest()
        self._save_hash_to_cache(video_path_str, hash_val)
        return hash_val

    def get_video_dir(self, video_path: str, create: bool = False) -> Path:
        """
        Get directory for a video's cache.

        Args:
            video_path: Path to video file
            create: Whether to create directory if it doesn't exist

        Returns:
            Path to video cache directory
        """
        video_md5 = self.get_video_hash(video_path)
        video_dir = self.cache_dir / video_md5

        if create:
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / 'frames').mkdir(exist_ok=True)
            (video_dir / 'emo_frames').mkdir(exist_ok=True)

        return video_dir

    def get_metadata_path(self, video_path: str) -> Path:
        """Get path to video's metadata H5 file."""
        video_dir = self.get_video_dir(video_path)
        return video_dir / 'metadata.h5'

    def get_frames_dir(self, video_path: str, frame_type: str = 'frames') -> Path:
        """
        Get frames directory for a video.

        Args:
            video_path: Path to video file
            frame_type: 'frames' or 'emo_frames'
        """
        video_dir = self.get_video_dir(video_path)
        return video_dir / frame_type

    def has_video_cache(self, video_path: str) -> bool:
        """Check if cache exists for this video."""
        metadata_path = self.get_metadata_path(video_path)
        return metadata_path.exists()

    def save_video_windows(
        self,
        video_path: str,
        windows_data: List[Dict[str, Any]]
    ):
        """
        Save all windows for a video.

        Args:
            video_path: Path to video file
            windows_data: List of window dictionaries
        """
        if not windows_data:
            logger.warning(f"No windows to save for {video_path}")
            return

        video_dir = self.get_video_dir(video_path, create=True)
        metadata_path = video_dir / 'metadata.h5'
        temp_path = metadata_path.with_suffix('.tmp')

        try:
            with h5py.File(temp_path, 'w') as f:
                # Save metadata
                f.attrs['num_windows'] = len(windows_data)
                f.attrs['video_path'] = str(video_path)
                f.attrs['video_md5'] = self.get_video_hash(video_path)
                f.attrs['cache_version'] = '2.0'  # Per-video version

                # Save each window
                for window_idx, window_data in enumerate(windows_data):
                    window_group = f.create_group(f'window_{window_idx}')

                    # Store window index and metadata
                    window_group.attrs['window_idx'] = window_idx

                    # Save window data (excluding frames - those go to disk)
                    for key, value in window_data.items():
                        if key in ['frames', 'emo_frames']:
                            # Frames saved to disk separately
                            continue

                        elif key == 'metadata':
                            # Save metadata as attributes
                            meta_group = window_group.create_group('metadata')
                            for meta_key, meta_value in value.items():
                                if isinstance(meta_value, (str, int, float, bool)):
                                    meta_group.attrs[meta_key] = meta_value
                                elif isinstance(meta_value, (list, dict)):
                                    meta_group.attrs[meta_key] = json.dumps(meta_value)
                                elif isinstance(meta_value, (np.integer, np.int64, np.int32)):
                                    meta_group.attrs[meta_key] = int(meta_value)
                                elif isinstance(meta_value, (np.floating, np.float64, np.float32)):
                                    meta_group.attrs[meta_key] = float(meta_value)
                                elif isinstance(meta_value, np.ndarray):
                                    meta_group.attrs[meta_key] = meta_value.tolist()

                        elif key == 'emotion_label':
                            # Save emotion_label as JSON string
                            if isinstance(value, list):
                                window_group.attrs['emotion_label'] = json.dumps(value)

                        elif key == 'lip_metrics':
                            # Handle nested lip metrics
                            lip_group = window_group.create_group('lip_metrics')
                            for metric_key, metric_value in value.items():
                                if isinstance(metric_value, torch.Tensor):
                                    lip_group.create_dataset(
                                        metric_key,
                                        data=metric_value.cpu().numpy(),
                                        compression=self.compression,
                                        compression_opts=self.compression_level
                                    )

                        elif isinstance(value, torch.Tensor):
                            # Save tensor data
                            ds = window_group.create_dataset(
                                key,
                                data=value.cpu().numpy(),
                                compression=self.compression,
                                compression_opts=self.compression_level
                            )
                            ds.attrs['dtype'] = str(value.dtype)
                            ds.attrs['shape'] = value.shape

                        elif isinstance(value, np.ndarray):
                            # Save numpy array
                            window_group.create_dataset(
                                key,
                                data=value,
                                compression=self.compression,
                                compression_opts=self.compression_level
                            )

            # Atomic rename
            if temp_path.exists():
                if metadata_path.exists():
                    metadata_path.unlink()
                temp_path.rename(metadata_path)
                logger.info(f"✅ Saved {len(windows_data)} windows for video {Path(video_path).name}")

        except Exception as e:
            logger.error(f"❌ Error saving video cache: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_frames(
        self,
        video_path: str,
        window_idx: int,
        frames: torch.Tensor,
        frame_type: str = 'frames',
        format: str = 'png',
        remove_green_background: bool = False
    ):
        """
        Save frames to disk.

        Args:
            video_path: Path to video file
            window_idx: Window index
            frames: Tensor of frames [T, C, H, W] or [T, H, W, C]
            frame_type: 'frames' or 'emo_frames'
            format: Image format ('png', 'jpg')
            remove_green_background: Remove green chroma key (for EMO frames)
        """
        video_dir = self.get_video_dir(video_path, create=True)
        frames_dir = video_dir / frame_type
        window_dir = frames_dir / f'window_{window_idx}'
        window_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor to numpy
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Ensure [T, H, W, C] format
        if frames.shape[1] == 3 or frames.shape[1] == 1:  # [T, C, H, W]
            frames = np.transpose(frames, (0, 2, 3, 1))

        T = frames.shape[0]

        # Save each frame
        for t in range(T):
            frame = frames[t].copy()

            # Convert to uint8 if float
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_min, frame_max = frame.min(), frame.max()

                if frame_min >= 0.0 and frame_max <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                elif frame_min >= -1.0 and frame_max <= 1.0:
                    frame = ((frame + 1.0) * 127.5).astype(np.uint8)
                elif frame_max <= 255.0:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                else:
                    frame = ((frame - frame_min) / (frame_max - frame_min + 1e-8) * 255).astype(np.uint8)

            # Apply green screen removal if needed
            if remove_green_background and frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)
                alpha = 255 - green_mask

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gray_background = np.full_like(frame_rgb, 128, dtype=np.uint8)

                alpha_float = alpha.astype(np.float32) / 255.0
                alpha_3ch = np.stack([alpha_float, alpha_float, alpha_float], axis=-1)
                frame = (frame_rgb * alpha_3ch + gray_background * (1 - alpha_3ch)).astype(np.uint8)

            # Save image
            frame_path = window_dir / f"frame_{t:03d}.{format}"

            if frame.shape[-1] == 4:
                img = Image.fromarray(frame, mode='RGBA')
            else:
                img = Image.fromarray(frame)

            img.save(frame_path, format=format.upper())

        logger.debug(f"Saved {T} {frame_type} to {window_dir}")

    def load_frames(
        self,
        video_path: str,
        window_idx: int,
        frame_type: str = 'frames',
        as_tensor: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Load frames from disk.

        Args:
            video_path: Path to video file
            window_idx: Window index
            frame_type: 'frames' or 'emo_frames'
            as_tensor: Return as torch.Tensor

        Returns:
            Frames as tensor [T, C, H, W] or None if not found
        """
        video_dir = self.get_video_dir(video_path)
        frames_dir = video_dir / frame_type
        window_dir = frames_dir / f'window_{window_idx}'

        if not window_dir.exists():
            return None

        # Find all frame files
        frame_files = sorted(window_dir.glob(f"frame_*.png")) + sorted(window_dir.glob(f"frame_*.jpg"))

        if not frame_files:
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

        return frames

    def has_frames(
        self,
        video_path: str,
        window_idx: int,
        frame_type: str = 'frames'
    ) -> bool:
        """Check if frames exist for this window."""
        video_dir = self.get_video_dir(video_path)
        frames_dir = video_dir / frame_type
        window_dir = frames_dir / f'window_{window_idx}'

        if not window_dir.exists():
            return False

        frame_files = list(window_dir.glob("frame_*.png")) + list(window_dir.glob("frame_*.jpg"))
        return len(frame_files) > 0

    def load_window(
        self,
        video_path: str,
        window_idx: int,
        load_frames: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load a single window's metadata (and optionally frames).

        Args:
            video_path: Path to video file
            window_idx: Window index
            load_frames: Whether to load frame images from disk

        Returns:
            Window data dictionary or None if not found
        """
        metadata_path = self.get_metadata_path(video_path)

        if not metadata_path.exists():
            return None

        try:
            with h5py.File(metadata_path, 'r') as f:
                window_key = f'window_{window_idx}'
                if window_key not in f:
                    return None

                window_group = f[window_key]
                window_data = {}

                # Load emotion_label if present
                if 'emotion_label' in window_group.attrs:
                    emotion_label_json = window_group.attrs['emotion_label']
                    try:
                        window_data['emotion_label'] = json.loads(emotion_label_json)
                    except json.JSONDecodeError:
                        pass

                # Load all datasets
                for key in window_group.keys():
                    if key == 'metadata':
                        meta_group = window_group['metadata']
                        metadata = {}
                        for meta_key in meta_group.attrs:
                            value = meta_group.attrs[meta_key]
                            if isinstance(value, str) and value.startswith(('[', '{')):
                                try:
                                    metadata[meta_key] = json.loads(value)
                                except:
                                    metadata[meta_key] = value
                            else:
                                metadata[meta_key] = value
                        window_data['metadata'] = metadata

                    elif key == 'lip_metrics':
                        lip_group = window_group['lip_metrics']
                        lip_metrics = {}
                        for metric_key in lip_group.keys():
                            data = lip_group[metric_key][()]
                            lip_metrics[metric_key] = torch.from_numpy(data).float()
                        window_data['lip_metrics'] = lip_metrics

                    else:
                        dataset = window_group[key]
                        data = dataset[()]
                        tensor = torch.from_numpy(data)

                        # Restore dtype
                        if 'dtype' in dataset.attrs:
                            dtype_str = dataset.attrs['dtype']
                            if 'float32' in dtype_str:
                                tensor = tensor.float()
                            elif 'float16' in dtype_str:
                                tensor = tensor.half()
                            elif 'long' in dtype_str:
                                tensor = tensor.long()

                        window_data[key] = tensor

                # Optionally load frames from disk
                if load_frames:
                    frames = self.load_frames(video_path, window_idx, frame_type='frames')
                    if frames is not None:
                        window_data['frames'] = frames

                    emo_frames = self.load_frames(video_path, window_idx, frame_type='emo_frames')
                    if emo_frames is not None:
                        window_data['emo_frames'] = emo_frames

                return window_data

        except Exception as e:
            logger.error(f"Error loading window {window_idx} from {video_path}: {e}")
            return None

    def load_all_video_windows(
        self,
        video_path: str,
        load_frames: bool = False
    ) -> List[Dict[str, Any]]:
        """Load all windows for a video."""
        metadata_path = self.get_metadata_path(video_path)

        if not metadata_path.exists():
            return []

        try:
            with h5py.File(metadata_path, 'r') as f:
                num_windows = f.attrs.get('num_windows', 0)

            windows = []
            for i in range(num_windows):
                window = self.load_window(video_path, i, load_frames=load_frames)
                if window is not None:
                    windows.append(window)

            return windows

        except Exception as e:
            logger.error(f"Error loading windows from {video_path}: {e}")
            return []

    def get_num_windows(self, video_path: str) -> int:
        """Get number of windows for a video."""
        metadata_path = self.get_metadata_path(video_path)

        if not metadata_path.exists():
            return 0

        try:
            with h5py.File(metadata_path, 'r') as f:
                return f.attrs.get('num_windows', 0)
        except:
            return 0

    def delete_video(self, video_path: str):
        """Delete all cache data for a video."""
        video_dir = self.get_video_dir(video_path)

        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            logger.info(f"Deleted cache for video: {video_path}")

    def rebuild_index(self):
        """
        Scan cache directory and rebuild index.json.

        This provides fast lookup of all cached videos without opening H5 files.
        """
        index = {}

        # Scan all MD5 directories
        for video_dir in self.cache_dir.iterdir():
            if not video_dir.is_dir():
                continue

            video_md5 = video_dir.name
            metadata_path = video_dir / 'metadata.h5'

            if not metadata_path.exists():
                continue

            try:
                with h5py.File(metadata_path, 'r') as f:
                    num_windows = f.attrs.get('num_windows', 0)
                    video_path = f.attrs.get('video_path', 'unknown')

                # Count frames
                frames_dir = video_dir / 'frames'
                emo_frames_dir = video_dir / 'emo_frames'

                num_frame_windows = len(list(frames_dir.glob('window_*'))) if frames_dir.exists() else 0
                num_emo_frame_windows = len(list(emo_frames_dir.glob('window_*'))) if emo_frames_dir.exists() else 0

                index[video_md5] = {
                    'video_path': str(video_path),
                    'num_windows': int(num_windows),  # Convert numpy.int64 to int
                    'num_frame_windows': int(num_frame_windows),
                    'num_emo_frame_windows': int(num_emo_frame_windows),
                    'metadata_size_mb': float(metadata_path.stat().st_size / (1024**2))
                }

            except Exception as e:
                logger.warning(f"Failed to index {video_md5}: {e}")

        # Save index
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"✅ Rebuilt index with {len(index)} videos")
        return index

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the entire cache."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                index = json.load(f)
        else:
            index = self.rebuild_index()

        total_windows = sum(v['num_windows'] for v in index.values())
        total_metadata_size = sum(v['metadata_size_mb'] for v in index.values())

        return {
            'total_videos': len(index),
            'total_windows': total_windows,
            'total_metadata_size_mb': total_metadata_size,
            'videos': index
        }

    def validate_video(self, video_path: str) -> Tuple[bool, List[str]]:
        """
        Validate cache for a specific video.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not self.has_video_cache(video_path):
            return False, ["No cache exists for this video"]

        try:
            metadata_path = self.get_metadata_path(video_path)

            with h5py.File(metadata_path, 'r') as f:
                num_windows = f.attrs.get('num_windows', 0)

                # Check all windows exist
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        issues.append(f"Missing window {i}")

            if issues:
                return False, issues
            return True, []

        except Exception as e:
            return False, [f"Error reading metadata: {str(e)}"]


def main():
    """Test the per-video cache"""
    import argparse

    parser = argparse.ArgumentParser(description='Per-video cache utility')
    parser.add_argument('--cache-dir', type=str, default='cache_per_video',
                       help='Cache directory')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Rebuild cache index')
    parser.add_argument('--stats', action='store_true',
                       help='Show cache statistics')
    args = parser.parse_args()

    cache = PerVideoCache(Path(args.cache_dir))

    if args.rebuild_index:
        cache.rebuild_index()

    if args.stats:
        stats = cache.get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Total windows: {stats['total_windows']}")
        print(f"  Total metadata size: {stats['total_metadata_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
