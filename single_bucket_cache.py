#!/usr/bin/env python3
"""
Single Bucket Cache for VASA Dataset
====================================
Stores all windows from all videos in a single cache file for faster access.
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class SingleBucketCache:
    """
    Cache all windows in a single H5 file for faster training.

    Features:
    - All windows stored in one file
    - Efficient random access
    - Includes face attributes (gaze, emotion, head_distance)
    - Metadata tracking for each window
    """

    def __init__(
        self,
        cache_dir: Path,
        cache_name: str = "all_windows_cache.h5",
        compression: str = 'gzip',
        compression_level: int = 4
    ):
        """
        Initialize SingleBucketCache.

        Args:
            cache_dir: Directory for cache file
            cache_name: Name of the cache file
            compression: HDF5 compression type
            compression_level: Compression level (1-9)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / cache_name
        self.compression = compression
        self.compression_level = compression_level

        logger.info(f"Initialized SingleBucketCache at {self.cache_path}")

    def has_cache(self) -> bool:
        """Check if cache file exists."""
        return self.cache_path.exists()

    def get_num_windows(self) -> int:
        """Get total number of cached windows."""
        if not self.has_cache():
            return 0

        try:
            with h5py.File(self.cache_path, 'r') as f:
                return f.attrs.get('num_windows', 0)
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return 0

    def save_all_windows(
        self,
        windows_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save all windows to single cache file.

        Args:
            windows_data: List of window dictionaries
            metadata: Optional global metadata
        """
        temp_path = self.cache_path.with_suffix('.tmp')

        try:
            with h5py.File(temp_path, 'w') as f:
                # Save global metadata
                f.attrs['num_windows'] = len(windows_data)
                f.attrs['cache_version'] = '1.0'

                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            f.attrs[key] = value

                # Save each window
                for i, window in enumerate(windows_data):
                    if i % 100 == 0:
                        logger.info(f"Saving window {i}/{len(windows_data)}")

                    window_group = f.create_group(f'window_{i}')

                    # Save window data
                    for key, value in window.items():
                        if key == 'metadata':
                            # Save metadata as attributes
                            meta_group = window_group.create_group('metadata')
                            for meta_key, meta_value in value.items():
                                if isinstance(meta_value, (str, int, float, bool)):
                                    meta_group.attrs[meta_key] = meta_value
                                elif isinstance(meta_value, (list, dict)):
                                    meta_group.attrs[meta_key] = json.dumps(meta_value)

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
                            # Handle frames specially - only save first frame as identity_frame
                            if key == 'frames':
                                if len(value) > 0:
                                    # Save only the first frame as identity_frame
                                    ds = window_group.create_dataset(
                                        'identity_frame',
                                        data=value[0].cpu().numpy(),  # Just the first frame
                                        compression=self.compression,
                                        compression_opts=self.compression_level
                                    )
                                    ds.attrs['dtype'] = str(value.dtype)
                                    ds.attrs['shape'] = value[0].shape
                            else:
                                # Save other tensor data normally
                                ds = window_group.create_dataset(
                                    key,
                                    data=value.cpu().numpy(),
                                    compression=self.compression,
                                    compression_opts=self.compression_level
                                )
                                # Store original dtype
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
                if self.cache_path.exists():
                    self.cache_path.unlink()
                temp_path.rename(self.cache_path)
                logger.info(f"Successfully saved {len(windows_data)} windows to {self.cache_path}")

        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_window(self, window_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a single window by index.

        Args:
            window_idx: Index of window to load

        Returns:
            Window data dictionary or None if not found
        """
        if not self.has_cache():
            return None

        try:
            with h5py.File(self.cache_path, 'r') as f:
                window_key = f'window_{window_idx}'
                if window_key not in f:
                    return None

                window_group = f[window_key]
                window_data = {}

                # Load all data from window
                for key in window_group.keys():
                    if key == 'identity_frame':
                        # Load identity frame and expand to full frames tensor
                        dataset = window_group[key]
                        identity_frame_data = dataset[()]
                        identity_frame = torch.from_numpy(identity_frame_data)

                        # Restore original dtype if stored
                        if 'dtype' in dataset.attrs:
                            dtype_str = dataset.attrs['dtype']
                            if 'float32' in dtype_str:
                                identity_frame = identity_frame.float()
                            elif 'float16' in dtype_str:
                                identity_frame = identity_frame.half()

                        # Duplicate identity frame for all frame positions (assuming window_size=50)
                        # This ensures compatibility with models expecting full frame sequences
                        window_data['frames'] = identity_frame.unsqueeze(0).repeat(50, 1, 1, 1)

                    elif key == 'metadata':
                        # Load metadata
                        meta_group = window_group['metadata']
                        metadata = {}
                        for meta_key in meta_group.attrs:
                            value = meta_group.attrs[meta_key]
                            # Try to parse JSON if it's a string
                            if isinstance(value, str) and value.startswith(('[', '{')):
                                try:
                                    metadata[meta_key] = json.loads(value)
                                except:
                                    metadata[meta_key] = value
                            else:
                                metadata[meta_key] = value
                        window_data['metadata'] = metadata

                    elif key == 'lip_metrics':
                        # Load lip metrics
                        lip_group = window_group['lip_metrics']
                        lip_metrics = {}
                        for metric_key in lip_group.keys():
                            data = lip_group[metric_key][()]
                            lip_metrics[metric_key] = torch.from_numpy(data).float()
                        window_data['lip_metrics'] = lip_metrics

                    else:
                        # Load tensor data
                        dataset = window_group[key]
                        data = dataset[()]
                        tensor = torch.from_numpy(data)

                        # Restore original dtype if stored
                        if 'dtype' in dataset.attrs:
                            dtype_str = dataset.attrs['dtype']
                            if 'float32' in dtype_str:
                                tensor = tensor.float()
                            elif 'float16' in dtype_str:
                                tensor = tensor.half()
                            elif 'long' in dtype_str:
                                tensor = tensor.long()

                        window_data[key] = tensor

                return window_data

        except Exception as e:
            logger.error(f"Error loading window {window_idx}: {str(e)}")
            return None

    def load_all_windows(self) -> List[Dict[str, torch.Tensor]]:
        """
        Load all windows from cache.

        Returns:
            List of all window dictionaries
        """
        if not self.has_cache():
            return []

        try:
            windows = []
            num_windows = self.get_num_windows()

            for i in range(num_windows):
                if i % 100 == 0:
                    logger.info(f"Loading window {i}/{num_windows}")

                window_data = self.load_window(i)
                if window_data is not None:
                    windows.append(window_data)

            logger.info(f"Loaded {len(windows)} windows from cache")
            return windows

        except Exception as e:
            logger.error(f"Error loading windows: {str(e)}")
            return []

    def append_windows(
        self,
        new_windows: List[Dict[str, Any]]
    ):
        """
        Append new windows to existing cache without loading all into memory.

        Args:
            new_windows: List of new window dictionaries to append
        """
        if not self.has_cache():
            # If no cache exists, just save the new windows
            self.save_all_windows(new_windows)
            return

        # Get current number of windows without loading them
        current_num_windows = self.get_num_windows()

        # Append directly to the HDF5 file without loading existing windows
        try:
            with h5py.File(self.cache_path, 'a') as f:  # 'a' mode for append
                # Add new windows starting from the next available index
                for i, window_data in enumerate(new_windows):
                    window_idx = current_num_windows + i
                    window_group_name = f'window_{window_idx}'

                    if window_group_name in f:
                        del f[window_group_name]  # Remove if exists

                    window_group = f.create_group(window_group_name)

                    # Save each tensor/data in the window
                    for key, value in window_data.items():
                        if key == 'frames':
                            # Store only identity frame to save space
                            identity_frame = value[0] if value.ndim == 4 else value
                            dataset = window_group.create_dataset(
                                'identity_frame',
                                data=identity_frame.numpy() if isinstance(identity_frame, torch.Tensor) else identity_frame,
                                compression='gzip',
                                compression_opts=1
                            )
                            if isinstance(identity_frame, torch.Tensor):
                                dataset.attrs['dtype'] = str(identity_frame.dtype)

                        elif key == 'metadata':
                            # Store metadata as attributes
                            meta_group = window_group.create_group('metadata')
                            for meta_key, meta_value in value.items():
                                if isinstance(meta_value, (list, dict)):
                                    meta_group.attrs[meta_key] = json.dumps(meta_value)
                                elif isinstance(meta_value, (np.integer, np.int64, np.int32)):
                                    # Convert numpy integers to Python int
                                    meta_group.attrs[meta_key] = int(meta_value)
                                elif isinstance(meta_value, (np.floating, np.float64, np.float32)):
                                    # Convert numpy floats to Python float
                                    meta_group.attrs[meta_key] = float(meta_value)
                                elif isinstance(meta_value, np.ndarray):
                                    # Convert numpy arrays to lists
                                    meta_group.attrs[meta_key] = meta_value.tolist()
                                else:
                                    meta_group.attrs[meta_key] = meta_value

                        elif key == 'lip_metrics':
                            # Store lip metrics
                            lip_group = window_group.create_group('lip_metrics')
                            for metric_key, metric_value in value.items():
                                if isinstance(metric_value, torch.Tensor):
                                    lip_group.create_dataset(
                                        metric_key,
                                        data=metric_value.numpy(),
                                        compression='gzip'
                                    )

                        elif isinstance(value, torch.Tensor):
                            # Store tensor data - move to CPU first if on CUDA
                            tensor_data = value.cpu().numpy() if value.is_cuda else value.numpy()
                            dataset = window_group.create_dataset(
                                key,
                                data=tensor_data,
                                compression='gzip',
                                compression_opts=1
                            )
                            dataset.attrs['dtype'] = str(value.dtype)
                            dataset.attrs['shape'] = value.shape

                # Update the total window count
                if 'metadata' not in f.attrs:
                    f.attrs['metadata'] = json.dumps({})

                metadata = json.loads(f.attrs.get('metadata', '{}'))
                # Ensure the count is a Python int, not numpy int64
                metadata['num_windows'] = int(current_num_windows + len(new_windows))
                f.attrs['metadata'] = json.dumps(metadata)

            logger.info(f"Appended {len(new_windows)} windows, total: {current_num_windows + len(new_windows)}")

        except Exception as e:
            logger.error(f"Error appending windows: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Try to identify the problematic data type
            if "JSON" in str(e):
                logger.error("JSON serialization issue detected. Checking data types...")
                for i, window_data in enumerate(new_windows[:1]):  # Check first window
                    if 'metadata' in window_data:
                        for k, v in window_data['metadata'].items():
                            logger.debug(f"  metadata[{k}]: type={type(v)}, value={v}")

            # Fall back to the old method if direct append fails
            logger.warning("Falling back to load-all-and-save method (memory intensive)")
            existing_windows = self.load_all_windows()
            all_windows = existing_windows + new_windows
            self.save_all_windows(all_windows)
            logger.info(f"Appended {len(new_windows)} windows, total: {len(all_windows)}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        if not self.has_cache():
            return {
                'exists': False,
                'path': str(self.cache_path)
            }

        try:
            with h5py.File(self.cache_path, 'r') as f:
                info = {
                    'exists': True,
                    'path': str(self.cache_path),
                    'num_windows': f.attrs.get('num_windows', 0),
                    'cache_version': f.attrs.get('cache_version', 'unknown'),
                    'file_size_mb': self.cache_path.stat().st_size / (1024 * 1024)
                }

                # Get sample window info
                if 'window_0' in f:
                    window_0 = f['window_0']
                    info['sample_keys'] = list(window_0.keys())

                    # Check for face attributes
                    has_face_attrs = all(
                        key in window_0
                        for key in ['gaze', 'emotion', 'head_distance']
                    )
                    info['has_face_attributes'] = has_face_attrs

                return info

        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {
                'exists': True,
                'path': str(self.cache_path),
                'error': str(e)
            }

    def clear_cache(self):
        """Delete the cache file."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info(f"Deleted cache file: {self.cache_path}")

    def validate_cache(self) -> Tuple[bool, List[str]]:
        """
        Validate cache integrity.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not self.has_cache():
            return False, ["Cache file does not exist"]

        try:
            with h5py.File(self.cache_path, 'r') as f:
                num_windows = f.attrs.get('num_windows', 0)

                # Check all windows exist
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        issues.append(f"Missing window {i}")
                    else:
                        # Check essential keys
                        window = f[window_key]
                        essential_keys = ['frames', 'audio_features', 'gaze', 'emotion']
                        for key in essential_keys:
                            if key not in window:
                                issues.append(f"Window {i} missing key: {key}")

                if issues:
                    return False, issues
                return True, []

        except Exception as e:
            return False, [f"Error reading cache: {str(e)}"]

    def invalidate_video(self, video_path: str) -> int:
        """
        Remove all windows from a specific video from the cache.

        Args:
            video_path: Path to the video to invalidate

        Returns:
            Number of windows removed
        """
        if not self.has_cache():
            logger.warning("No cache to invalidate")
            return 0

        try:
            # Load all windows
            all_windows = []
            removed_count = 0

            with h5py.File(self.cache_path, 'r') as f:
                num_windows = f.attrs.get('num_windows', 0)

                for i in range(num_windows):
                    window = self.load_window(i)
                    if window is None:
                        continue

                    # Check if this window belongs to the video to invalidate
                    window_video_path = window.get('metadata', {}).get('video_path', '')

                    if window_video_path == video_path:
                        removed_count += 1
                        logger.info(f"Removing window {i} from video {video_path}")
                    else:
                        all_windows.append(window)

            if removed_count > 0:
                # Resave cache without the invalidated windows
                logger.info(f"Rebuilding cache without {removed_count} windows from {video_path}")

                # Preserve original metadata
                metadata = self.get_cache_info()
                metadata['total_windows'] = metadata.get('total_windows', 0) - removed_count
                metadata['successful_windows'] = len(all_windows)

                self.save_all_windows(all_windows, metadata)
                logger.info(f"✅ Invalidated {removed_count} windows from {video_path}")
            else:
                logger.info(f"No windows found for video {video_path}")

            return removed_count

        except Exception as e:
            logger.error(f"Error invalidating video: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    def validate_window_quality(self, window_idx: int) -> Tuple[bool, List[str]]:
        """
        Check if a specific window meets quality thresholds.

        Args:
            window_idx: Index of window to check

        Returns:
            Tuple of (is_valid, list_of_reasons)
        """
        window = self.load_window(window_idx)
        if window is None:
            return False, ["Window could not be loaded"]

        issues = []

        # Check 1: UV warp magnitude
        if 'uv_warps' in window:
            uv_warps = window['uv_warps']
            uv_magnitude = uv_warps.abs().mean().item()
            if uv_magnitude < 0.15:
                issues.append(f"UV magnitude too low: {uv_magnitude:.4f} < 0.15")

        # Check 2: Expression variance
        if 'expression_embed' in window:
            expr_std = window['expression_embed'].std().item()
            if expr_std < 0.01:
                issues.append(f"Expression variance too low: {expr_std:.6f} < 0.01")

        # Check 3: Valid landmarks
        if 'landmarks' in window:
            landmarks = window['landmarks']
            valid_frames = (landmarks.abs().sum(dim=-1).sum(dim=-1) > 0).float().mean().item()
            if valid_frames < 0.5:
                issues.append(f"Valid landmarks too low: {valid_frames:.1%} < 50%")

        return len(issues) == 0, issues

    def validate_all_quality(self) -> Dict[str, Any]:
        """
        Validate quality of all windows and return report.

        Returns:
            Dictionary with validation results
        """
        if not self.has_cache():
            return {"error": "No cache exists"}

        num_windows = self.get_num_windows()
        bad_windows = []
        bad_videos = set()

        logger.info(f"Validating quality of {num_windows} windows...")

        for i in range(num_windows):
            is_valid, issues = self.validate_window_quality(i)
            if not is_valid:
                window = self.load_window(i)
                video_path = window.get('metadata', {}).get('video_path', 'unknown')
                bad_windows.append({
                    'window_idx': i,
                    'video_path': video_path,
                    'issues': issues
                })
                bad_videos.add(video_path)

                if len(bad_windows) <= 10:
                    logger.warning(f"❌ Window {i} from {video_path}: {', '.join(issues)}")

            if (i + 1) % 100 == 0:
                logger.info(f"Validated {i + 1}/{num_windows} windows...")

        report = {
            'total_windows': num_windows,
            'bad_windows': len(bad_windows),
            'bad_videos': list(bad_videos),
            'bad_video_count': len(bad_videos),
            'quality_pass_rate': (num_windows - len(bad_windows)) / num_windows if num_windows > 0 else 0,
            'details': bad_windows
        }

        logger.info(f"Quality validation complete:")
        logger.info(f"  Total windows: {num_windows}")
        logger.info(f"  Bad windows: {len(bad_windows)}")
        logger.info(f"  Bad videos: {len(bad_videos)}")
        logger.info(f"  Pass rate: {report['quality_pass_rate']:.1%}")

        return report