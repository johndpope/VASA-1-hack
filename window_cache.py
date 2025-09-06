#!/usr/bin/env python3
"""
Window Cache with Chunking Support for Flexible Video Processing
=================================================================
Handles caching of video data in chunks to support varying window sizes
and long videos without memory constraints.
"""

import hashlib
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class WindowCache:
    """
    Chunked caching system for video data that supports flexible window sizes.
    
    Key features:
    - Chunks video data into manageable segments (default 1000 frames)
    - Includes overlap between chunks for context preservation
    - Loads only relevant chunks based on requested windows
    - Supports dynamic window_size and stride adjustments
    """
    
    def __init__(
        self, 
        cache_dir: Path, 
        chunk_size: int = 1000,
        overlap_size: int = 50,
        max_memory_cache: int = 5  # Maximum chunks to keep in memory
    ):
        """
        Initialize WindowCache with chunking parameters.
        
        Args:
            cache_dir: Directory for storing cache files
            chunk_size: Number of frames per chunk
            overlap_size: Overlap between chunks for context
            max_memory_cache: Maximum number of chunks to keep in memory
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_memory_cache = max_memory_cache
        self.chunk_cache = {}  # In-memory cache for loaded chunks
        self.cache_access_order = []  # Track access order for LRU eviction
        
        logger.info(f"Initialized WindowCache with chunk_size={chunk_size}, overlap={overlap_size}")
    
    def _get_video_hash(self, video_path: str) -> str:
        """Generate unique hash for video path."""
        return hashlib.md5(video_path.encode()).hexdigest()
    
    def _get_cache_path(self, video_path: str, chunk_idx: int) -> Path:
        """Generate unique cache file path for a specific chunk."""
        video_hash = self._get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}_chunk_{chunk_idx}.h5"
    
    def _get_metadata_path(self, video_path: str) -> Path:
        """Generate path for video metadata file."""
        video_hash = self._get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}_metadata.h5"
    
    def has_cache(self, video_path: str) -> bool:
        """Check if any cache exists for the video."""
        # Check if chunk 0 exists (which contains all windows for now)
        cache_path = self._get_cache_path(video_path, 0)
        return cache_path.exists()
    
    def _manage_memory_cache(self):
        """Manage in-memory cache size using LRU eviction."""
        while len(self.chunk_cache) > self.max_memory_cache:
            # Remove least recently used chunk
            if self.cache_access_order:
                oldest_key = self.cache_access_order.pop(0)
                if oldest_key in self.chunk_cache:
                    del self.chunk_cache[oldest_key]
                    logger.debug(f"Evicted chunk from memory: {oldest_key}")
    
    def _load_chunk(self, video_path: str, chunk_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a specific chunk from HDF5 file with LRU caching.
        
        Args:
            video_path: Path to the video file
            chunk_idx: Index of the chunk to load
            
        Returns:
            Dictionary containing chunk data or None if not found
        """
        cache_key = (video_path, chunk_idx)
        
        # Check memory cache first
        if cache_key in self.chunk_cache:
            # Move to end of access order (most recently used)
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.chunk_cache[cache_key]
        
        cache_path = self._get_cache_path(video_path, chunk_idx)
        if not cache_path.exists():
            logger.warning(f"Chunk {chunk_idx} not found: {cache_path}")
            return None
        
        chunk_data = {}
        try:
            with h5py.File(cache_path, 'r') as f:
                # Verify metadata
                if 'video_path' not in f.attrs or f.attrs['video_path'] != video_path:
                    logger.warning("Chunk metadata mismatch")
                    return None
                
                # Load metadata
                chunk_data['metadata'] = {
                    'video_path': video_path,
                    'start_frame': f.attrs['start_frame'],
                    'end_frame': f.attrs['end_frame'],
                    'chunk_idx': chunk_idx,
                    'actual_frames': f.attrs.get('actual_frames', 0)
                }
                
                # Load tensor data
                for key in f.keys():
                    data = f[key][:]
                    chunk_data[key] = torch.from_numpy(data).float()
                
            # Add to memory cache
            self.chunk_cache[cache_key] = chunk_data
            self.cache_access_order.append(cache_key)
            self._manage_memory_cache()
            
            logger.debug(f"Loaded chunk {chunk_idx} for {video_path}")
            return chunk_data
            
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_idx}: {str(e)}")
            return None
    
    def save_chunk(
        self, 
        video_path: str, 
        chunk_idx: int, 
        chunk_data: Dict[str, Any],
        start_frame: int,
        end_frame: int
    ):
        """
        Save a chunk of video data with metadata.
        
        Args:
            video_path: Path to the video file
            chunk_idx: Index of the chunk
            chunk_data: Dictionary containing tensor data
            start_frame: Starting frame index
            end_frame: Ending frame index
        """
        cache_path = self._get_cache_path(video_path, chunk_idx)
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            with h5py.File(temp_path, 'w') as f:
                # Save metadata
                f.attrs['video_path'] = video_path
                f.attrs['start_frame'] = start_frame
                f.attrs['end_frame'] = end_frame
                f.attrs['chunk_idx'] = chunk_idx
                f.attrs['actual_frames'] = end_frame - start_frame
                
                # Save tensor data
                for key, tensor in chunk_data.items():
                    if isinstance(tensor, torch.Tensor):
                        f.create_dataset(
                            key, 
                            data=tensor.cpu().numpy(),
                            compression='gzip',
                            compression_opts=4  # Moderate compression
                        )
            
            # Atomic rename
            if temp_path.exists():
                if cache_path.exists():
                    cache_path.unlink()
                temp_path.rename(cache_path)
                logger.debug(f"Saved chunk {chunk_idx} to {cache_path}")
                
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_idx}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
    
    def save_metadata(
        self,
        video_path: str,
        total_frames: int,
        fps: float,
        num_chunks: int,
        window_params: Dict[str, Any]
    ):
        """
        Save video metadata including chunking information.
        
        Args:
            video_path: Path to the video file
            total_frames: Total number of frames in video
            fps: Frames per second
            num_chunks: Number of chunks created
            window_params: Parameters used for windowing
        """
        metadata_path = self._get_metadata_path(video_path)
        
        try:
            with h5py.File(metadata_path, 'w') as f:
                f.attrs['video_path'] = video_path
                f.attrs['total_frames'] = total_frames
                f.attrs['fps'] = fps
                f.attrs['num_chunks'] = num_chunks
                f.attrs['chunk_size'] = self.chunk_size
                f.attrs['overlap_size'] = self.overlap_size
                
                # Save window parameters for validation
                for key, value in window_params.items():
                    f.attrs[f'window_{key}'] = value
                    
            logger.info(f"Saved metadata for {video_path}: {num_chunks} chunks, {total_frames} frames")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def load_metadata(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Load video metadata."""
        metadata_path = self._get_metadata_path(video_path)
        
        if not metadata_path.exists():
            return None
        
        try:
            with h5py.File(metadata_path, 'r') as f:
                metadata = {
                    'video_path': f.attrs['video_path'],
                    'total_frames': f.attrs['total_frames'],
                    'fps': f.attrs['fps'],
                    'num_chunks': f.attrs['num_chunks'],
                    'chunk_size': f.attrs.get('chunk_size', self.chunk_size),
                    'overlap_size': f.attrs.get('overlap_size', self.overlap_size)
                }
                
                # Load window parameters
                window_params = {}
                for key in f.attrs.keys():
                    if key.startswith('window_'):
                        window_params[key[7:]] = f.attrs[key]
                metadata['window_params'] = window_params
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    def generate_chunks(
        self,
        video_path: str,
        video_data: Dict[str, torch.Tensor],
        metadata: Dict[str, Any]
    ) -> int:
        """
        Generate and save chunks from full video data.
        
        Args:
            video_path: Path to the video file
            video_data: Dictionary containing full video tensors
            metadata: Video metadata including total_frames and fps
            
        Returns:
            Number of chunks created
        """
        total_frames = metadata['total_frames']
        
        # Calculate chunk boundaries with overlap
        effective_chunk_size = self.chunk_size - self.overlap_size
        num_chunks = max(1, (total_frames - self.overlap_size + effective_chunk_size - 1) // effective_chunk_size)
        
        logger.info(f"Generating {num_chunks} chunks for {total_frames} frames")
        
        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            if chunk_idx == 0:
                start_frame = 0
            else:
                start_frame = chunk_idx * effective_chunk_size
            
            end_frame = min(start_frame + self.chunk_size, total_frames)
            
            # Extract chunk data
            chunk_data = {}
            for key, tensor in video_data.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                    # Handle different tensor dimensions
                    if tensor.shape[0] == total_frames:
                        chunk_data[key] = tensor[start_frame:end_frame]
                    elif tensor.shape[0] > total_frames:
                        # Handle padded data
                        chunk_data[key] = tensor[start_frame:end_frame]
            
            # Save chunk
            self.save_chunk(video_path, chunk_idx, chunk_data, start_frame, end_frame)
        
        # Save metadata
        self.save_metadata(
            video_path,
            total_frames,
            metadata['fps'],
            num_chunks,
            metadata.get('window_params', {})
        )
        
        return num_chunks
    
    def generate_windows(
        self,
        video_path: str,
        window_size: int,
        stride: int,
        context_size: int = 10
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate windows from chunked video data.
        
        Args:
            video_path: Path to the video file
            window_size: Size of each window in frames
            stride: Stride between windows
            context_size: Additional context frames
            
        Returns:
            List of window dictionaries
        """
        metadata = self.load_metadata(video_path)
        if metadata is None:
            logger.error(f"No metadata found for {video_path}")
            return []
        
        total_frames = metadata['total_frames']
        num_chunks = metadata['num_chunks']
        
        # Calculate window positions
        n_windows = max(0, (total_frames - window_size) // stride + 1)
        
        if n_windows == 0:
            logger.warning(f"Video too short for window_size={window_size}: {total_frames} frames")
            return []
        
        logger.info(f"Generating {n_windows} windows with size={window_size}, stride={stride}")
        
        windows = []
        for window_idx in range(n_windows):
            start_frame = window_idx * stride
            end_frame = min(start_frame + window_size, total_frames)
            
            # Determine which chunks are needed
            chunk_start_idx = max(0, (start_frame - self.overlap_size) // (self.chunk_size - self.overlap_size))
            chunk_end_idx = min(num_chunks - 1, end_frame // (self.chunk_size - self.overlap_size))
            
            # Collect data from relevant chunks
            window_data = {}
            
            for chunk_idx in range(chunk_start_idx, chunk_end_idx + 1):
                chunk = self._load_chunk(video_path, chunk_idx)
                if chunk is None:
                    continue
                
                chunk_start = chunk['metadata']['start_frame']
                chunk_end = chunk['metadata']['end_frame']
                
                # Calculate slice within chunk
                local_start = max(0, start_frame - chunk_start)
                local_end = min(chunk_end - chunk_start, end_frame - chunk_start)
                
                if local_end <= local_start:
                    continue
                
                # Extract and concatenate data
                for key, tensor in chunk.items():
                    if key == 'metadata':
                        continue
                    
                    if key not in window_data:
                        window_data[key] = []
                    
                    # Slice the chunk data
                    slice_data = tensor[local_start:local_end]
                    window_data[key].append(slice_data)
            
            # Concatenate chunks
            for key in list(window_data.keys()):
                if window_data[key]:
                    concatenated = torch.cat(window_data[key], dim=0)
                    
                    # Ensure correct window size (pad if necessary)
                    if concatenated.shape[0] < window_size:
                        pad_size = window_size - concatenated.shape[0]
                        padding_shape = (pad_size,) + concatenated.shape[1:]
                        padding = torch.zeros(padding_shape, dtype=concatenated.dtype)
                        concatenated = torch.cat([concatenated, padding], dim=0)
                    elif concatenated.shape[0] > window_size:
                        concatenated = concatenated[:window_size]
                    
                    window_data[key] = concatenated
            
            # Add metadata
            window_data['metadata'] = {
                'video_path': video_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'window_idx': window_idx,
                'total_windows': n_windows,
                'fps': metadata['fps'],
                'has_context': window_idx > 0
            }
            
            windows.append(window_data)
        
        return windows
    
    def load_windows(self, video_path: str) -> List[Dict[str, torch.Tensor]]:
        """Load all windows for a video from cache."""
        cache_path = self._get_cache_path(video_path, 0)  # Use chunk 0 for now
        
        if not cache_path.exists():
            logger.debug(f"No cache found for {video_path}")
            return []
        
        try:
            windows_data = []
            with h5py.File(cache_path, 'r') as f:
                # Check metadata
                if 'video_path' not in f.attrs:
                    logger.warning(f"Invalid cache file: {cache_path}")
                    return []
                
                num_windows = f.attrs.get('num_windows', 0)
                logger.info(f"Loading {num_windows} windows from cache for {video_path}")
                
                # Load each window
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        logger.warning(f"Missing window {i} in cache")
                        continue
                    
                    window_group = f[window_key]
                    window_data = {}
                    
                    # Load tensors
                    for key in window_group.keys():
                        if key == 'metadata':
                            # Load metadata separately
                            metadata = {}
                            meta_group = window_group['metadata']
                            for k in meta_group.attrs:
                                metadata[k] = meta_group.attrs[k]
                            window_data['metadata'] = metadata
                        else:
                            # Load tensor data
                            dataset = window_group[key]
                            data = dataset[()]
                            tensor = torch.from_numpy(data)
                            
                            # Restore dtype if saved
                            if 'dtype' in dataset.attrs:
                                dtype_str = dataset.attrs['dtype']
                                if 'float32' in dtype_str:
                                    tensor = tensor.float()
                                elif 'float16' in dtype_str:
                                    tensor = tensor.half()
                            
                            window_data[key] = tensor
                    
                    windows_data.append(window_data)
            
            logger.info(f"Successfully loaded {len(windows_data)} windows from cache")
            return windows_data
            
        except Exception as e:
            logger.error(f"Error loading windows from cache: {str(e)}")
            return []
    
    def save_windows(self, video_path: str, windows_data: List[Dict[str, torch.Tensor]]):
        """Save window data with proper metadata handling."""
        cache_path = self._get_cache_path(video_path, 0)  # Use chunk 0 for now
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            with h5py.File(temp_path, 'w') as f:
                # Save metadata
                f.attrs['video_path'] = video_path
                f.attrs['num_windows'] = len(windows_data)
                
                # Save each window
                for i, window in enumerate(windows_data):
                    window_group = f.create_group(f'window_{i}')
                    
                    # Save tensors
                    for key, tensor in window.items():
                        if key == 'metadata':
                            continue
                            
                        if isinstance(tensor, torch.Tensor):
                            # Convert to numpy and save
                            data = tensor.cpu().numpy()
                            ds = window_group.create_dataset(
                                key,
                                data=data,
                                compression='gzip'
                            )
                            # Save tensor metadata as string
                            ds.attrs['dtype'] = str(tensor.dtype)
                            ds.attrs['shape'] = tensor.shape
                    
                    # Save metadata dict if present
                    if 'metadata' in window:
                        meta_group = window_group.create_group('metadata')
                        for k, v in window['metadata'].items():
                            # Convert any non-string values to strings
                            if not isinstance(v, (str, bytes)):
                                v = str(v)
                            meta_group.attrs[k] = v

            # Only after successful save, replace old cache
            if temp_path.exists():
                if cache_path.exists():
                    cache_path.unlink()
                temp_path.rename(cache_path)
                logger.info(f"Successfully saved cache to {cache_path}")

        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def clear_cache(self, video_path: Optional[str] = None):
        """
        Clear cache for a specific video or all videos.
        
        Args:
            video_path: If provided, clear cache for this video only
        """
        if video_path:
            # Clear specific video
            video_hash = self._get_video_hash(video_path)
            pattern = f"{video_hash}_*.h5"
            
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Deleted cache file: {cache_file}")
            
            # Clear from memory cache
            keys_to_remove = [k for k in self.chunk_cache.keys() if k[0] == video_path]
            for key in keys_to_remove:
                del self.chunk_cache[key]
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.h5"):
                cache_file.unlink()
            
            self.chunk_cache.clear()
            self.cache_access_order.clear()
            logger.info("Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        total_cache_files = len(list(self.cache_dir.glob("*.h5")))
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.h5")) / (1024 * 1024)
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': total_cache_files,
            'total_size_mb': total_size_mb,
            'memory_cached_chunks': len(self.chunk_cache),
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size
        }