"""
Custom sampler for VASA dataset that maintains temporal window sequences.
Ensures that windows from the same video are grouped together for context.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Optional
import random
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class WindowSequenceSampler(Sampler):
    """
    Batch sampler that groups consecutive windows from the same video.
    Returns batches of window indices that maintain temporal relationships.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 4,  # Number of windows per batch
        windows_per_sequence: int = 4,  # Number of consecutive windows per sequence
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the sampler.
        
        Args:
            dataset: VASAIntegratedDataset instance
            batch_size: Number of windows per batch
            windows_per_sequence: Number of consecutive windows to group
            shuffle: Whether to shuffle sequence order
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.windows_per_sequence = windows_per_sequence
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group windows by video
        self.video_windows = defaultdict(list)
        for idx, window in enumerate(dataset.windows):
            # Windows can be either dict with metadata or window info dict
            if isinstance(window, dict):
                # Check if it's a window info dict (from _create_window_indices)
                if 'video_path' in window:
                    # Window info dict structure
                    video_path = window['video_path']
                    window_idx = window.get('window_idx', idx)
                    has_context = window.get('has_context', idx > 0)
                else:
                    # Window data dict with metadata
                    video_path = window.get('metadata', {}).get('video_path', f'unknown_{idx}')
                    window_idx = window.get('metadata', {}).get('window_idx', idx)
                    has_context = window.get('metadata', {}).get('has_context', idx > 0)
            else:
                # Fallback for unknown structure
                video_path = f'unknown_{idx}'
                window_idx = idx
                has_context = idx > 0

            self.video_windows[video_path].append({
                'index': idx,
                'window_idx': window_idx,
                'has_context': has_context
            })
        
        # Sort windows within each video by window index
        for video_path in self.video_windows:
            self.video_windows[video_path].sort(key=lambda x: x['window_idx'])

        # Create sequences of consecutive windows
        self.sequences = []

        for video_path, windows in self.video_windows.items():
            # Skip videos with insufficient windows
            if len(windows) < windows_per_sequence:
                logger.debug(f"Skipping video {video_path}: only {len(windows)} windows, need {windows_per_sequence}")
                continue
                
            # Create NON-overlapping sequences for more diversity
            # Use stride equal to windows_per_sequence to avoid repetition
            stride = max(1, windows_per_sequence // 2)  # 50% overlap for some temporal context
            for start_idx in range(0, len(windows) - windows_per_sequence + 1, stride):
                sequence = windows[start_idx:start_idx + windows_per_sequence]
                # Ensure we have the right number of windows
                if len(sequence) == windows_per_sequence:
                    self.sequences.append([w['index'] for w in sequence])
        
        # Create batches from sequences - mix sequences from different videos
        self.batches = []
        sequences_per_batch = max(1, batch_size // windows_per_sequence)  # How many sequences per batch

        # Shuffle sequences first for better mixing across videos
        if self.shuffle:
            random.shuffle(self.sequences)

        for i in range(0, len(self.sequences), sequences_per_batch):
            # Combine multiple sequences into one batch for more variety
            batch = []
            for j in range(sequences_per_batch):
                if i + j < len(self.sequences):
                    batch.extend(self.sequences[i + j])

            if len(batch) >= windows_per_sequence or not drop_last:
                self.batches.append(batch)
        
        # Ensure we have at least some sequences
        if len(self.sequences) == 0:
            logger.error(f"❌ No valid sequences created! Check your data and windows_per_sequence setting.")
            logger.error(f"   Available windows per video: {[(path, len(wins)) for path, wins in self.video_windows.items()]}")
            raise ValueError(f"No valid sequences could be created with windows_per_sequence={windows_per_sequence}")
        
        logger.info(f"Created {len(self.sequences)} window sequences from {len(self.video_windows)} videos")
        logger.info(f"Created {len(self.batches)} batches with {sequences_per_batch} sequences per batch")
        logger.info(f"Using stride {stride} for sequence creation (50% overlap)")
        logger.info(f"Each sequence contains {windows_per_sequence} consecutive windows")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through batches of window indices."""
        # Shuffle batches if requested
        if self.shuffle:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            batches = [self.batches[i] for i in indices]
            logger.info(f"🔀 Shuffled {len(batches)} batches for new epoch")
            # Log first few batches to debug
            for i in range(min(3, len(batches))):
                logger.debug(f"  Batch {i}: windows {batches[i]}")
        else:
            batches = self.batches
            logger.info(f"📋 Using {len(batches)} batches in sequential order")

        # Yield batches of window indices
        for i, batch in enumerate(batches):
            if i < 3 or i % 50 == 0:  # Log first few batches and periodic updates
                # Find which videos these windows come from
                video_sources = []
                for idx in batch[:4]:  # Check first 4 windows
                    for video_path, windows in self.video_windows.items():
                        if any(w['index'] == idx for w in windows):
                            video_name = video_path.split('/')[-1]
                            if video_name not in video_sources:
                                video_sources.append(video_name)
                            break
                logger.info(f"📦 Batch {i}/{len(batches)}: {len(batch)} windows from videos: {', '.join(video_sources[:2])}")
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


class VideoSequenceSampler(Sampler):
    """
    Alternative sampler that returns all windows from a video at once.
    This maintains the original behavior where __getitem__ returns video data.
    """
    
    def __init__(
        self,
        dataset,
        shuffle: bool = True
    ):
        """
        Initialize the sampler.
        
        Args:
            dataset: VASAIntegratedDataset instance
            shuffle: Whether to shuffle video order
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_videos = len(dataset.video_paths)
        
    def __iter__(self) -> Iterator[int]:
        """Iterate through video indices."""
        indices = list(range(self.num_videos))
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of videos."""
        return self.num_videos


def create_window_sequence_collate_fn(context_size: int = 10):
    """
    Create a custom collate function that adds prev_context to windows.
    
    Args:
        context_size: Number of frames to use as context
        
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(batch: List[dict]) -> dict:
        """
        Custom collate that maintains window sequences and adds prev_context.

        Args:
            batch: List of window dictionaries

        Returns:
            Batched dictionary with prev_context added
        """
        # Filter out None values from problematic videos
        batch = [b for b in batch if b is not None]

        # If all windows were None, return None
        if not batch:
            logger.warning("All windows in batch were None, skipping batch")
            return None

        # Sort batch by video path and window index to ensure correct order
        batch_sorted = sorted(batch, key=lambda x: (
            x['metadata']['video_path'],
            x['metadata']['start_frame']
        ))
        
        # Group by video
        video_groups = defaultdict(list)
        for window in batch_sorted:
            video_groups[window['metadata']['video_path']].append(window)
        
        # Process each video group to add prev_context
        processed_windows = []
        for video_path, windows in video_groups.items():
            for i, window in enumerate(windows):
                # Add prev_context from previous window if available
                if i > 0 and windows[i-1]['metadata']['has_context']:
                    prev_window = windows[i-1]
                    # Get the device of current window tensors (should be CUDA)
                    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    for key in ['theta', 'rotation', 'translation', 'expression_embed', 'audio_features']:
                        if key in window and isinstance(window[key], torch.Tensor):
                            target_device = window[key].device
                            break

                    # Extract last context_size frames from previous window and ensure same device
                    window['prev_theta'] = prev_window['theta'][-context_size:].to(target_device)
                    window['prev_rotation'] = prev_window['rotation'][-context_size:].to(target_device)
                    window['prev_translation'] = prev_window['translation'][-context_size:].to(target_device)
                    window['prev_expression'] = prev_window['expression_embed'][-context_size:].to(target_device)
                    window['prev_audio'] = prev_window['audio_features'][-context_size:].to(target_device)
                else:
                    # No previous context - use zeros with correct shapes
                    # Get device from existing tensors in the window (should be CUDA)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    for key in ['theta', 'rotation', 'translation', 'expression_embed', 'audio_features']:
                        if key in window and isinstance(window[key], torch.Tensor):
                            device = window[key].device
                            break

                    window['prev_theta'] = torch.zeros(context_size, 3, 4, device=device)  # Fixed shape for theta
                    window['prev_rotation'] = torch.zeros(context_size, 3, device=device)
                    window['prev_translation'] = torch.zeros(context_size, 3, device=device)
                    window['prev_expression'] = torch.zeros(context_size, 128, device=device)  # Fixed expression dim
                    window['prev_audio'] = torch.zeros(context_size, 768, device=device)  # Fixed audio dim
                
                processed_windows.append(window)
        
        # Stack into batch
        if not processed_windows:
            return None
            
        # Create batched dictionary
        batched = {}

        # Debug: Log what keys are available in the first window
        logger.info(f"🔍 Keys available in first processed window: {sorted([k for k in processed_windows[0].keys() if not k.startswith('metadata')])}")

        # Check how many windows have frames
        windows_with_frames = sum(1 for w in processed_windows if 'frames' in w)
        logger.info(f"🖼️  Windows with 'frames': {windows_with_frames}/{len(processed_windows)}")

        keys_to_stack = [
            'frames', 'theta', 'scale', 'rotation', 'translation',
            'expression_embed', 'audio_features', 'audio_mfcc', 'audio_mel_spec',
            # NOTE: audio_waveform excluded - too large for batching (26k samples), causes OOM
            # Synchformer uses audio_mel_spec instead
            'gaze', 'emotion', 'head_distance', 'speed_bucket',
            'lips', 'right_eye', 'left_eye', 'jaw', 'nose',
            'lip_motion', 'blink_state',
            'prev_theta', 'prev_rotation', 'prev_translation',
            'prev_expression', 'prev_audio',
            # REQUIRED warping fields for MotionTransformer
            'xy_warps', 'rigid_warps', 'uv_warps', 'source_theta_warp',
            # EMO (Volumetric Avatar) generated frames for comparison
            'emo_frames', 'emo_keyframe_indices',
            # Flow-DPO velocity fields for preference-based alignment (VideoReward)
            'velocity_gt', 'velocity_dispreferred',
            'theta_dispreferred', 'expression_dispreferred'
        ]

        for key in keys_to_stack:
            if key in processed_windows[0]:
                try:
                    # Check if ALL windows have this key before stacking
                    if not all(key in w for w in processed_windows):
                        missing_count = sum(1 for w in processed_windows if key not in w)
                        logger.warning(f"Skipping {key}: {missing_count}/{len(processed_windows)} windows missing this key")
                        continue

                    # Move all tensors to CPU before stacking to avoid device mismatch
                    tensors_to_stack = [w[key].cpu() if isinstance(w[key], torch.Tensor) and w[key].is_cuda else w[key] for w in processed_windows]
                    batched[key] = torch.stack(tensors_to_stack)
                except Exception as e:
                    logger.warning(f"Could not stack {key}: {e}")
            else:
                if key in ['audio_mel_spec', 'audio_mfcc']:  # Log missing audio keys
                    logger.warning(f"⚠️ Key '{key}' not found in processed windows")

        # Handle lip_metrics separately as it's a dictionary of tensors
        if 'lip_metrics' in processed_windows[0]:
            batched['lip_metrics'] = {}
            for metric_key in processed_windows[0]['lip_metrics'].keys():
                try:
                    batched['lip_metrics'][metric_key] = torch.stack([
                        w['lip_metrics'][metric_key] for w in processed_windows
                    ])
                except Exception as e:
                    logger.warning(f"Could not stack lip_metrics[{metric_key}]: {e}")

        # Handle metadata separately (don't stack)
        batched['metadata'] = [w['metadata'] for w in processed_windows]

        # Handle emotion_label separately (list of strings, don't stack)
        # Check if emotion_label exists in all windows, not just the first one
        if 'emotion_label' in processed_windows[0]:
            # Only include if ALL windows have it, otherwise use None
            if all('emotion_label' in w for w in processed_windows):
                batched['emotion_label'] = [w['emotion_label'] for w in processed_windows]
            else:
                # Some windows missing emotion_label, use None for all
                batched['emotion_label'] = [w.get('emotion_label', None) for w in processed_windows]

        # Log final batched keys for debugging
        logger.info(f"📦 Final batched keys: {sorted(batched.keys())}")

        return batched
    
    return collate_fn