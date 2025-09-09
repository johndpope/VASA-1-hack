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
            self.video_windows[window['video_path']].append({
                'index': idx,
                'window_idx': window['window_idx'],
                'has_context': window['has_context']
            })
        
        # Sort windows within each video by window index
        for video_path in self.video_windows:
            self.video_windows[video_path].sort(key=lambda x: x['window_idx'])
        
        # Create sequences of consecutive windows
        self.sequences = []
        for video_path, windows in self.video_windows.items():
            # Create overlapping sequences
            for start_idx in range(len(windows) - windows_per_sequence + 1):
                sequence = windows[start_idx:start_idx + windows_per_sequence]
                # Ensure we have the right number of windows
                if len(sequence) == windows_per_sequence:
                    self.sequences.append([w['index'] for w in sequence])
        
        # Create batches from sequences
        self.batches = []
        for i in range(0, len(self.sequences), 1):  # Process one sequence at a time
            batch = self.sequences[i]
            if len(batch) == windows_per_sequence or not drop_last:
                self.batches.append(batch)
        
        logger.info(f"Created {len(self.sequences)} window sequences from {len(self.video_windows)} videos")
        logger.info(f"Created {len(self.batches)} batches")
        logger.info(f"Each batch contains {windows_per_sequence} consecutive windows")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through batches of window indices."""
        # Shuffle batches if requested
        if self.shuffle:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        
        # Yield batches of window indices
        for batch in batches:
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
                    # Extract last context_size frames from previous window
                    window['prev_theta'] = prev_window['theta'][-context_size:]
                    window['prev_rotation'] = prev_window['rotation'][-context_size:]
                    window['prev_translation'] = prev_window['translation'][-context_size:]
                    window['prev_expression'] = prev_window['expression_embed'][-context_size:]
                    window['prev_audio'] = prev_window['audio_features'][-context_size:]
                else:
                    # No previous context - use zeros with correct shapes
                    window['prev_theta'] = torch.zeros(context_size, 3, 4)  # Fixed shape for theta
                    window['prev_rotation'] = torch.zeros(context_size, 3)
                    window['prev_translation'] = torch.zeros(context_size, 3)
                    window['prev_expression'] = torch.zeros(context_size, 128)  # Fixed expression dim
                    window['prev_audio'] = torch.zeros(context_size, 768)  # Fixed audio dim
                
                processed_windows.append(window)
        
        # Stack into batch
        if not processed_windows:
            return None
            
        # Create batched dictionary
        batched = {}
        keys_to_stack = [
            'frames', 'theta', 'scale', 'rotation', 'translation', 
            'expression_embed', 'audio_features', 'audio_mfcc',
            'gaze', 'emotion', 'head_distance', 'speed_bucket',
            'lips', 'right_eye', 'left_eye', 'jaw', 'nose',
            'lip_motion', 'blink_state',
            'prev_theta', 'prev_rotation', 'prev_translation', 
            'prev_expression', 'prev_audio'
        ]
        
        for key in keys_to_stack:
            if key in processed_windows[0]:
                try:
                    batched[key] = torch.stack([w[key] for w in processed_windows])
                except Exception as e:
                    logger.warning(f"Could not stack {key}: {e}")
        
        # Handle metadata separately (don't stack)
        batched['metadata'] = [w['metadata'] for w in processed_windows]
        
        return batched
    
    return collate_fn