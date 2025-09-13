"""Blink condition handler for VASA model."""
import torch
from typing import Dict


class BlinkConditionHandler:
    """Handles blink state processing for facial landmarks."""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        # Typical blink parameters (in frames at 30fps)
        self.blink_freq = 0.08  # Probability of starting a blink per frame (~every 2-3 seconds)
        self.close_duration = 2  # Frames to close eye
        self.hold_duration = 1   # Frames to hold closed
        self.open_duration = 3   # Frames to open eye
        self.total_duration = self.close_duration + self.hold_duration + self.open_duration

    def generate_blink_sequence(self, sequence_length: int) -> torch.Tensor:
        """
        Generate a sequence of blink states.
        Returns: Tensor [sequence_length, 3] containing:
            - Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
            - Channel 1: Left eye openness (0-1)
            - Channel 2: Right eye openness (0-1)
        """
        states = torch.zeros(sequence_length, 3)
        states[:, 1:] = 1.0  # Start with eyes fully open
        
        current_frame = 0
        while current_frame < sequence_length:
            if torch.rand(1).item() < self.blink_freq and current_frame + self.total_duration < sequence_length:
                # Generate blink sequence
                # Closing phase
                for i in range(self.close_duration):
                    t = i / (self.close_duration - 1)
                    openness = 1 - t
                    frame = current_frame + i
                    states[frame, 0] = 1  # Closing phase
                    states[frame, 1:] = openness
                
                # Hold phase
                for i in range(self.hold_duration):
                    frame = current_frame + self.close_duration + i
                    states[frame, 0] = 2  # Closed phase
                    states[frame, 1:] = 0
                
                # Opening phase
                for i in range(self.open_duration):
                    t = i / (self.open_duration - 1)
                    openness = t
                    frame = current_frame + self.close_duration + self.hold_duration + i
                    states[frame, 0] = 3  # Opening phase
                    states[frame, 1:] = openness
                
                current_frame += self.total_duration
            else:
                states[current_frame, 0] = 0  # Open phase
                states[current_frame, 1:] = 1.0  # Fully open
                current_frame += 1
        
        return states

    def apply_blink_to_landmarks(
        self, 
        landmarks_dict: Dict[str, torch.Tensor],
        blink_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply blink states to eye landmarks.
        
        Args:
            landmarks_dict: Dictionary containing eye landmarks
            blink_states: [B, T, 3] tensor of blink states
            
        Returns:
            Updated landmarks with blink applied
        """
        out_dict = {}
        B, T = blink_states.shape[:2]
        
        # Process left eye
        if 'left_eye' in landmarks_dict:
            left_eye = landmarks_dict['left_eye']  # [B, T, N, 3]
            left_mean = left_eye.mean(dim=2, keepdim=True)  # Mean position (closed state)
            left_openness = blink_states[..., 1].unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            out_dict['left_eye'] = (
                left_eye * left_openness + 
                left_mean * (1 - left_openness)
            )
            
        # Process right eye
        if 'right_eye' in landmarks_dict:
            right_eye = landmarks_dict['right_eye']
            right_mean = right_eye.mean(dim=2, keepdim=True)
            right_openness = blink_states[..., 2].unsqueeze(-1).unsqueeze(-1)
            out_dict['right_eye'] = (
                right_eye * right_openness + 
                right_mean * (1 - right_openness)
            )
            
        # Copy other landmarks unchanged
        for k, v in landmarks_dict.items():
            if k not in out_dict:
                out_dict[k] = v
                
        return out_dict