"""
Wav2Vec2 module aligned with JoyVASA implementation.
Handles linear interpolation for frame rate alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Optional


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Linear interpolation layer for resampling features.
    
    Args:
        features: (N, C, L) feature tensor
        input_fps: Input frame rate (50 for wav2vec)
        output_fps: Target output frame rate
        output_len: Optional fixed output length
    
    Returns:
        Interpolated features (N, C, output_len)
    """
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=False, mode='linear')
    return output_features


class AlignedWav2Vec2Model(nn.Module):
    """
    Wav2Vec2 model with frame rate alignment via linear interpolation.
    Aligned with JoyVASA's implementation.
    """
    
    def __init__(self, model_name='facebook/wav2vec2-base', freeze_feature_extractor=True):
        super().__init__()
        
        # Load pretrained wav2vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze feature extractor if requested (like JoyVASA)
        if freeze_feature_extractor:
            self.wav2vec.feature_extractor._freeze_parameters()
            
        # Wav2vec outputs at 50 fps
        self.input_fps = 50
        
    def forward(
        self, 
        input_values: torch.Tensor,
        output_fps: int = 25,
        frame_num: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_back_resample: bool = True
    ):
        """
        Forward pass with automatic frame rate alignment.
        
        Args:
            input_values: Raw audio waveform tensor
            output_fps: Target output frame rate (default 25)
            frame_num: Target number of frames
            attention_mask: Optional attention mask
            use_back_resample: If True, use JoyVASA's BackResample strategy
        
        Returns:
            Audio features aligned to target frame rate (N, L, 768)
        """
        
        if use_back_resample and frame_num is not None:
            # JoyVASA's BackResample strategy: 
            # Extract at 2x frame rate, then downsample
            # This helps preserve more temporal information
            
            # First extract features
            outputs = self.wav2vec(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.last_hidden_state  # (N, L, 768)
            
            # Transpose for interpolation (N, 768, L)
            hidden_states = hidden_states.transpose(1, 2)
            
            # Calculate proper length for 2x oversampling
            target_len_2x = frame_num * 2
            
            # First interpolate to 2x target length
            hidden_states = linear_interpolation(
                hidden_states, 
                self.input_fps, 
                output_fps * 2,
                output_len=target_len_2x
            )
            
            # Then downsample to target length
            hidden_states = F.interpolate(
                hidden_states, 
                size=frame_num, 
                align_corners=False, 
                mode='linear'
            )
            
            # Transpose back (N, L, 768)
            hidden_states = hidden_states.transpose(1, 2)
            
        else:
            # Standard forward pass
            outputs = self.wav2vec(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.last_hidden_state  # (N, L, 768)
            
            if frame_num is not None:
                # Transpose for interpolation (N, 768, L)
                hidden_states = hidden_states.transpose(1, 2)
                
                # Interpolate to target frame count
                hidden_states = linear_interpolation(
                    hidden_states,
                    self.input_fps,
                    output_fps,
                    output_len=frame_num
                )
                
                # Transpose back (N, L, 768)
                hidden_states = hidden_states.transpose(1, 2)
        
        return hidden_states


class Wav2VecAudioEncoder(nn.Module):
    """
    Complete audio encoder with wav2vec and projection layer.
    Matches JoyVASA's audio processing pipeline.
    """
    
    def __init__(
        self, 
        model_name='facebook/wav2vec2-base',
        feature_dim=512,
        output_fps=25,
        freeze_encoder=True
    ):
        super().__init__()
        
        # Wav2vec model with alignment
        self.wav2vec = AlignedWav2Vec2Model(model_name, freeze_encoder)
        
        # Projection to target feature dimension (like JoyVASA)
        self.audio_feature_map = nn.Linear(768, feature_dim)
        
        self.output_fps = output_fps
        
        # Processor for audio preprocessing
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
    def process_audio(self, audio, sample_rate=16000):
        """
        Preprocess raw audio for wav2vec.
        
        Args:
            audio: Raw audio waveform (numpy array or tensor)
            sample_rate: Audio sample rate
            
        Returns:
            Processed audio tensor ready for wav2vec
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values
    
    def forward(
        self,
        audio: torch.Tensor,
        frame_num: Optional[int] = None,
        use_back_resample: bool = True
    ):
        """
        Extract audio features with projection.
        
        Args:
            audio: Preprocessed audio tensor from process_audio()
            frame_num: Target number of frames
            use_back_resample: Use JoyVASA's 2x oversampling strategy
            
        Returns:
            Audio features (N, L, feature_dim)
        """
        # Extract wav2vec features with alignment
        hidden_states = self.wav2vec(
            audio,
            output_fps=self.output_fps,
            frame_num=frame_num,
            use_back_resample=use_back_resample
        )
        
        # Project to target dimension
        audio_features = self.audio_feature_map(hidden_states)
        
        return audio_features


def pad_audio(audio, target_length=None):
    """
    Pad audio to target length if needed.
    """
    if target_length is None:
        return audio
        
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio[..., :target_length]
    
    pad_length = target_length - current_length
    padding = (0, pad_length)
    return F.pad(audio, padding, mode='constant', value=0)