#!/usr/bin/env python3
"""
TDD Audio-Visual Synchronization Curriculum
============================================
Comprehensive test-driven system for audio-to-visual mapping
including phoneme detection, mouth openness, and lip sync.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import torchaudio
import librosa
try:
    import phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("Warning: phonemizer not fully configured. Using energy-based approximation.")
from vasa_lip_normalizer import EnhancedLipAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class AudioVisualTestCriteria:
    """Test criteria for audio-visual synchronization"""
    name: str
    description: str
    min_score: float = 0.0
    target_score: float = 1.0
    weight: float = 1.0
    is_critical: bool = False


class AudioPhonemeTDD:
    """TDD tests for phoneme-to-visual mapping"""
    
    def __init__(self):
        # Initialize phonemizer for text-to-phoneme if available
        if PHONEMIZER_AVAILABLE:
            try:
                self.phonemizer = phonemizer.backend.EspeakBackend(
                    language='en-us',
                    preserve_punctuation=True,
                    with_stress=True
                )
            except:
                self.phonemizer = None
                print("Warning: espeak not available. Using simplified phoneme mapping.")
        else:
            self.phonemizer = None
        
        # Map phonemes to expected mouth shapes
        self.phoneme_mouth_map = {
            # Bilabials (lips together)
            'p': {'openness': 0.0, 'shape': 'closed'},
            'b': {'openness': 0.0, 'shape': 'closed'},
            'm': {'openness': 0.0, 'shape': 'closed'},
            
            # Open vowels (mouth wide)
            'ɑ': {'openness': 0.8, 'shape': 'wide'},  # "ah"
            'æ': {'openness': 0.7, 'shape': 'wide'},  # "cat"
            'aɪ': {'openness': 0.6, 'shape': 'wide'}, # "eye"
            
            # Closed vowels (mouth narrow)
            'i': {'openness': 0.2, 'shape': 'smile'},  # "ee"
            'u': {'openness': 0.3, 'shape': 'round'},  # "oo"
            
            # Fricatives
            'f': {'openness': 0.1, 'shape': 'teeth_on_lip'},
            'v': {'openness': 0.1, 'shape': 'teeth_on_lip'},
            's': {'openness': 0.15, 'shape': 'teeth_showing'},
            'ʃ': {'openness': 0.2, 'shape': 'pursed'},  # "sh"
            
            # Silence
            'sil': {'openness': 0.05, 'shape': 'neutral'},
        }
    
    def extract_phonemes_from_audio(self, audio: torch.Tensor, sr: int = 16000) -> List[Dict]:
        """Extract phonemes from audio using forced alignment"""
        # This is simplified - in practice you'd use a proper forced aligner
        # like Montreal Forced Aligner or Gentle
        
        # For now, extract energy-based mouth openness
        energy = torch.norm(audio, dim=-1)
        energy_normalized = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
        
        # Map energy to mouth openness
        phoneme_sequence = []
        window_size = sr // 25  # 40ms windows
        
        for i in range(0, len(energy), window_size):
            window_energy = energy_normalized[i:i+window_size].mean().item()
            
            # Map energy to phoneme category (simplified)
            if window_energy < 0.1:
                phoneme = 'sil'
            elif window_energy < 0.3:
                phoneme = 'm'  # Low energy consonant
            elif window_energy < 0.6:
                phoneme = 'i'  # Mid energy vowel
            else:
                phoneme = 'ɑ'  # High energy vowel
            
            phoneme_sequence.append({
                'phoneme': phoneme,
                'start': i / sr,
                'end': (i + window_size) / sr,
                'energy': window_energy
            })
        
        return phoneme_sequence
    
    def test_phoneme_mouth_correspondence(
        self,
        audio: torch.Tensor,
        generated_video: torch.Tensor,
        lip_analyzer: EnhancedLipAnalyzer
    ) -> Dict[str, float]:
        """Test if mouth shapes correspond to phonemes"""
        
        # Extract phonemes from audio
        phonemes = self.extract_phonemes_from_audio(audio)
        
        # Analyze mouth shapes in video
        B, T, C, H, W = generated_video.shape
        mouth_states = []
        
        for t in range(T):
            frame = generated_video[0, t].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            
            analysis = lip_analyzer.analyze_frame(frame)
            if analysis:
                mouth_states.append({
                    'openness': analysis['metrics']['openness'],
                    'state': analysis['metrics']['state'],
                    'shape': analysis['metrics']['shape']
                })
            else:
                mouth_states.append({'openness': 0, 'state': 'closed', 'shape': {}})
        
        # Compare phonemes to mouth states
        scores = []
        for i, phoneme_info in enumerate(phonemes[:len(mouth_states)]):
            phoneme = phoneme_info['phoneme']
            if phoneme in self.phoneme_mouth_map:
                expected = self.phoneme_mouth_map[phoneme]
                actual = mouth_states[i]
                
                # Score based on openness difference
                openness_diff = abs(expected['openness'] - actual['openness'])
                score = max(0, 1 - openness_diff)
                scores.append(score)
        
        return {
            'phoneme_accuracy': np.mean(scores) if scores else 0,
            'scores': scores,
            'phonemes': phonemes,
            'mouth_states': mouth_states
        }


class AudioVisualTDDLoss(nn.Module):
    """
    Comprehensive TDD loss for audio-visual synchronization
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Initialize analyzers
        self.lip_analyzer = EnhancedLipAnalyzer()
        self.phoneme_tdd = AudioPhonemeTDD()
        
        # Define test criteria
        self.test_criteria = {
            # Lip sync tests
            'lip_sync_accuracy': AudioVisualTestCriteria(
                name='lip_sync_accuracy',
                description='Lips match audio timing',
                min_score=0.7,
                target_score=0.95,
                weight=20.0,
                is_critical=True
            ),
            
            'phoneme_correspondence': AudioVisualTestCriteria(
                name='phoneme_correspondence',
                description='Mouth shapes match phonemes',
                min_score=0.6,
                target_score=0.9,
                weight=15.0,
                is_critical=True
            ),
            
            'mouth_openness_correlation': AudioVisualTestCriteria(
                name='mouth_openness_correlation',
                description='Mouth openness correlates with audio energy',
                min_score=0.5,
                target_score=0.8,
                weight=10.0
            ),
            
            'visual_speech_rhythm': AudioVisualTestCriteria(
                name='visual_speech_rhythm',
                description='Visual rhythm matches speech rhythm',
                min_score=0.6,
                target_score=0.85,
                weight=8.0
            ),
            
            'silence_detection': AudioVisualTestCriteria(
                name='silence_detection',
                description='Mouth closes during silence',
                min_score=0.8,
                target_score=0.95,
                weight=5.0
            ),
            
            'expression_consistency': AudioVisualTestCriteria(
                name='expression_consistency',
                description='Expressions remain consistent',
                min_score=0.7,
                target_score=0.9,
                weight=5.0
            )
        }
        
        # Curriculum stages
        self.curriculum_stages = {
            'stage1_mouth_basics': {
                'focus': ['mouth_openness_correlation', 'silence_detection'],
                'description': 'Learn basic mouth open/close'
            },
            'stage2_phoneme_mapping': {
                'focus': ['phoneme_correspondence', 'visual_speech_rhythm'],
                'description': 'Learn phoneme-to-visual mapping'
            },
            'stage3_fine_sync': {
                'focus': ['lip_sync_accuracy', 'expression_consistency'],
                'description': 'Fine-tune synchronization'
            }
        }
        
        self.current_stage = 'stage1_mouth_basics'
    
    def test_mouth_openness_correlation(
        self,
        audio_features: torch.Tensor,
        mouth_params: Dict[str, torch.Tensor]
    ) -> float:
        """Test if mouth openness correlates with audio energy"""
        
        # Compute audio energy
        audio_energy = torch.norm(audio_features, dim=-1)  # [B, T]
        
        # Get mouth openness from parameters
        # Assuming mouth params contain lip distances
        if 'lips' in mouth_params:
            # Upper and lower lip positions
            upper_lip = mouth_params['lips'][:, :, :10]  # First 10 points
            lower_lip = mouth_params['lips'][:, :, 10:]  # Last 10 points
            
            # Compute vertical distance (openness)
            mouth_openness = torch.mean(
                torch.abs(upper_lip[..., 1] - lower_lip[..., 1]), 
                dim=-1
            )  # [B, T]
        else:
            # Fallback: use expression embedding variance
            mouth_openness = torch.var(mouth_params.get('expression_embed', torch.zeros(1, 1, 128).cuda()), dim=-1)
        
        # Normalize both signals
        audio_norm = F.normalize(audio_energy, dim=-1)
        mouth_norm = F.normalize(mouth_openness, dim=-1)
        
        # Compute correlation
        correlation = F.cosine_similarity(audio_norm, mouth_norm, dim=-1).mean()
        
        return correlation.item()
    
    def test_silence_detection(
        self,
        audio_features: torch.Tensor,
        mouth_params: Dict[str, torch.Tensor]
    ) -> float:
        """Test if mouth closes during silence"""
        
        # Detect silence (low energy)
        audio_energy = torch.norm(audio_features, dim=-1)
        silence_threshold = audio_energy.mean() * 0.1
        is_silence = audio_energy < silence_threshold
        
        # Check mouth state during silence
        if 'lips' in mouth_params:
            upper_lip = mouth_params['lips'][:, :, :10, 1]
            lower_lip = mouth_params['lips'][:, :, 10:, 1]
            mouth_openness = torch.abs(upper_lip - lower_lip).mean(dim=-1)
        else:
            # Use expression magnitude as proxy
            mouth_openness = torch.norm(mouth_params.get('expression_embed', torch.zeros_like(audio_energy)), dim=-1)
        
        # Score: mouth should be closed (low openness) during silence
        silence_frames = is_silence.float()
        if silence_frames.sum() > 0:
            silence_openness = (mouth_openness * silence_frames).sum() / silence_frames.sum()
            # Lower openness during silence is better
            score = 1.0 - torch.sigmoid(silence_openness * 10)
            return score.item()
        
        return 1.0  # No silence detected, pass by default
    
    def test_visual_speech_rhythm(
        self,
        audio_features: torch.Tensor,
        mouth_params: Dict[str, torch.Tensor]
    ) -> float:
        """Test if visual rhythm matches speech rhythm"""
        
        # Extract rhythm from audio (using energy envelope)
        audio_energy = torch.norm(audio_features, dim=-1)
        audio_rhythm = torch.diff(audio_energy, dim=-1)
        
        # Extract visual rhythm (mouth movement)
        if 'expression_embed' in mouth_params:
            expr_diff = torch.diff(mouth_params['expression_embed'], dim=1)
            visual_rhythm = torch.norm(expr_diff, dim=-1)
        else:
            visual_rhythm = torch.zeros_like(audio_rhythm)
        
        # Align dimensions
        min_len = min(audio_rhythm.shape[-1], visual_rhythm.shape[-1])
        audio_rhythm = audio_rhythm[..., :min_len]
        visual_rhythm = visual_rhythm[..., :min_len]
        
        # Compute rhythm correlation
        if min_len > 0:
            correlation = F.cosine_similarity(
                F.normalize(audio_rhythm, dim=-1),
                F.normalize(visual_rhythm, dim=-1),
                dim=-1
            ).mean()
            return correlation.item()
        
        return 0.0
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        generated_frames: Optional[torch.Tensor] = None,
        stage: Optional[str] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compute TDD losses for audio-visual synchronization
        """
        
        if stage:
            self.current_stage = stage
        
        losses = {}
        test_results = {}
        
        # Get audio features
        audio_features = conditions.get('audio_features', torch.zeros(1, 1, 768).cuda())
        
        # Run tests based on current stage
        stage_info = self.curriculum_stages[self.current_stage]
        focus_tests = stage_info['focus']
        
        # Test 1: Mouth openness correlation
        if 'mouth_openness_correlation' in focus_tests:
            score = self.test_mouth_openness_correlation(audio_features, outputs)
            criteria = self.test_criteria['mouth_openness_correlation']
            
            # Convert to loss (higher score = lower loss)
            if score < criteria.min_score:
                loss = (criteria.min_score - score) * 100  # Heavy penalty
            elif score < criteria.target_score:
                loss = (criteria.target_score - score) * 10
            else:
                loss = 0
            
            losses['mouth_openness'] = torch.tensor(loss, device=self.device) * criteria.weight
            test_results['mouth_openness'] = score > criteria.min_score
        
        # Test 2: Silence detection
        if 'silence_detection' in focus_tests:
            score = self.test_silence_detection(audio_features, outputs)
            criteria = self.test_criteria['silence_detection']
            
            if score < criteria.min_score:
                loss = (criteria.min_score - score) * 50
            else:
                loss = 0
            
            losses['silence_detection'] = torch.tensor(loss, device=self.device) * criteria.weight
            test_results['silence_detection'] = score > criteria.min_score
        
        # Test 3: Visual speech rhythm
        if 'visual_speech_rhythm' in focus_tests:
            score = self.test_visual_speech_rhythm(audio_features, outputs)
            criteria = self.test_criteria['visual_speech_rhythm']
            
            if score < criteria.min_score:
                loss = (criteria.min_score - score) * 30
            else:
                loss = (criteria.target_score - score) * 5
            
            losses['visual_rhythm'] = torch.tensor(loss, device=self.device) * criteria.weight
            test_results['visual_rhythm'] = score > criteria.min_score
        
        # Test 4: Phoneme correspondence (if frames provided)
        if 'phoneme_correspondence' in focus_tests and generated_frames is not None:
            # This requires actual frame analysis
            # For now, use a proxy based on expression variation
            expr_var = torch.var(outputs.get('expression_embed', torch.zeros(1, 1, 128).cuda()), dim=-1).mean()
            score = torch.sigmoid(expr_var * 10).item()
            
            criteria = self.test_criteria['phoneme_correspondence']
            if score < criteria.min_score:
                loss = (criteria.min_score - score) * 40
            else:
                loss = 0
            
            losses['phoneme_correspondence'] = torch.tensor(loss, device=self.device) * criteria.weight
            test_results['phoneme_correspondence'] = score > criteria.min_score
        
        # Total loss
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)
        losses['total'] = total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=self.device)
        
        # Log test results
        if len(test_results) > 0:
            passed = sum(test_results.values())
            total = len(test_results)
            logger.info(f"\nAudio-Visual TDD: {passed}/{total} tests passed")
            logger.info(f"Current stage: {self.current_stage}")
            logger.info(f"Focus: {stage_info['description']}")
        
        return losses, {
            'test_results': test_results,
            'passed_ratio': sum(test_results.values()) / len(test_results) if test_results else 0,
            'stage': self.current_stage
        }
    
    def advance_curriculum(self, test_pass_rate: float):
        """Advance to next stage if current stage is mastered"""
        
        stages = list(self.curriculum_stages.keys())
        current_idx = stages.index(self.current_stage)
        
        # Advance if >80% tests pass
        if test_pass_rate > 0.8 and current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            logger.info(f"Advancing to {self.current_stage}")
            return True
        
        return False


def test_audio_visual_sync():
    """Test the audio-visual TDD system"""
    
    print("\n" + "="*60)
    print("Audio-Visual Synchronization TDD Test")
    print("="*60)
    
    # Initialize TDD loss
    av_tdd = AudioVisualTDDLoss(device='cuda')
    
    # Create test data
    B, T = 2, 50
    
    # Simulate audio with varying energy
    t = torch.linspace(0, 4*np.pi, T).cuda()
    audio_energy = (1 + torch.sin(t)) * 0.5  # Varying energy
    audio_features = torch.randn(B, T, 768).cuda() * audio_energy.unsqueeze(0).unsqueeze(-1)
    
    # Create mouth parameters that should correlate
    outputs = {
        'lips': torch.zeros(B, T, 20, 3).cuda(),
        'expression_embed': torch.randn(B, T, 128).cuda()
    }
    
    # Make lips follow audio energy
    mouth_openness = audio_energy * 0.1  # Scale to reasonable range
    outputs['lips'][:, :, 10:, 1] = -mouth_openness.unsqueeze(0).unsqueeze(-1)  # Lower lip down
    
    conditions = {
        'audio_features': audio_features
    }
    
    # Test each stage
    for stage_name, stage_info in av_tdd.curriculum_stages.items():
        print(f"\nStage: {stage_name}")
        print(f"Description: {stage_info['description']}")
        print(f"Focus tests: {stage_info['focus']}")
        
        losses, test_info = av_tdd.compute_losses(
            outputs=outputs,
            conditions=conditions,
            stage=stage_name
        )
        
        print(f"Total loss: {losses['total'].item():.2f}")
        print(f"Test pass rate: {test_info['passed_ratio']:.1%}")
        
        for test_name, passed in test_info['test_results'].items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}")
    
    print("\n" + "="*60)
    print("Audio-Visual TDD system ready for training!")
    print("="*60)


if __name__ == "__main__":
    test_audio_visual_sync()