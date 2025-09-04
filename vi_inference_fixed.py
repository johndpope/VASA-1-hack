#!/usr/bin/env python3
"""
Fixed VI Inference with Multi-step Denoising
=============================================
Properly implements DDIM sampling with configurable inference steps.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDIMScheduler:
    """DDIM scheduler for multi-step denoising"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule
            s = 0.008
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        # Create inference timesteps
        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference"""
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)  # Reverse to go from T to 0
        
        self.timesteps = timesteps
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples for a given timestep"""
        
        # Ensure timesteps are on CPU for indexing
        if timesteps.is_cuda:
            timesteps_cpu = timesteps.cpu()
        else:
            timesteps_cpu = timesteps
            
        sqrt_alpha_prod = self.alphas_cumprod[timesteps_cpu] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps_cpu]) ** 0.5
        
        # Move to same device as samples
        sqrt_alpha_prod = sqrt_alpha_prod.to(original_samples.device)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(original_samples.device)
        
        # Expand dimensions for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Perform one DDIM step"""
        
        # Get current and previous alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        # Find previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        if prev_timestep < 0:
            alpha_prod_t_prev = torch.tensor(1.0).to(sample.device)
        else:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device)
        
        # Current and previous sqrt values (ensure on same device)
        sqrt_alpha_prod_t = (alpha_prod_t ** 0.5).to(sample.device)
        sqrt_one_minus_alpha_prod_t = ((1 - alpha_prod_t) ** 0.5).to(sample.device)
        sqrt_alpha_prod_t_prev = (alpha_prod_t_prev ** 0.5).to(sample.device)
        sqrt_one_minus_alpha_prod_t_prev = ((1 - alpha_prod_t_prev) ** 0.5).to(sample.device)
        
        # Expand dimensions for broadcasting
        while len(sqrt_alpha_prod_t.shape) < len(sample.shape):
            sqrt_alpha_prod_t = sqrt_alpha_prod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.unsqueeze(-1)
            sqrt_alpha_prod_t_prev = sqrt_alpha_prod_t_prev.unsqueeze(-1)
            sqrt_one_minus_alpha_prod_t_prev = sqrt_one_minus_alpha_prod_t_prev.unsqueeze(-1)
        
        # Predict original sample
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t
        
        # Compute variance for DDIM
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = eta * variance ** 0.5
        
        # Direction pointing to x_t
        pred_sample_direction = sqrt_one_minus_alpha_prod_t_prev * model_output
        
        # Previous sample
        prev_sample = sqrt_alpha_prod_t_prev * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0 (stochastic)
        if eta > 0 and generator is not None:
            noise = torch.randn_like(sample, generator=generator)
            prev_sample = prev_sample + std_dev * noise
        
        return prev_sample


class ImprovedVASAInference:
    """Improved inference with multi-step denoising and audio-visual TDD"""
    
    def __init__(
        self,
        model,
        config,
        num_inference_steps: int = 50,
        use_ddim: bool = True
    ):
        self.model = model
        self.config = config
        self.num_inference_steps = num_inference_steps
        self.use_ddim = use_ddim
        
        # Initialize scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion.num_steps,
            num_inference_steps=num_inference_steps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        
        logger.info(f"Initialized with {num_inference_steps} inference steps")
        logger.info(f"Using DDIM: {use_ddim}")
    
    def generate(
        self,
        conditions: Dict[str, torch.Tensor],
        batch_size: int = 1,
        sequence_length: int = 50,
        guidance_scale: float = 1.0,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate motion sequence with multi-step denoising
        
        Args:
            conditions: Conditioning signals (audio, gaze, etc.)
            batch_size: Batch size
            sequence_length: Number of frames to generate
            guidance_scale: Classifier-free guidance scale
            show_progress: Show progress bar
        
        Returns:
            Generated motion parameters
        """
        
        B, T = batch_size, sequence_length
        device = next(self.model.parameters()).device
        
        # Initialize with random noise
        motion_sample = {
            'theta': torch.randn(B, T, 3, 4, device=device),
            'scale': torch.randn(B, T, 3, device=device),
            'rotation': torch.randn(B, T, 3, device=device),
            'translation': torch.randn(B, T, 3, device=device),
            'expression_embed': torch.randn(B, T, 128, device=device)
        }
        
        # Setup timesteps
        timesteps = self.scheduler.timesteps.to(device)
        
        # Denoising loop
        iterator = tqdm(timesteps, desc="Denoising", disable=not show_progress)
        
        for t in iterator:
            # Expand timestep to batch dimension
            timestep = t.expand(B)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(
                    motion_data=motion_sample,
                    noise_level=timestep,
                    conditions=conditions
                )
            
            # Perform DDIM step for each component
            for key in motion_sample.keys():
                if key in noise_pred:
                    if self.use_ddim:
                        # DDIM update
                        motion_sample[key] = self.scheduler.step(
                            model_output=noise_pred[key],
                            timestep=t,
                            sample=motion_sample[key],
                            eta=0.0  # Deterministic
                        )
                    else:
                        # Simple denoising (for testing)
                        alpha = 1.0 - (t.float() / self.scheduler.num_train_timesteps)
                        motion_sample[key] = (
                            motion_sample[key] - (1 - alpha) * noise_pred[key]
                        ) / alpha.clamp(min=0.01)
            
            # Update progress
            if show_progress:
                iterator.set_postfix({'t': t.item()})
        
        logger.info(f"Generated sequence with shape: {motion_sample['theta'].shape}")
        
        return motion_sample
    
    def generate_with_audio_tdd(
        self,
        audio_features: torch.Tensor,
        identity_embed: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        test_lip_sync: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate with audio-visual TDD validation
        """
        
        if num_inference_steps:
            self.scheduler.set_timesteps(num_inference_steps)
        
        B, T, D = audio_features.shape
        
        # Prepare conditions
        conditions = {
            'audio_features': audio_features,
            'identity': identity_embed,
            'gaze': torch.zeros(B, T, 2).cuda(),
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.zeros(B, T, 2).cuda()
        }
        
        # Generate
        outputs = self.generate(
            conditions=conditions,
            batch_size=B,
            sequence_length=T
        )
        
        # Test lip sync if requested
        if test_lip_sync:
            from tdd_audio_visual_sync import AudioVisualTDDLoss
            
            av_tdd = AudioVisualTDDLoss()
            losses, test_info = av_tdd.compute_losses(
                outputs=outputs,
                conditions=conditions
            )
            
            logger.info(f"Lip sync test pass rate: {test_info['passed_ratio']:.1%}")
            
            for test_name, passed in test_info['test_results'].items():
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {test_name}")
        
        return outputs


def test_improved_inference():
    """Test the improved inference system"""
    import sys
    sys.path.insert(0, 'nemo')
    from vasa_model import VASAModel
    from omegaconf import OmegaConf
    import importlib
    
    print("\n" + "="*60)
    print("Testing Improved Multi-step Inference")
    print("="*60)
    
    # Load model
    config = OmegaConf.load('vasa_config_fixed.yaml')
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda().eval()
    
    model = VASAModel(config, volumetric_avatar).cuda().eval()
    checkpoint = torch.load('checkpoints/tdd_wandb/best_model.pth', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize improved inference
    inferencer = ImprovedVASAInference(
        model=model,
        config=config,
        num_inference_steps=20  # Faster for testing
    )
    
    # Test with different step counts
    print("\nTesting different inference steps:")
    print("-" * 40)
    
    audio_features = torch.randn(1, 30, 768).cuda()
    identity = torch.randn(1, 1, 128).cuda()
    
    for num_steps in [1, 5, 10, 20]:
        print(f"\nSteps: {num_steps}")
        inferencer.scheduler.set_timesteps(num_steps)
        
        outputs = inferencer.generate_with_audio_tdd(
            audio_features=audio_features,
            identity_embed=identity,
            num_inference_steps=num_steps,
            test_lip_sync=True
        )
        
        motion_mag = torch.norm(torch.diff(outputs['theta'], dim=1)).item()
        print(f"  Motion magnitude: {motion_mag:.3f}")
    
    print("\n" + "="*60)
    print("Improved inference ready!")
    print("="*60)


if __name__ == "__main__":
    test_improved_inference()