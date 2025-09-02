from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import time
from dataclasses import dataclass
import logging
from pathlib import Path
from omegaconf import OmegaConf
import importlib
from logger import logger
import traceback
import psutil
import torch.cuda

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import math


@dataclass
class CrystalSchedulerOutput:
    """Output from Crystal scheduler step."""
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class CrystalGrowthSchedulerOrig:
    """
    A scheduler inspired by crystal formation in nature.
    
    Key concepts:
    1. Nucleation: Initial "seed points" in the expression space
    2. Growth: Structured pattern formation around seed points
    3. Annealing: Temperature-based refinement
    4. Symmetry: Maintaining balanced growth patterns
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_seed_points: int = 4,  # Number of initial crystallization points
        temperature_start: float = 2.0,  # High initial temperature
        temperature_end: float = 0.1,  # Cool temperature for final refinement
        growth_rate: float = 0.85,  # Rate of crystal growth
        symmetry_weight: float = 0.3,  # Weight for symmetry preservation
        device: str = "cuda"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_seed_points = num_seed_points
        self.device = device
        
        # Temperature schedule (exponential cooling)
        # self.temperatures = torch.exp(
        #     torch.linspace(
        #         np.log(temperature_start),
        #         np.log(temperature_end),
        #         num_train_timesteps
        #     )
        # ).to(device)
        
        self.temperatures = temperature_start * torch.cos(
            torch.linspace(0, math.pi/2, num_train_timesteps)
        ) + temperature_end
        # Growth rate schedule (increases as temperature decreases)
        self.growth_rates = growth_rate * (1 - torch.exp(-torch.linspace(0, 5, num_train_timesteps))).to(device)
        
        # Symmetry weight schedule (increases over time)
        self.symmetry_weights = symmetry_weight * torch.linspace(0.5, 1.0, num_train_timesteps).to(device)
        
        # Initialize timesteps
        self.timesteps = None
        self.step_index = None

    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps).long()
        self.step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """Scale the model input based on temperature."""
        if timestep is None:
            timestep = self.timesteps[self.step_index]
            
        temperature = self.temperatures[timestep]
        return sample / (temperature ** 0.5)

    def _get_seed_points(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate crystallization seed points."""
        # Create evenly spaced seed points along the embedding dimension
        indices = torch.linspace(0, shape[-1]-1, self.num_seed_points).long()
        seeds = torch.zeros(shape, device=self.device)
        seeds[..., indices] = 1.0
        return seeds

    def _apply_symmetry(
        self,
        sample: torch.Tensor,
        weight: float
    ) -> torch.Tensor:
        """Apply symmetry constraints to maintain balanced expressions."""
        # Reshape to 2D grid for symmetry operations (8x16 for 128-dim)
        B = sample.shape[0]
        grid = sample.view(B, 8, 16)
        
        # Apply horizontal and vertical reflection symmetry
        h_reflected = torch.flip(grid, dims=[-1])  # Horizontal reflection
        v_reflected = torch.flip(grid, dims=[-2])  # Vertical reflection
        
        # Blend original with symmetrical versions
        symmetrical = (grid + h_reflected + v_reflected) / 3
        blended = weight * symmetrical + (1 - weight) * grid
        
        return blended.view(sample.shape)

    def _crystal_growth_step(
        self,
        sample: torch.Tensor,
        noise_pred: torch.Tensor,
        temperature: float,
        growth_rate: float,
        symmetry_weight: float
    ) -> torch.Tensor:
        """Perform one step of crystal growth."""
        # Get seed points if first step
        if self.step_index == 0:
            seeds = self._get_seed_points(sample.shape)
        else:
            seeds = torch.zeros_like(sample)

        # Compute growth direction
        growth_direction = noise_pred - sample
        
        # Apply temperature-based noise
        noise = torch.randn_like(sample) * temperature
        
        # Update sample through crystal growth
        new_sample = sample + (
            growth_rate * growth_direction +  # Directed growth
            (1 - growth_rate) * noise +      # Random fluctuations
            seeds                            # Seed influence
        )
        
        # Apply symmetry constraints
        new_sample = self._apply_symmetry(new_sample, symmetry_weight)
        
        return new_sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True
    ) -> Union[CrystalSchedulerOutput, Tuple[torch.Tensor, ...]]:
        """Perform a denoising step using crystal growth dynamics."""
        if self.step_index is None:
            self.step_index = 0
            
        # Get current schedule values
        temperature = self.temperatures[timestep]
        growth_rate = self.growth_rates[timestep]
        symmetry_weight = self.symmetry_weights[timestep]
        
        # Perform crystal growth step
        prev_sample = self._crystal_growth_step(
            sample=sample,
            noise_pred=model_output,
            temperature=temperature,
            growth_rate=growth_rate,
            symmetry_weight=symmetry_weight
        )
        
        # Update counter
        self.step_index += 1
        
        if not return_dict:
            return (prev_sample,)
        
        return CrystalSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=model_output
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples."""
        # Scale noise by temperature
        temperatures = self.temperatures[timesteps].view(-1, 1, 1)
        noised_samples = original_samples + noise * temperatures.sqrt()
        return noised_samples


class CrystalGrowthScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_seed_points: int = 4,
        temperature_start: float = 2.0,
        temperature_end: float = 0.1,
        growth_rate: float = 0.85,
        symmetry_weight: float = 0.3,
        device: str = "cuda",
        # New performance-related parameters
        dtype: torch.dtype = torch.float32,
        schedule_resolution: int = None,  # If None, uses num_train_timesteps
        enable_memory_efficient: bool = True,
        symmetry_compute_interval: int = 1  # Apply symmetry every N steps
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_seed_points = num_seed_points
        self.device = device
        self.dtype = dtype
        self.enable_memory_efficient = enable_memory_efficient
        self.symmetry_compute_interval = symmetry_compute_interval
        
        # Use lower resolution for schedules if specified
        resolution = schedule_resolution or num_train_timesteps
        
        # Temperature schedule (exponential cooling)
        temp_schedule = torch.exp(
            torch.linspace(
                np.log(temperature_start),
                np.log(temperature_end),
                resolution
            )
        )
        
        # Interpolate if using different resolution
        if resolution != num_train_timesteps:
            indices = torch.linspace(0, resolution-1, num_train_timesteps)
            self.temperatures = torch.nn.functional.interpolate(
                temp_schedule.unsqueeze(0).unsqueeze(0),
                size=num_train_timesteps,
                mode='linear'
            ).squeeze()
        else:
            self.temperatures = temp_schedule
            
        self.temperatures = self.temperatures.to(device).to(dtype)
        
        # Growth rate schedule (with resolution handling)
        growth_schedule = growth_rate * (1 - torch.exp(-torch.linspace(0, 5, resolution)))
        if resolution != num_train_timesteps:
            self.growth_rates = torch.nn.functional.interpolate(
                growth_schedule.unsqueeze(0).unsqueeze(0),
                size=num_train_timesteps,
                mode='linear'
            ).squeeze()
        else:
            self.growth_rates = growth_schedule
            
        self.growth_rates = self.growth_rates.to(device).to(dtype)
        
        # Symmetry weight schedule
        self.symmetry_weights = (symmetry_weight * 
            torch.linspace(0.5, 1.0, num_train_timesteps)).to(device).to(dtype)
        
        # Initialize timesteps
        self.timesteps = None
        self.step_index = None


    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=torch.long)
        self.step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """Scale the model input based on temperature."""
        if timestep is None:
            timestep = self.timesteps[self.step_index]
            
        # Use temperature as scaling factor
        temperature = self.temperatures[timestep]
        return sample / (temperature ** 0.5)
    
    def _get_seed_points(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate crystallization seed points."""
        # Create evenly spaced seed points along the embedding dimension
        indices = torch.linspace(0, shape[-1]-1, self.num_seed_points).long()
        seeds = torch.zeros(shape, device=self.device, dtype=self.dtype)
        
        # Set seed points to 1.0 at specified indices
        for i in range(shape[0]):  # Handle batched inputs
            seeds[i, ..., indices] = 1.0
        
        return seeds
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True
    ) -> Union[CrystalSchedulerOutput, Tuple[torch.Tensor, ...]]:
        """Perform a denoising step using crystal growth dynamics."""
        if self.step_index is None:
            self.step_index = 0
            
        # Get current schedule values
        temperature = self.temperatures[timestep]
        growth_rate = self.growth_rates[timestep]
        symmetry_weight = self.symmetry_weights[timestep]
        
        # Perform crystal growth step
        prev_sample = self._crystal_growth_step(
            sample=sample,
            noise_pred=model_output,
            temperature=temperature,
            growth_rate=growth_rate,
            symmetry_weight=symmetry_weight
        )
        
        # Update counter
        self.step_index += 1
        
        if not return_dict:
            return (prev_sample,)
        
        return CrystalSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=model_output
        )

    def _apply_symmetry(
        self,
        sample: torch.Tensor,
        weight: float
    ) -> torch.Tensor:
        """Apply symmetry constraints with memory optimization."""
        if (self.symmetry_compute_interval > 1 and 
            self.step_index % self.symmetry_compute_interval != 0):
            return sample
            
        if self.enable_memory_efficient:
            # Process in chunks for memory efficiency
            chunk_size = 32  # Adjust based on available memory
            B = sample.shape[0]
            grid = sample.view(B, 8, 16)
            
            results = []
            for i in range(0, B, chunk_size):
                chunk = grid[i:i+chunk_size]
                h_reflected = torch.flip(chunk, dims=[-1])
                v_reflected = torch.flip(chunk, dims=[-2])
                
                symmetrical = (chunk + h_reflected + v_reflected) / 3
                blended = weight * symmetrical + (1 - weight) * chunk
                results.append(blended)
                
            return torch.cat(results, dim=0).view(sample.shape)
        else:
            # Original implementation
            B = sample.shape[0]
            grid = sample.view(B, 8, 16)
            h_reflected = torch.flip(grid, dims=[-1])
            v_reflected = torch.flip(grid, dims=[-2])
            symmetrical = (grid + h_reflected + v_reflected) / 3
            blended = weight * symmetrical + (1 - weight) * grid
            return blended.view(sample.shape)

    def _crystal_growth_step(
        self,
        sample: torch.Tensor,
        noise_pred: torch.Tensor,
        temperature: float,
        growth_rate: float,
        symmetry_weight: float
    ) -> torch.Tensor:
        """Perform one step of crystal growth with memory optimizations."""
        # Get seed points if first step
        if self.step_index == 0:
            seeds = self._get_seed_points(sample.shape)
        else:
            seeds = torch.zeros_like(sample, device=self.device, dtype=self.dtype)

        # Compute growth direction
        growth_direction = noise_pred - sample
        
        # Generate noise efficiently
        if self.enable_memory_efficient:
            noise = torch.randn_like(
                sample, 
                device=self.device, 
                dtype=self.dtype,
                memory_format=torch.contiguous_format
            )
        else:
            noise = torch.randn_like(sample)
            
        # Update sample through crystal growth
        new_sample = sample + (
            growth_rate * growth_direction +
            (1 - growth_rate) * noise * temperature +
            seeds
        )
        
        # Apply symmetry constraints
        new_sample = self._apply_symmetry(new_sample, symmetry_weight)
        
        return new_sample

logger.setLevel(logging.DEBUG)
@dataclass
class ExpressionSchedulerMetrics:
    """Metrics for comparing schedulers on expression generation."""
    frame_metrics: Dict[int, Dict[str, float]]  # Per-frame metrics
    avg_max_diff: float
    avg_mean_diff: float
    avg_correlation: float
    temporal_consistency: float
    inference_time: float

class VolumetricExpressionComparison:
    """Compare different schedulers on expression generation using volumetric avatar."""
    
    def __init__(
        self,
        volumetric_avatar: torch.nn.Module,
        device: str = "cuda",
        save_dir: str = "scheduler_comparisons",
        num_frames: int = 50,  # Added frame count parameter
        num_inference_steps: int = 50

    ):

        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        logger.info(f"Initializing with {num_inference_steps} inference steps")

        self.volumetric_avatar = volumetric_avatar
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transforms - same as VASAInference
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize schedulers
        self.schedulers = {
            "CrystalOrig": CrystalGrowthSchedulerOrig(
                num_train_timesteps=num_inference_steps,
                num_seed_points=4,
                temperature_start=2.0,
                temperature_end=0.1,
                growth_rate=0.85,
                symmetry_weight=0.3,
                device="cuda"
            ),
            "Crystal": CrystalGrowthScheduler(
                num_train_timesteps=num_inference_steps,
                num_seed_points=4,
                dtype=torch.float16,  # Use half precision
                schedule_resolution=25, # Lower resolution schedules
                enable_memory_efficient=True,
                symmetry_compute_interval=2,
                temperature_start=2.0,
                temperature_end=0.1,
                growth_rate=0.85,
                symmetry_weight=0.3,
                device="cuda"
            ),
            "DDIM": DDIMScheduler(
                num_train_timesteps=num_inference_steps,
                beta_start=0.00085,
                beta_end=0.012,
            ),
            "DPMSolver": DPMSolverMultistepScheduler(
                num_train_timesteps=num_inference_steps,
                beta_start=0.00085,
                beta_end=0.012,
                solver_order=1,  # Use first order solver for stability
            ),
            "Euler": EulerDiscreteScheduler(
                num_train_timesteps=num_inference_steps,
                beta_start=0.00085,
                beta_end=0.012,
            ),
            "EulerAncestral": EulerAncestralDiscreteScheduler(
                num_train_timesteps=num_inference_steps,
                beta_start=0.00085,
                beta_end=0.012,
            )
        }


    def log_tensor_info(self, name: str, tensor: torch.Tensor):
        """Helper to log tensor information."""
        if tensor is None:
            logger.debug(f"{name}: None")
            return
        logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"device={tensor.device}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")



    def _generate_expression(
        self,
        scheduler,
        ground_truth: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """Generate expression using given scheduler."""
        try:
            logger.debug("\n=== Starting Expression Generation ===")
            self.log_tensor_info("Ground truth expression", ground_truth)

            # Create dummy image for initial pose estimation
            dummy_img = torch.zeros(1, 3, 512, 512, device=self.device)
            with torch.no_grad():
                init_theta = self.volumetric_avatar.head_pose_regressor.forward(dummy_img)

            # Start from noise with expression shape
            noisy_expression = torch.randn_like(ground_truth)
            self.log_tensor_info("Initial noisy expression", noisy_expression)
            
            # Set timesteps for this generation
            scheduler.set_timesteps(num_steps)
            
            # Denoising loop
            for i, t in enumerate(scheduler.timesteps):
                logger.debug(f"\nProcessing timestep {i}/{num_steps} (t={t})")
                
                # Scale input
                model_input = scheduler.scale_model_input(noisy_expression, t)
                self.log_tensor_info("Scaled model input", model_input)

                # Create data dict for expression embedder
                data_dict = {
                    'source_img': dummy_img,
                    'source_mask': torch.ones(1, 1, 512, 512, device=self.device),
                    'source_theta': init_theta,
                    'source_pose_embed': model_input,
                    'target_img': dummy_img,
                    'target_mask': torch.ones(1, 1, 512, 512, device=self.device),
                    'target_theta': init_theta,
                    'target_pose_embed': model_input
                }

                # Get identity embedding once
                idt_embed = self.volumetric_avatar.idt_embedder_nw(data_dict['source_img'])
                data_dict['idt_embed'] = idt_embed

                logger.debug("\nData dictionary contents:")
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        self.log_tensor_info(f"  {k}", v)

                # Get refined expression embedding
                data_dict = self.volumetric_avatar.expression_embedder_nw(data_dict, True, False)
                noise_pred = data_dict['source_pose_embed']
                self.log_tensor_info("Predicted noise", noise_pred)

                # Scheduler step
                scheduler_output = scheduler.step(noise_pred, t, noisy_expression)
                noisy_expression = scheduler_output.prev_sample
                self.log_tensor_info("Updated noisy expression", noisy_expression)

                # Optional: Clear cache every few steps
                if i % 10 == 0:
                    torch.cuda.empty_cache()

            logger.debug("=== Expression Generation Complete ===")
            return noisy_expression

        except Exception as e:
            logger.error(f"Error in expression generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    def _evaluate_scheduler(
        self,
        scheduler,
        ground_truth: Dict[int, torch.Tensor],
        num_steps: int
    ) -> ExpressionSchedulerMetrics:
        """Evaluate a single scheduler."""
        frame_metrics = {}
        total_max_diff = 0
        total_mean_diff = 0
        total_correlation = 0
        
        for frame_idx, gt_expression in ground_truth.items():
            logger.debug(f"\nEvaluating frame {frame_idx}")
            
            # Generate expression with current scheduler
            generated = self._generate_expression(
                scheduler=scheduler,
                ground_truth=gt_expression,
                num_steps=num_steps
            )
            
            # Compute frame metrics
            metrics = self._compute_frame_metrics(
                generated=generated,
                ground_truth=gt_expression
            )
            frame_metrics[frame_idx] = metrics
            
            # Accumulate metrics
            total_max_diff += metrics['max_diff']
            total_mean_diff += metrics['mean_diff']
            total_correlation += metrics['correlation']
        
        num_frames = len(ground_truth)
        temporal_consistency = self._compute_temporal_consistency(frame_metrics)
        
        return ExpressionSchedulerMetrics(
            frame_metrics=frame_metrics,
            avg_max_diff=total_max_diff / num_frames,
            avg_mean_diff=total_mean_diff / num_frames,
            avg_correlation=total_correlation / num_frames,
            temporal_consistency=temporal_consistency,
            inference_time=0.0  # Will be set later
        )

    def log_memory_usage(self, message: str = ""):
        """Log current memory usage."""
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024 / 1024  # MB
        gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        
        logger.info(f"Memory Usage {message}:")
        logger.info(f"  RAM Usage: {ram_usage:.2f} MB")
        logger.info(f"  GPU Memory Used: {gpu_usage:.2f} MB")
        logger.info(f"  GPU Memory Cached: {gpu_cached:.2f} MB")

    def extract_expressions_from_video(
        self,
        video_path: str
    ) -> Dict[int, torch.Tensor]:
        """Extract expression embeddings from sampled frames."""
        expressions = {}
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        logger.info(f"Sampling {self.num_frames} frames at indices: {frame_indices}")

        self.log_memory_usage("Before extraction")
        
        try:
            with torch.no_grad():
                current_frame = 0
                
                for frame_idx in frame_indices:
                    # Skip to desired frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Failed to read frame {frame_idx}")
                        continue
                    
                    logger.debug(f"\nProcessing frame {frame_idx}")
                    
                    # Convert frame to RGB and tensor
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
                    
                    # Get face mask
                    face_mask = self.volumetric_avatar.face_idt.forward(frame_tensor)[0]
                    face_mask = (face_mask > 0.6).float()
                    
                    # Get head pose
                    pred_theta = self.volumetric_avatar.head_pose_regressor.forward(frame_tensor)
                    
                    # Extract expression embedding
                    data_dict = {
                        'source_img': frame_tensor,
                        'source_mask': face_mask,
                        'source_theta': pred_theta,
                        'target_img': frame_tensor,
                        'target_mask': face_mask,
                        'target_theta': pred_theta
                    }
                    
                    # Get identity embedding
                    idt_embed = self.volumetric_avatar.idt_embedder_nw(frame_tensor * face_mask)
                    data_dict['idt_embed'] = idt_embed
                    
                    # Get expression embedding
                    data_dict = self.volumetric_avatar.expression_embedder_nw(data_dict, True, False)
                    expression_embed = data_dict['source_pose_embed']
                    
                    # Save expression
                    expressions[frame_idx] = expression_embed
                    
                    # Save visualization
                    self._save_expression_visualization(
                        expression_embed,
                        frame_idx,
                        'ground_truth'
                    )
                    
                    # Clear cache periodically
                    if current_frame % 5 == 0:
                        torch.cuda.empty_cache()
                        self.log_memory_usage(f"After frame {frame_idx}")
                    
                    current_frame += 1
                
            logger.info(f"Extracted expressions from {len(expressions)} frames")
            self.log_memory_usage("After extraction")
            return expressions
            
        finally:
            cap.release()


    def compare_schedulers(
        self,
        ground_truth_expressions: Dict[int, torch.Tensor],
        num_inference_steps: Optional[int] = None
    ) -> Dict[str, ExpressionSchedulerMetrics]:
        """Compare different schedulers on expression generation."""
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
            
        results = {}
        
        for name, scheduler in self.schedulers.items():
            logger.info(f"\nEvaluating {name} scheduler...")
            
            # Start evaluation
            start_time = time.time()
            metrics = self._evaluate_scheduler(
                scheduler=scheduler,
                ground_truth=ground_truth_expressions,
                num_steps=num_inference_steps
            )
            metrics.inference_time = time.time() - start_time
            
            results[name] = metrics
            
            # Create visualizations
            self._visualize_scheduler_results(
                name=name,
                metrics=metrics,
                ground_truth=ground_truth_expressions
            )
            
            logger.info(f"Completed evaluation of {name} scheduler")
            logger.info(f"Average max difference: {metrics.avg_max_diff:.4f}")
            logger.info(f"Average correlation: {metrics.avg_correlation:.4f}")
            logger.info(f"Inference time: {metrics.inference_time:.2f}s")
            
            # Clear cache between schedulers
            torch.cuda.empty_cache()
            
        return results
    def _evaluate_scheduler(
        self,
        scheduler,
        ground_truth: Dict[int, torch.Tensor],
        num_steps: int
    ) -> ExpressionSchedulerMetrics:
        """Evaluate a single scheduler."""
        frame_metrics = {}
        total_max_diff = 0
        total_mean_diff = 0
        total_correlation = 0
        
        for frame_idx, gt_expression in ground_truth.items():
            # Generate expression with current scheduler
            generated = self._generate_expression(
                scheduler=scheduler,
                ground_truth=gt_expression,
                num_steps=num_steps
            )
            
            # Save visualization for first few frames
            if frame_idx < 5:
                self._save_expression_visualization(
                    generated,
                    frame_idx,
                    scheduler.__class__.__name__
                )
            
            # Compute frame metrics
            metrics = self._compute_frame_metrics(
                generated=generated,
                ground_truth=gt_expression
            )
            frame_metrics[frame_idx] = metrics
            
            # Accumulate metrics
            total_max_diff += metrics['max_diff']
            total_mean_diff += metrics['mean_diff']
            total_correlation += metrics['correlation']
            
            if frame_idx % 100 == 0:
                logger.debug(f"Processed frame {frame_idx}")
        
        num_frames = len(ground_truth)
        temporal_consistency = self._compute_temporal_consistency(frame_metrics)
        
        return ExpressionSchedulerMetrics(
            frame_metrics=frame_metrics,
            avg_max_diff=total_max_diff / num_frames,
            avg_mean_diff=total_mean_diff / num_frames,
            avg_correlation=total_correlation / num_frames,
            temporal_consistency=temporal_consistency,
            inference_time=0.0  # Will be set later
        )


    
    def _save_expression_visualization(
        self,
        expression: torch.Tensor,
        frame_idx: int,
        name: str
    ):
        """Create visualization of expression embedding."""
        try:
            # Ensure save directory exists
            save_dir = self.save_dir / 'expression_visualizations'
            save_dir.mkdir(exist_ok=True)
            
            # Detach and convert to numpy
            expr_np = expression.detach().cpu().numpy()
            
            # Log shape information
            logger.debug(f"Expression shape before reshape: {expr_np.shape}")
            
            # Reshape for visualization (16 x 8 for 128-dim vector)
            expr_2d = expr_np.reshape(16, -1)
            logger.debug(f"Expression shape after reshape: {expr_2d.shape}")
            
            # Create figure
            plt.figure(figsize=(10, 5))
            im = plt.imshow(expr_2d, cmap='RdBu', aspect='auto')
            plt.colorbar(im, label='Expression Value')
            plt.title(f'{name} Expression (Frame {frame_idx})')
            plt.xlabel('Dimension')
            plt.ylabel('Channel')
            
            # Save visualization
            save_path = save_dir / f'expression_{name}_frame_{frame_idx}.png'
            plt.savefig(save_path)
            plt.close()
            
            logger.debug(f"Saved expression visualization to {save_path}")
            
            # Save raw tensor for further analysis
            torch.save(expression.detach().cpu(), save_dir / f'expression_{name}_frame_{frame_idx}.pt')
            
        except Exception as e:
            logger.error(f"Error saving expression visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def _compute_frame_metrics(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics between generated and ground truth expressions."""
        # Detach tensors before computation
        gen_np = generated.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        # Compute differences
        diff = np.abs(gen_np - gt_np)
        
        return {
            'max_diff': np.max(diff),
            'mean_diff': np.mean(diff),
            'correlation': np.corrcoef(gen_np.flatten(), gt_np.flatten())[0, 1]
        }

    def _compute_temporal_consistency(
        self,
        frame_metrics: Dict[int, Dict[str, float]]
    ) -> float:
        """Compute temporal consistency score."""
        diffs = []
        frames = sorted(frame_metrics.keys())
        
        for i in range(len(frames) - 1):
            curr_metrics = frame_metrics[frames[i]]
            next_metrics = frame_metrics[frames[i + 1]]
            
            diff = abs(curr_metrics['mean_diff'] - next_metrics['mean_diff'])
            diffs.append(diff)
            
        return -np.mean(diffs)  # Negative so lower is worse

    def visualize_metrics_comparison(
        self,
        results: Dict[str, ExpressionSchedulerMetrics]
    ) -> None:
        """Create comparison visualization of metrics across schedulers."""
        metrics = ['avg_max_diff', 'avg_mean_diff', 'avg_correlation', 
                  'temporal_consistency', 'inference_time']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        fig.suptitle('Scheduler Expression Generation Comparison')
        
        for i, metric in enumerate(metrics):
            values = [getattr(m, metric) for m in results.values()]
            axes[i].bar(results.keys(), values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_comparison.png')
        plt.close()
        
    def _visualize_scheduler_results(
        self,
        name: str,
        metrics: ExpressionSchedulerMetrics,
        ground_truth: Dict[int, torch.Tensor]
    ):
        """Create visualizations for scheduler results."""
        # Create output directory
        out_dir = self.save_dir / name
        out_dir.mkdir(exist_ok=True)
        
        # Plot metrics over time
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        frames = sorted(metrics.frame_metrics.keys())
        
        # Max difference
        max_diffs = [metrics.frame_metrics[f]['max_diff'] for f in frames]
        axes[0].plot(frames, max_diffs)
        axes[0].set_title(f'{name} - Max Difference Over Time')
        axes[0].set_ylabel('Max Difference')
        
        # Mean difference
        mean_diffs = [metrics.frame_metrics[f]['mean_diff'] for f in frames]
        axes[1].plot(frames, mean_diffs)
        axes[1].set_title(f'{name} - Mean Difference Over Time')
        axes[1].set_ylabel('Mean Difference')
        
        # Correlation
        correlations = [metrics.frame_metrics[f]['correlation'] for f in frames]
        axes[2].plot(frames, correlations)
        axes[2].set_title(f'{name} - Correlation Over Time')
        axes[2].set_ylabel('Correlation')
        
        plt.tight_layout()
        plt.savefig(out_dir / 'metrics_over_time.png')
        plt.close()
        
        # Save summary metrics
        with open(out_dir / 'summary.txt', 'w') as f:
            f.write(f"Scheduler: {name}\n")
            f.write(f"Average Max Difference: {metrics.avg_max_diff:.4f}\n")
            f.write(f"Average Mean Difference: {metrics.avg_mean_diff:.4f}\n")
            f.write(f"Average Correlation: {metrics.avg_correlation:.4f}\n")
            f.write(f"Temporal Consistency: {metrics.temporal_consistency:.4f}\n")
            f.write(f"Inference Time: {metrics.inference_time:.2f}s\n")




if __name__ == "__main__":
    # Load EMO model first
    logger.info("Loading EMO model...")
    
    # Load config to get paths
    from omegaconf import OmegaConf
    config = OmegaConf.load('vasa_config.yaml')
    
    model_path = config.paths.volumetric_model
    emo_config = OmegaConf.load(config.paths.volumetric_config)
    
    import sys
    sys.path.insert(0, 'nemo')
    volumetric_avatar = importlib.import_module(f'models.stage_1.volumetric_avatar.va').Model(emo_config, training=False)
    
    # Load EMO model weights
    model_dict = torch.load(model_path, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    
    
    comparison = VolumetricExpressionComparison(
        volumetric_avatar=volumetric_avatar,
        save_dir="expression_scheduler_comparison"
    )
    
    ground_truth = comparison.extract_expressions_from_video(f"{config.paths.video_folder}/ovs-GiY_848_1.mp4")
    
    
    results = comparison.compare_schedulers(
        ground_truth_expressions=ground_truth,
        num_inference_steps=50
    )