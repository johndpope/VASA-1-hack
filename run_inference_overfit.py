#!/usr/bin/env python3
"""
Inference script for testing overfitted VASA model.
"""

import torch
from pathlib import Path
from omegaconf import OmegaConf
import sys
import logging
import numpy as np
import cv2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run inference on the best checkpoint."""
    
    # Load configuration
    config_path = 'overfit_config.yaml'
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Import after config to ensure paths are set
    sys.path.append('nemo')
    sys.path.append('.')
    
    from vasa_model import VASAModel
    from vasa_dataset import VASAIntegratedDataset
    from vasa_sampler import WindowSequenceSampler, create_window_sequence_collate_fn
    from torch.utils.data import DataLoader
    import importlib
    
    # Initialize volumetric avatar
    logger.info("Loading volumetric avatar...")
    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)
    
    # Load EMO weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        volumetric_avatar.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded volumetric avatar weights from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load volumetric avatar weights: {e}")
    
    volumetric_avatar.eval()
    logger.info("Volumetric avatar loaded successfully")
    
    # Initialize VASA model
    logger.info("Initializing VASA model...")
    model = VASAModel(config, volumetric_avatar)
    model.eval()
    
    # Load checkpoint
    checkpoint_path = "checkpoints_overfit/best_checkpoint.pt"
    if not Path(checkpoint_path).exists():
        # Try latest checkpoint
        checkpoint_path = "checkpoints_overfit/checkpoint_epoch_0999.pt"
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(config.device)
    
    # Create dataset for inference
    logger.info("Creating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder='junk',  # Use the test videos directory
        emo_model=volumetric_avatar,
        window_size=config.motion.window_size,
        stride=config.motion.stride,
        context_size=config.motion.context_size,
        max_videos=1,  # Just test on one video
        frame_size=(512, 512),
        sequence_length=config.motion.window_size,
        cache_audio=True,
        preextract_audio=True,
        random_seed=42,
        cache_dir='cache_overfit',
        device=config.device
    )
    
    # Create sampler and dataloader
    sampler = WindowSequenceSampler(
        dataset,
        batch_size=1,
        windows_per_sequence=1,
        shuffle=False
    )
    
    collate_fn = create_window_sequence_collate_fn(
        context_size=config.motion.context_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Run inference
    logger.info("Running inference...")
    output_dir = Path("inference_output")
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx >= 5:  # Just process first 5 batches
                break
                
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)
            
            # Forward pass through model
            logger.info(f"Processing batch {batch_idx}")
            
            # Extract motion and conditions
            motion_data = {
                'theta': batch['theta'],
                'rotation': batch['rotation'],
                'translation': batch['translation'],
                'expression_embed': batch['expression_embed']
            }
            
            conditions = {
                'audio_features': batch['audio_features'],
                'gaze': batch['gaze'],
                'head_distance': batch['head_distance'],
                'emotion': batch['emotion']
            }
            
            prev_context = {
                'theta': batch.get('prev_theta'),
                'rotation': batch.get('prev_rotation'),
                'translation': batch.get('prev_translation'),
                'expression_embed': batch.get('prev_expression')
            } if 'prev_theta' in batch else None
            
            # Generate frames
            try:
                # Use the volumetric avatar bridge to generate frames
                from vasa_va_bridge_v2 import VASAVolumetricAvatarBridgeV2
                
                bridge = VASAVolumetricAvatarBridgeV2(
                    volumetric_avatar=volumetric_avatar,
                    config=config,
                    device=config.device
                )
                
                # Use identity image
                identity_img = batch.get('identity_image', batch['frames'][:, 0])
                
                # Generate frames with the bridge
                generated_frames = bridge.generate_frames_batch(
                    motion_outputs=motion_data,
                    source_img=identity_img,
                    batch_idx=batch_idx
                )
                
                # Save generated frames
                for frame_idx, frame in enumerate(generated_frames):
                    frame_path = output_dir / f"batch_{batch_idx:03d}_frame_{frame_idx:03d}.png"
                    
                    # Convert tensor to image
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                        if frame.shape[0] == 3:  # CHW to HWC
                            frame = np.transpose(frame, (1, 2, 0))
                        frame = ((frame + 1) * 127.5).astype(np.uint8)
                    
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                logger.info(f"Saved {len(generated_frames)} frames for batch {batch_idx}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    logger.info(f"Inference complete. Output saved to {output_dir}")

if __name__ == "__main__":
    main()