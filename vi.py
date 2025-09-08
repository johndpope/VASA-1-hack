import torch
import torchaudio
from pathlib import Path
from omegaconf import OmegaConf
import importlib
from PIL import Image
import numpy as np
from torchvision import transforms
from vasa_model import VASAModel, MotionSequenceHandler
import cv2
import subprocess
from typing import *
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from logger import logger
from tqdm import tqdm
import torch.nn.functional as F
import traceback
from PIL import Image
from vis_helper import save_expression_embed
from torchvision.utils import save_image
import torchvision
import torch.nn as nn
from repos.MODNet.src.models.modnet import MODNet

class VASAInference:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = OmegaConf.load(config_path)
        

        # In your __init__ or setup
        self.debug_dir = Path("debug_outputs")
        self.debug_dir.mkdir(exist_ok=True)

        # Add asset extraction directory
        self.asset_dir = Path("./data")
        self.asset_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize audio processors first
        logger.info("Initializing audio processors...")
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(device).eval()
        
        # Load EMO model with proper initialization
        logger.info("Loading EMO model...")
        model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
        emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
        self.volumetric_avatar = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)
        
        # Load EMO weights with proper error handling
        try:
            model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
            self.volumetric_avatar.load_state_dict(model_dict, strict=False)
            self.volumetric_avatar = self.volumetric_avatar.cuda()
            self.volumetric_avatar.eval()
        except Exception as e:
            logger.error(f"Error loading EMO model: {str(e)}")
            raise
            


        # Initialize VASA model
        logger.info("Loading VASA model...")
        self.model = VASAModel(
            config=self.config,
            volumetric_avatar=self.volumetric_avatar,
            device=device
        ).to(device)
        
        # Load VASA checkpoint with proper handling
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Get current state dict
            model_state = self.model.state_dict()
            
            # Load only matching keys from checkpoint
            checkpoint_state = checkpoint['model_state_dict']
            matched_state_dict = {}
            
            for key in model_state.keys():
                # Skip volumetric_avatar parameters
                if key.startswith('volumetric_avatar.'):
                    matched_state_dict[key] = model_state[key]
                # Load other parameters from checkpoint if they exist
                elif key in checkpoint_state:
                    matched_state_dict[key] = checkpoint_state[key]
                else:
                    logger.warning(f"Parameter {key} not found in checkpoint, using initialization")
                    matched_state_dict[key] = model_state[key]
                    
            # Load state dict with strict=False to handle missing volumetric_avatar parameters
            self.model.load_state_dict(matched_state_dict, strict=False)
            logger.info("Successfully loaded checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        self.model.eval()
        
     
            
        # Run EMO sanity check
        logger.info("Running EMO sanity check...")
        try:
            # Load and preprocess test image
            test_img = Image.open("./data/A.png").convert('RGB')
            test_tensor = self.transform(test_img).unsqueeze(0).to(device)
            emo_check = self.sanity_check_emo_pipeline(test_tensor)
            logger.info("EMO sanity check passed!")
        except Exception as e:
            logger.error(f"EMO sanity check failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Initialize motion sequence handler
        self.motion_handler = MotionSequenceHandler(
            window_size=self.config.motion.window_size,
            stride=self.config.motion.stride,
            context_size=self.config.motion.context_size
        )
        self.window_size = self.config.motion.window_size
        self.stride = self.config.motion.stride

   
    def generate_motion_sequence(self, source_params, audio_windows, fps=25.0):
        """Generate motion sequence from source parameters and audio."""
        try:
            device = source_params['theta'].device
            logger.info("\n=== Starting Motion Generation ===")
            
            # Initialize motion from first frame
            motion_data = {
                'theta': source_params['theta'],
                'rotation': source_params['rotation'],
                'translation': source_params['translation'],
                'expression_embed': source_params['expression_embed']
            }
            
            # Keep track of generated sequence
            generated_sequence = {k: [] for k in motion_data.keys()}
            
            num_windows = len(audio_windows)
            logger.info(f"Processing {num_windows} windows")
            
            for window_idx, audio_window in enumerate(audio_windows):
                logger.info(f"Window {window_idx+1}/{num_windows}")
                
                # Prepare conditions
                conditions = {
                    'audio_features': audio_window['audio_features'],
                    'speed_bucket': torch.ones(1, self.window_size, 1, device=device) * 4
                }
                
                # Generate window sequence
                motion_sequence = self.model.generate_sequence_inference(
                    initial_pose=motion_data,
                    initial_dynamics=motion_data['expression_embed'],
                    conditions=conditions,
                    num_steps=50  
                )
                
                # For subsequent windows, only keep the non-overlapping portion
                if window_idx > 0:
                    # Skip overlapped frames
                    for k in generated_sequence:
                        start_idx = self.motion_handler.overlap_size
                        sequence = motion_sequence[k][:, start_idx:]
                        generated_sequence[k].append(sequence)
                else:
                    # Keep full first window
                    for k in generated_sequence:
                        generated_sequence[k].append(motion_sequence[k])
                
                # Update motion data for next window using last frame
                motion_data = {
                    k: v[:, -1:] for k, v in motion_sequence.items()
                }
            
            # Concatenate all sequences
            final_sequence = {
                k: torch.cat(v, dim=1) for k, v in generated_sequence.items()
            }
            
            logger.info("Motion generation complete")
            for k, v in final_sequence.items():
                logger.info(f"{k} shape: {v.shape}")
                
            return final_sequence
                
        except Exception as e:
            logger.error(f"Error in motion generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    def generate_from_video(
        self,
        input_video: str,
        output_path: str,
        fps: float = 25.0,
        neutral_expression: bool = True
    ):
        """Generate animated sequence from input video with background preservation."""
        try:
            with torch.no_grad():  # No gradients needed for inference
                # Extract source image and audio
                source_image_path, audio_path = self.extract_video_assets(
                    input_video,
                    self.asset_dir
                )
                
                # Load source image
                source_img = Image.open(source_image_path).convert('RGB')
                source_tensor = self.transform(source_img).unsqueeze(0).to(self.device)

                # Load and process audio first to determine frame count
                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Calculate exact number of frames needed
                audio_length_seconds = waveform.shape[1] / sr
                total_frames = int(audio_length_seconds * fps)
                logger.info(f"Audio length: {audio_length_seconds:.2f} seconds")
                logger.info(f"Required frames at {fps} fps: {total_frames}")
                
                # Process audio into windows
                audio_windows = self.process_audio(waveform, sr=16000, fps=fps)
                logger.info(f"Number of audio windows: {len(audio_windows)}")

                # Extract source parameters
                source_params = self.extract_emo_parameters(source_tensor)
                
                # Generate frames with exact length match
                frames = self.generate_frames_from_audio(
                    source_params=source_params,
                    audio_windows=audio_windows,
                    initial_expression=source_params['expression_embed']
                )

                # Convert frames tensor to list for individual processing
                frames = list(frames.unbind(0))

                # Extract background once
                logger.info("Extracting background...")
                modnet = MODNet(backbone_pretrained=False)
                modnet = nn.DataParallel(modnet).cuda() if torch.cuda.is_available() else modnet
                modnet.load_state_dict(torch.load('repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', map_location=self.device))
                modnet.eval()
                modnet = modnet.to(self.device)

                # Get background using first frame
                source_mask = self._get_modnet_mask(source_tensor, modnet)
                
                lama = torch.jit.load('repos/jit_lama.pt')
                lama = lama.to(self.device)

                kernel_back = np.ones((21, 21), 'uint8')
                mask = (source_mask >= 0.8).float()
                mask = mask[0].permute(1, 2, 0)
                dilate_mask = cv2.dilate(mask.cpu().numpy(), kernel_back, iterations=2)
                dilate_mask = torch.FloatTensor(dilate_mask).unsqueeze(0).unsqueeze(0).to(self.device)
                
                background = lama(source_tensor, dilate_mask)
                background_img = transforms.ToPILImage()(background[0].cpu())
                background_tensor = self.transform(background_img).to(self.device)

                # Clear unused memory
                del lama, source_mask, mask, dilate_mask, background
                torch.cuda.empty_cache()

                # Process frames one at a time
                logger.info("Compositing frames with background...")
                composited_frames = []
                num_frames = len(frames)

                for i in tqdm(range(num_frames)):
                    try:
                        # Process single frame
                        frame = frames[i].unsqueeze(0).to(self.device)
                        
                        # Get mask for current frame
                        frame_mask = self._get_modnet_mask(frame, modnet)
                        frame_mask = torch.where(frame_mask > 0.3, frame_mask, frame_mask * 0) ** 8
                        
                        # Composite frame
                        composited = frame_mask * frame + (1 - frame_mask) * background_tensor
                        composited_frames.append(composited.squeeze(0).cpu())

                        # Clear memory after each frame
                        del frame, frame_mask
                        torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {str(e)}")
                        raise

                    if i % 50 == 0:
                        logger.info(f"Processed {i}/{num_frames} frames")

                logger.info(f"Stacking {len(composited_frames)} composited frames...")
                composited_frames = torch.stack(composited_frames)
                logger.info(f"Final composited frames shape: {composited_frames.shape}")
                
                # Save video with audio
                self._save_video({'frames': composited_frames}, audio_path, output_path, fps)
                logger.info(f"Successfully generated animation: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in generate_from_video: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        

    def _get_modnet_mask(self, img: torch.Tensor, modnet: nn.Module) -> torch.Tensor:
        """Get foreground mask using MODNet."""
        # Normalize image for MODNet
        im_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        im = im_transform(img)
        
        # Get dimensions
        ref_size = 512
        im_b, im_c, im_h, im_w = im.shape
        
        # Resize if needed
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        
        # Get mask from MODNet
        _, _, matte = modnet(im.to(self.device), True)
        
        # Resize mask back to original size
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        
        return matte
        
    def _get_modnet_mask_batch(self, batch_frames: torch.Tensor, modnet: nn.Module) -> torch.Tensor:
        """Get foreground masks for a batch of frames using MODNet."""
        
        # Normalize images for MODNet
        im_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        im = im_transform(batch_frames)
        
        # Get dimensions
        ref_size = 512
        im_b, im_c, im_h, im_w = im.shape
        
        # Resize if needed
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        
        # Get masks from MODNet (in chunks if batch is too large)
        max_chunk_size = 4  # Adjust based on your GPU memory
        mattes = []
        
        for j in range(0, im_b, max_chunk_size):
            chunk = im[j:j + max_chunk_size]
            with torch.cuda.amp.autocast():
                _, _, matte = modnet(chunk, True)
            mattes.append(matte)
            
        matte = torch.cat(mattes, dim=0)
        
        # Resize masks back to original size
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        
        return matte
        
    def generate_frames_from_audio(self, source_params, audio_windows, initial_expression):
        """Generate frames using audio features to drive expressions."""
        try:
            logger.info("\n=== Starting Audio-Driven Generation ===")
            device = source_params['theta'].device
            
            # Initialize motion data with source params for first frame
            motion_data = {
                'theta': source_params['theta'],
                'scale': source_params['scale'],
                'rotation': source_params['rotation'], 
                'translation': source_params['translation'],
                'expression_embed': initial_expression
            }
            
            # Track all generated frames and previous motion parameters
            generated_frames = []
            prev_motion = {
                'expression': None,
                'scale': None,
                'theta': None,
                'rotation': None,
                'translation': None
            }
            
            # Process each audio window
            for window_idx, window_data in enumerate(audio_windows):
                logger.info(f"Processing window {window_idx}/{len(audio_windows)}")

                conditions = {
                    'audio_features': window_data['audio_features'],
                    'speed_bucket': window_data['speed_bucket']
                }

                # Generate sequence using last available motion
                # Use CFG scales from config for stronger audio conditioning
                cfg_scales = {
                    'audio': self.config.get('cfg_scale', 3.0),
                    'speed': 1.0
                }
                
                motion_sequence = self.model.generate_sequence(
                    initial_pose=motion_data,
                    initial_dynamics=motion_data['expression_embed'],
                    conditions=conditions,
                    num_steps=50,
                    eta=0.0,  # Deterministic DDIM
                    cfg_scales=cfg_scales
                )
                
                logger.info(f"Generated sequence shape: {motion_sequence['expression_embed'].shape}")

                # Generate frames for this window
                for t in range(motion_sequence['expression_embed'].size(1)):
                    # Get current motion parameters
                    curr_expression = motion_sequence['expression_embed'][:, t]
                    curr_theta = motion_sequence['theta'][:, t]
                    curr_scale = motion_sequence['scale'][:, t]
                    curr_rotation = motion_sequence['rotation'][:, t]
                    curr_translation = motion_sequence['translation'][:, t]

                    # Calculate and log differences if previous values exist
                    if prev_motion['expression'] is not None:
                        expr_diff = (curr_expression - prev_motion['expression']).abs().mean().item()
                        theta_diff = (curr_theta - prev_motion['theta']).abs().mean().item()
                        rot_diff = (curr_rotation - prev_motion['rotation']).abs().mean().item()
                        scale_diff = (curr_scale - prev_motion['scale']).abs().mean().item()
                        trans_diff = (curr_translation - prev_motion['translation']).abs().mean().item()
                        
                        logger.info(f"Frame {window_idx * 50 + t} differences:")
                        logger.info(f"  Expression diff: {expr_diff:.4f}")
                        logger.info(f"  Theta diff: {theta_diff:.4f}")
                        logger.info(f"  Rotation diff: {rot_diff:.4f}")
                        logger.info(f"  Scale diff: {scale_diff:.4f}")
                        logger.info(f"  Translation diff: {trans_diff:.4f}")

                    # Generate the frame
                    frame = self._generate_frame(source_params, curr_expression, device)
                    generated_frames.append(frame)

                    # Update previous motion parameters
                    prev_motion = {
                        'expression': curr_expression,
                        'theta': curr_theta,
                        'scale': curr_scale,
                        'rotation': curr_rotation,
                        'translation': curr_translation
                    }

                # Update motion data using the last frame from this window
                motion_data = {
                    'theta': motion_sequence['theta'][:, -1:],
                    'rotation': motion_sequence['rotation'][:, -1:],
                    'scale': motion_sequence['scale'][:, -1:],
                    'translation': motion_sequence['translation'][:, -1:],
                    'expression_embed': motion_sequence['expression_embed'][:, -1]
                }

            logger.info(f"Total frames generated: {len(generated_frames)}")
            return torch.stack(generated_frames).squeeze(1)

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def test_grid_sample(self):
        """Test EMO's grid sample function works correctly."""
        try:
            logger.info("\n=== Testing Grid Sample ===")
            device = next(self.volumetric_avatar.parameters()).device
            
            # Create test volume
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size
            
            test_volume = torch.zeros(1, c, d, s, s, device=device)
            test_volume[:, :, d//2, s//2, s//2] = 1.0
            
            # Use original grid construction from EMO model
            grid_s = torch.linspace(-1, 1, s)
            grid_z = torch.linspace(-1, 1, d)
            w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
            e = torch.ones_like(u)
            grid = torch.stack([u, v, w, e], dim=3).view(1, -1, 4).to(device)
            
            # Get identity transform
            theta = torch.eye(4, device=device).unsqueeze(0)[:, :3]
            rotation_warp = grid.bmm(theta.transpose(1, 2)).view(-1, d, s, s, 3)
            
            # Test grid sample
            output = self.volumetric_avatar.grid_sample(test_volume, rotation_warp)
            
            logger.info(f"Input volume shape: {test_volume.shape}")
            logger.info(f"Grid shape: {grid.shape}")
            logger.info(f"Rotation warp shape: {rotation_warp.shape}")
            logger.info(f"Output volume shape: {output.shape}")
            
            # Check if pattern is preserved
            in_center = test_volume[:, :, d//2, s//2, s//2]
            out_center = output[:, :, d//2, s//2, s//2]
            logger.info(f"Center value preserved: {torch.allclose(in_center, out_center)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Grid sample test failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def log_tensor_range(self,name, tensor):
        if tensor is not None:
            logger.info(f"{name} range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
   
    def sanity_check_emo_pipeline(self, source_img, frame_idx=0):
        with torch.no_grad():
            # 1. Face mask processing
            import torchvision.transforms.functional as FF
            source_gray = FF.rgb_to_grayscale(source_img)
            face_mask_source = self.volumetric_avatar.face_idt.forward(source_gray)[0]
            # face_mask_source = self.volumetric_avatar.face_idt.forward(source_img)[0]
            face_mask_source = (face_mask_source > 0.6).float()
            source_masked = source_img * face_mask_source

            # 2. Get initial parameters
            idt_embed = self.volumetric_avatar.idt_embedder_nw.forward_image(source_masked)
            theta = self.volumetric_avatar.head_pose_regressor.forward(source_img * face_mask_source)

            # 3. Create initial data dictionary
            data_dict = {
                'source_img': source_img,
                'source_mask': face_mask_source,
                'source_theta': theta,
                'target_img': source_img,
                'target_mask': face_mask_source,
                'target_theta': theta,
                'idt_embed': idt_embed
            }

            # 4. Get expression embedding through full pipeline
            data_dict = self.volumetric_avatar.expression_embedder_nw(data_dict, True, False)
            
            # 5. Get embedding dictionaries
            source_warp_embed_dict, _, _, embed_dict = self.volumetric_avatar.predict_embed(data_dict)

            # 6. Process source latents
            source_latents = self.volumetric_avatar.local_encoder_nw(source_masked)
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size

            source_latent_volume = source_latents.view(1, c, d, s, s)
            if self.volumetric_avatar.args.source_volume_num_blocks > 0:
                source_latent_volume = self.volumetric_avatar.volume_source_nw(source_latent_volume)

            canonical_volume = self.volumetric_avatar.volume_process_nw(source_latent_volume, embed_dict)
            target_latent_feats = canonical_volume.view(1, c * d, s, s)

            # 7. Generate frame
            frame, _, _, _ = self.volumetric_avatar.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            # Save images
            debug_dir = Path("debug_outputs")
            debug_dir.mkdir(exist_ok=True)
            torchvision.utils.save_image(source_img, debug_dir / "01_source.png")
            torchvision.utils.save_image(source_masked, debug_dir / "02_masked.png")
            torchvision.utils.save_image(frame, debug_dir / "03_output.png")
            comparison = torch.cat([source_img, frame], dim=3)
            torchvision.utils.save_image(comparison, debug_dir / "comparison.png")

            return frame


    def convert_theta_format(self, theta):
        if theta.ndim == 4:
            # shape [B, T, 4, 4], remove the last row and pick the first time step
            return theta[:, 0, :3, :]  # => [B, 3, 4]
        elif theta.ndim == 3:
            # shape [B, 4, 4], remove the last row
            return theta[:, :3, :]     # => [B, 3, 4]
        else:
            raise ValueError(f"Unexpected shape for theta: {theta.shape}")


    def extract_emo_parameters(self, source_img):
        """Extract EMO model parameters from source image."""
        with torch.no_grad():
            # Get face mask and process image
            source_mask = self.volumetric_avatar.face_idt.forward(source_img)[0]
            source_mask = (source_mask > 0.6).float()
            source_masked = source_img * source_mask
            
            # Get embeddings
            idt_embed = self.volumetric_avatar.idt_embedder_nw(source_masked)
            theta, scale, rotation, translation = self.volumetric_avatar.head_pose_regressor.forward(
                source_img, return_srt=True)
            expression_embed = self.volumetric_avatar.expression_embedder_nw.net_face(source_img)[0]
            
            # Get canonical volume
            source_latents = self.volumetric_avatar.local_encoder_nw(source_masked)
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size
            
            source_volume = source_latents.view(1, c, d, s, s)
            if hasattr(self.volumetric_avatar, 'volume_source_nw'):
                source_volume = self.volumetric_avatar.volume_source_nw(source_volume)
            
            canonical_volume = self.volumetric_avatar.volume_process_nw(source_volume)
            
            theta = self.convert_theta_format(theta)
            return {
                'idt_embed': idt_embed,
                'theta': theta,
                'scale': scale,
                'rotation': rotation,
                'translation': translation,
                'expression_embed': expression_embed,
                'source_mask': source_mask,
                'source_volume': source_volume,
                'canonical_volume': canonical_volume
            }
        

    def process_audio(self, waveform, sr=16000, fps=25.0):
        """Process audio into windows of features."""
        try:
            # Calculate exact audio duration and frame count
            audio_length_seconds = waveform.shape[1] / sr
            target_frames = int(audio_length_seconds * fps)
            logger.info(f"\n=== Audio Processing Stats ===")
            logger.info(f"Audio length: {audio_length_seconds:.2f} seconds")
            logger.info(f"Target frames @ {fps} fps: {target_frames}")
            
            window_duration = self.window_size / fps
            logger.info(f"Window duration: {window_duration:.2f} seconds")
            
            samples_per_window = int(window_duration * sr)
            stride_samples = int((self.stride / fps) * sr)
            logger.info(f"Samples per window: {samples_per_window}")
            logger.info(f"Stride samples: {stride_samples}")

            windows = []
            total_duration = 0
            
            for start_sample in range(0, waveform.size(1) - samples_per_window + 1, stride_samples):
                window = waveform[:, start_sample:start_sample + samples_per_window]
                
                # Process through wav2vec
                inputs = self.audio_processor(window.squeeze().numpy(), 
                                            sampling_rate=sr,
                                            return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.audio_model(**inputs.to(self.device))
                    features = outputs.last_hidden_state
                    
                    # Interpolate to match window size
                    features = F.interpolate(
                        features.transpose(1, 2),  # -> [1, D, L]
                        size=self.window_size,
                        mode='linear'
                    ).transpose(1, 2)  # -> [1, T, D]
                    
                    speed_bucket = torch.ones(1, self.window_size, 1).to(self.device) * 4

                    window_frames = self.window_size
                    total_duration += window_frames / fps
                    
                    windows.append({
                        'audio_features': features,
                        'speed_bucket': speed_bucket
                    })

                logger.info(f"Window {len(windows)}: {window_frames} frames")

            logger.info(f"\nTotal windows created: {len(windows)}")
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Target duration: {audio_length_seconds:.2f} seconds")

            return windows

        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    def _generate_frame(self, source_params, curr_expression, device):
        """Generate a single frame using EMO decoder with given expression."""
        try:
            # Get dimensions from EMO model
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size

            # Create identity grid first
            grid = self.volumetric_avatar.identity_grid_3d.repeat_interleave(1, dim=0)
            target_rotation_warp = grid.bmm(source_params['theta'][:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

            # Create source tensor with correct shape for RGB image (B, C, H, W)
            dummy_rgb = torch.zeros(1, 3, 512, 512).to(device)  # Create dummy RGB image 

            # Create complete data dict with all required fields
            data_dict = {
                'source_img': dummy_rgb,
                'target_img': dummy_rgb,
                'source_mask': source_params['source_mask'],
                'target_mask': source_params['source_mask'],
                'source_theta': source_params['theta'],
                'target_theta': source_params['theta'],
                'idt_embed': source_params['idt_embed'],
                'source_pose_embed': source_params['expression_embed'],
                'target_pose_embed': curr_expression,
                'target_delta_uv': torch.zeros(1, 3, d, s, s).to(device)
            }

            # Process through EMO pipeline
            source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = \
                self.volumetric_avatar.predict_embed(data_dict)

            # Generate UV warp
            target_uv_warp, _ = self.volumetric_avatar.uv_generator_nw(target_warp_embed_dict)
            
            # Handle resizing using avg_pool3d
            if self.volumetric_avatar.resize_warp:
                stride = self.volumetric_avatar.warp_resize_stride
                target_uv_warp = F.avg_pool3d(target_uv_warp.permute(0, 4, 1, 2, 3), 
                                            kernel_size=stride,
                                            stride=stride).permute(0, 2, 3, 4, 1)

            # Create target volume with grid sampling
            aligned_target_volume = self.volumetric_avatar.grid_sample(
                self.volumetric_avatar.grid_sample(source_params['canonical_volume'], target_uv_warp),
                target_rotation_warp
            )

            target_latent_feats = aligned_target_volume.view(1, c * d, s, s)

            # Generate frame through decoder
            frame, _, _, _ = self.volumetric_avatar.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            return frame

        except Exception as e:
            logger.error(f"Error generating frame: {str(e)}")
            logger.error(traceback.format_exc())
            logger.error(f"Available source_params keys: {list(source_params.keys())}")
            logger.error(f"Data dict keys: {list(data_dict.keys())}")
            raise
    
    def extract_video_assets(self,video_path: str, output_dir: Path) -> Tuple[str, str]:
        """
        Extract first frame and audio from video file.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted assets
            
        Returns:
            Tuple of (image_path, audio_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        image_path = output_dir / f"{video_name}_source.png"
        audio_path = output_dir / f"{video_name}_audio.wav"
        
        try:
            logger.info(f"Extracting assets from: {video_path}")
            
            # Extract first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read video frame")
                
            # Convert BGR to RGB and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(image_path)
            logger.info(f"Saved source frame to: {image_path}")
            
            # Extract audio using ffmpeg
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                str(audio_path)
            ]
            
            subprocess.run(command, check=True)
            logger.info(f"Saved audio to: {audio_path}")
            
            return str(image_path), str(audio_path)
            
        except Exception as e:
            logger.error(f"Error extracting video assets: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if 'cap' in locals():
                cap.release()

    def extract_source_params(self, source_img):
        """Extract initial parameters and canonical volume from source image."""
        with torch.no_grad():
            # Get face mask
            source_mask = self.volumetric_avatar.face_idt.forward(source_img)[0]
            source_mask = (source_mask > 0.6).float()
            source_masked = source_img * source_mask
            
            # Get identity embedding
            idt_embed = self.volumetric_avatar.idt_embedder_nw(source_masked)
            
            # Get pose parameters
            theta, scale, rotation, translation = self.volumetric_avatar.head_pose_regressor.forward(
                source_img, return_srt=True)
            
            # Get expression embedding
            expression_embed = self.volumetric_avatar.expression_embedder_nw.net_face(source_img)[0]

            # Extract source volume (following InferenceWrapper's process)
            source_latents = self.volumetric_avatar.local_encoder_nw(source_masked)
            
            # Get volume dimensions
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size
            
            # Process source volume
            source_volume = source_latents.view(1, c, d, s, s)
            source_volume = self.volumetric_avatar.volume_source_nw(source_volume)
                
            # Get canonical volume through the process_nw
            canonical_volume = self.volumetric_avatar.volume_process_nw(source_volume)
            
            return {
                'idt_embed': idt_embed,
                'theta': theta,
                'scale': scale,
                'rotation': rotation,
                'translation': translation,
                'expression_embed': expression_embed,
                'source_mask': source_mask,
                'source_volume': source_volume,
                'canonical_volume': canonical_volume
            }



    def _save_video(self, frames_dict, audio_path, output_path, fps):
        """Save video frames with audio."""
        try:
            # Save frames
            temp_video = str(Path(output_path).with_suffix('.tmp.mp4'))
            
            # Extract frames tensor from dictionary
            frames = frames_dict['frames']
            
            # Convert to CPU and numpy
            frames = frames.cpu().numpy().transpose(0, 2, 3, 1)
            
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
                
            writer = cv2.VideoWriter(
                temp_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frames.shape[2], frames.shape[1])
            )
            
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            
            # Add audio
            command = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                output_path
            ]
            
            subprocess.run(command, check=True)
            Path(temp_video).unlink()
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


    # def visualize_inference_outputs2(
    #     self,
    #     input_video: str,
    #     output_dir: str = "inference_vis",
    #     fps: float = 25.0
    # ):
    #     """
    #     Create visualizations of inference outputs for sanity checking.
    #     Saves frame-by-frame visualization of inputs and generated expressions.
        
    #     Args:
    #         input_video: Path to input video file
    #         output_dir: Directory to save visualizations
    #         fps: Video frame rate
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.gridspec import GridSpec
    #     import os
        
    #     try:
    #         os.makedirs(output_dir, exist_ok=True)
    #         logger.info(f"Saving visualizations to: {output_dir}")
            
    #         # Extract source image and audio
    #         source_image_path, audio_path = self.extract_video_assets(
    #             input_video, 
    #             Path(output_dir) / "assets"
    #         )
            
    #         # Load source image
    #         source_img = Image.open(source_image_path).convert('RGB')
    #         source_tensor = self.transform(source_img).unsqueeze(0).to(self.device)
            
    #         # Load and process audio
    #         waveform, sr = torchaudio.load(audio_path)
    #         if sr != 16000:
    #             resampler = torchaudio.transforms.Resample(sr, 16000)
    #             waveform = resampler(waveform)
            
    #         # Convert to mono if needed
    #         if waveform.shape[0] > 1:
    #             waveform = waveform.mean(dim=0, keepdim=True)
            
    #         # Process audio into windows with proper overlap
    #         audio_windows = self.process_audio(waveform, sr=16000, fps=fps)
            
    #         # Extract source parameters
    #         source_params = self.extract_source_params(source_tensor)
            
    #         # Process each window
    #         for window_idx, audio_window in enumerate(audio_windows):
    #             logger.info(f"Processing window {window_idx}")
                
    #             # Prepare conditions
    #             conditions = {
    #                 'audio_features': audio_window.unsqueeze(0),
    #                 'speed_bucket': torch.ones(
    #                     1, self.motion_handler.window_size, 1, 
    #                     device=self.device
    #                 ) * 4  # Middle speed bucket
    #             }
                
    #             # Generate motion sequence
    #             motion_sequence = self.model.generate_sequence(
    #                 initial_pose={
    #                     'theta': source_params['theta'],
    #                     'rotation': source_params['rotation'],
    #                     'translation': source_params['translation']
    #                 },
    #                 initial_dynamics=source_params['expression_embed'],
    #                 conditions=conditions,
    #                 num_steps=50
    #             )
                
    #             # Visualize each frame in the window
    #             for frame_idx in range(self.motion_handler.window_size):
    #                 fig = plt.figure(figsize=(20, 15))
    #                 gs = GridSpec(3, 3, figure=fig)
                    
    #                 # Generate current frame using EMO model
    #                 curr_frame = self.generate_frames(
    #                     source_params,
    #                     {
    #                         'theta': motion_sequence['theta'][:, frame_idx:frame_idx+1],
    #                         'expression_embed': motion_sequence['expression_embed'][:, frame_idx:frame_idx+1]
    #                     }
    #                 )
                    
    #                 # Plot generated frame
    #                 ax_source = fig.add_subplot(gs[0, 0])
    #                 frame_np = curr_frame[0].cpu().permute(1, 2, 0).numpy()
    #                 frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
    #                 ax_source.imshow(frame_np)
    #                 ax_source.set_title(f"Generated Frame {frame_idx}")
    #                 ax_source.axis('off')
                    
    #                 # 2. Plot audio features
    #                 ax_audio = fig.add_subplot(gs[0, 1:])
    #                 audio_feat = conditions['audio_features'][0, frame_idx].cpu().numpy()
    #                 im = ax_audio.imshow(audio_feat.reshape(1, -1), aspect='auto', cmap='viridis')
    #                 plt.colorbar(im, ax=ax_audio)
    #                 ax_audio.set_title(f"Audio Features (Frame {frame_idx})")
                    
    #                 # 3. Plot expression embedding
    #                 ax_expr = fig.add_subplot(gs[1, :])
    #                 expr = motion_sequence['expression_embed'][0, frame_idx].cpu().numpy()
    #                 expr_2d = expr.reshape(8, -1)  # Reshape for better visualization
    #                 im = ax_expr.imshow(expr_2d, cmap='RdBu', aspect='auto')
    #                 plt.colorbar(im, ax=ax_expr)
    #                 ax_expr.set_title(f"Generated Expression Embedding")
                    
    #                 # 4. Plot pose parameters
    #                 ax_pose = fig.add_subplot(gs[2, 0])
    #                 theta = motion_sequence['theta'][0, frame_idx].cpu().numpy()
    #                 im = ax_pose.imshow(theta, cmap='RdBu')
    #                 plt.colorbar(im, ax=ax_pose)
    #                 ax_pose.set_title("Pose Matrix")
                    
    #                 # 5. Plot rotation angles
    #                 ax_rot = fig.add_subplot(gs[2, 1])
    #                 rotation = motion_sequence['rotation'][0, frame_idx].cpu().numpy()
    #                 ax_rot.bar(['Pitch', 'Yaw', 'Roll'], rotation)
    #                 ax_rot.set_title("Rotation Angles")
                    
    #                 # 6. Plot translation
    #                 ax_trans = fig.add_subplot(gs[2, 2])
    #                 translation = motion_sequence['translation'][0, frame_idx].cpu().numpy()
    #                 ax_trans.bar(['X', 'Y', 'Z'], translation)
    #                 ax_trans.set_title("Translation")
                    
    #                 # Save frame visualization
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join(
    #                     output_dir,
    #                     f'window_{window_idx:03d}_frame_{frame_idx:03d}.png'
    #                 ))
    #                 plt.close()
                    
    #             # Save window statistics
    #             with open(os.path.join(output_dir, f'window_{window_idx:03d}_stats.txt'), 'w') as f:
    #                 f.write(f"Window {window_idx} Statistics\n")
    #                 f.write("-" * 50 + "\n")
    #                 f.write(f"Expression range: [{motion_sequence['expression_embed'].min().item():.3f}, "
    #                     f"{motion_sequence['expression_embed'].max().item():.3f}]\n")
    #                 f.write(f"Rotation range: [{motion_sequence['rotation'].min().item():.3f}, "
    #                     f"{motion_sequence['rotation'].max().item():.3f}]\n")
    #                 f.write(f"Translation range: [{motion_sequence['translation'].min().item():.3f}, "
    #                     f"{motion_sequence['translation'].max().item():.3f}]\n")
                    
    #         logger.info("Visualization complete!")
            
    #     except Exception as e:
    #         logger.error(f"Error in visualization: {str(e)}")
    #         logger.error(traceback.format_exc())
    #         raise


    
    # def visualize_inference_outputs(
    #     self,
    #     input_video: str,
    #     target_image_path: str,
    #     output_dir: str = "inference_vis",
    #     fps: float = 25.0
    # ):
    #     """
    #     Create visualizations of inference outputs, using expressions from input video 
    #     to drive a target identity image.
        
    #     Args:
    #         input_video: Path to input video file
    #         target_image_path: Path to target identity image
    #         output_dir: Directory to save visualizations
    #         fps: Video frame rate
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.gridspec import GridSpec
    #     import os
        
    #     try:
    #         os.makedirs(output_dir, exist_ok=True)
    #         logger.info(f"Saving visualizations to: {output_dir}")
            
    #         # Load target identity image
    #         target_img = Image.open(target_image_path).convert('RGB')
    #         target_tensor = self.transform(target_img).unsqueeze(0).to(self.device)
            
    #         # Extract canonical volume for target identity
    #         with torch.no_grad():
    #             # Get face mask
    #             target_mask = self.volumetric_avatar.face_idt.forward(target_tensor)[0]
    #             target_mask = (target_mask > 0.6).float()
    #             target_masked = target_tensor * target_mask
                
    #             # Get identity embedding
    #             idt_embed = self.volumetric_avatar.idt_embedder_nw(target_masked)
                
    #             # Extract source latents
    #             target_latents = self.volumetric_avatar.local_encoder_nw(target_masked)
                
    #             # Get volume dimensions
    #             c = self.volumetric_avatar.args.latent_volume_channels
    #             d = self.volumetric_avatar.args.latent_volume_depth
    #             s = self.volumetric_avatar.args.latent_volume_size
                
    #             # Process target volume
    #             target_latent_volume = target_latents.view(1, c, d, s, s)
    #             if self.volumetric_avatar.args.source_volume_num_blocks > 0:
    #                 target_latent_volume = self.volumetric_avatar.volume_source_nw(target_latent_volume)
                    
    #             # Get canonical volume
    #             canonical_volume = self.volumetric_avatar.volume_process_nw(target_latent_volume)
                
    #             target_params = {
    #                 'source_mask': target_mask,
    #                 'idt_embed': idt_embed,
    #                 'source_latent_volume': target_latent_volume,
    #                 'canonical_volume': canonical_volume
    #             }
            
    #         # Extract frames from input video
    #         cap = cv2.VideoCapture(input_video)
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         logger.info(f"Processing {total_frames} frames from input video")
            
    #         # Process each frame
    #         for frame_idx in range(total_frames):
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                    
    #             # Convert frame to RGB and proper format
    #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
    #             frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
                
    #             # Extract expression from frame
    #             with torch.no_grad():
    #                 # Get frame parameters
    #                 frame_mask = self.volumetric_avatar.face_idt.forward(frame_tensor)[0]
    #                 frame_mask = (frame_mask > 0.6).float()
    #                 frame_masked = frame_tensor * frame_mask
                    
    #                 # Get frame parameters
    #                 data_dict = {
    #                     'source_img': frame_tensor,
    #                     'source_mask': frame_mask,
    #                     'target_img': frame_tensor,
    #                     'target_mask': frame_mask
    #                 }
                    
    #                 # Get expression embedding
    #                 data_dict = self.volumetric_avatar.expression_embedder_nw(data_dict, True, False)
    #                 expression_embed = data_dict['source_pose_embed']
                    
    #                 # Get pose parameters
    #                 frame_theta = self.volumetric_avatar.head_pose_regressor.forward(frame_tensor)
                    
    #                 # Generate output using target identity
    #                 data_dict = {
    #                     'source_img': target_tensor,
    #                     'source_mask': target_params['source_mask'],
    #                     'target_pose_embed': expression_embed,
    #                     'target_theta': frame_theta
    #                 }
                    
    #                 # Generate frame using both EMO and VASA
    #                 emo_frame, _, _, _ = self.volumetric_avatar.decoder_nw(
    #                     data_dict,
    #                     None,  # embed_dict
    #                     target_params['canonical_volume'].view(1, -1, s, s),
    #                     False   # is_training
    #                 )
                    
    #                 vasa_frame = self.generate_frames(
    #                     target_params,
    #                     {
    #                         'theta': frame_theta,
    #                         'expression_embed': expression_embed
    #                     }
    #                 )
                
    #             # Create visualization
    #             fig = plt.figure(figsize=(20, 15))
    #             gs = GridSpec(3, 3, figure=fig)
                
    #             # Create grid of frames
    #             ax_frames = fig.add_subplot(gs[0, :])
                
    #             # Prepare frames for display
    #             input_np = frame_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    #             input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
                
    #             target_np = target_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    #             target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min())
                
    #             emo_np = emo_frame[0].cpu().permute(1, 2, 0).numpy()
    #             emo_np = (emo_np - emo_np.min()) / (emo_np.max() - emo_np.min())
                
    #             vasa_np = vasa_frame[0].cpu().permute(1, 2, 0).numpy()
    #             vasa_np = (vasa_np - vasa_np.min()) / (vasa_np.max() - vasa_np.min())
                
    #             # Create combined image
    #             padding = 10
    #             H, W = input_np.shape[:2]
    #             combined = np.zeros((H, W * 4 + padding * 3, 3))
                
    #             # Add frames
    #             combined[:, :W] = input_np  # Input frame
    #             combined[:, W+padding:2*W+padding] = target_np  # Target identity
    #             combined[:, 2*W+2*padding:3*W+2*padding] = emo_np  # EMO output
    #             combined[:, 3*W+3*padding:] = vasa_np  # VASA output
                
    #             ax_frames.imshow(combined)
    #             ax_frames.axvline(x=W + padding/2, color='white', linestyle='--', alpha=0.5)
    #             ax_frames.axvline(x=2*W + 3*padding/2, color='white', linestyle='--', alpha=0.5)
    #             ax_frames.axvline(x=3*W + 5*padding/2, color='white', linestyle='--', alpha=0.5)
    #             ax_frames.set_title(f"Input Frame | Target Identity | EMO Output | VASA Output (Frame {frame_idx})")
    #             ax_frames.axis('off')
                
    #             # Plot expression embedding
    #             ax_expr = fig.add_subplot(gs[1, :])
    #             expr = expression_embed[0].cpu().numpy()
    #             expr_2d = expr.reshape(8, -1)
    #             im = ax_expr.imshow(expr_2d, cmap='RdBu', aspect='auto')
    #             plt.colorbar(im, ax=ax_expr)
    #             ax_expr.set_title("Extracted Expression Embedding")
                
    #             # Plot pose parameters
    #             ax_pose = fig.add_subplot(gs[2, 0])
    #             theta = frame_theta[0].cpu().numpy()
    #             im = ax_pose.imshow(theta, cmap='RdBu')
    #             plt.colorbar(im, ax=ax_pose)
    #             ax_pose.set_title("Pose Matrix")
                
    #             # Save visualization
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    #             plt.close()
                
    #         cap.release()
    #         logger.info("Visualization complete!")
            
    #     except Exception as e:
    #         logger.error(f"Error in visualization: {str(e)}")
    #         logger.error(traceback.format_exc())
    #         raise

# Example usage:
# inferencer = VASAInference(...)
# inferencer.visualize_inference_outputs("input.mp4", "vis_output")
# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VASA-1 Video Inference')
    parser.add_argument('--config', type=str, default='vasa_config.yaml',
                        help='Path to config file (default: vasa_config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: auto-detect from config)')
    parser.add_argument('--input', type=str, default='./junk/11.mp4',
                        help='Input video path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: auto-generate)')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Output video FPS (default: 25.0)')
    parser.add_argument('--neutral', action='store_true',
                        help='Use neutral expression')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization outputs instead of video')
    parser.add_argument('--target-image', type=str, default='./data/A.png',
                        help='Target image for visualization (default: ./data/A.png)')
    parser.add_argument('--vis-dir', type=str, default='vis_output',
                        help='Output directory for visualizations (default: vis_output)')
    
    args = parser.parse_args()
    
    # Load config to determine checkpoint path
    config = OmegaConf.load(args.config)
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Auto-detect checkpoint based on config
        if 'overfit' in args.config:
            checkpoint_path = "./checkpoints_overfit/best_checkpoint.pt"
            if not Path(checkpoint_path).exists():
                # Try to find latest checkpoint
                checkpoint_dir = Path("./checkpoints_overfit")
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
                        logger.info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = "./checkpoints/best_checkpoint.pt"
            if not Path(checkpoint_path).exists():
                # Try to find latest checkpoint
                checkpoint_dir = Path("./checkpoints")
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])
                        logger.info(f"Using latest checkpoint: {checkpoint_path}")
    
    # Generate output path if not specified
    if args.output is None:
        config_name = Path(args.config).stem
        input_name = Path(args.input).stem
        args.output = f"vasa-output-{config_name}-{input_name}.mp4"
    
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Input video: {args.input}")
    
    # Create inferencer
    inferencer = VASAInference(
        checkpoint_path=checkpoint_path,
        config_path=args.config
    )

    if args.visualize:
        # Generate visualization outputs
        logger.info(f"Generating visualizations to: {args.vis_dir}")
        logger.info(f"Target image: {args.target_image}")
        inferencer.visualize_inference_outputs(
            input_video=args.input,
            target_image_path=args.target_image,
            output_dir=args.vis_dir
        )
    else:
        # Generate video
        logger.info(f"Output video: {args.output}")
        inferencer.generate_from_video(
            input_video=args.input,
            output_path=args.output,
            fps=args.fps,
            neutral_expression=args.neutral
        )

    # inferencer.visualize_inference_outputs(
    #     input_video="./junk/ovs-GiY_848_1.mp4",
    #     target_image_path="./data/A.png",
    #     output_dir="vis_output"
    # )

    # inferencer.visualize_inference_outputs2(
    #     input_video="./junk/ovs-GiY_848_1.mp4",

    # ) 
