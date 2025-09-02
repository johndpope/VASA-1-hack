import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Union, Tuple
from torchvision import transforms
import time
from PIL import Image
import PIL
import importlib
to_512 = lambda x: x.resize((512, 512), Image.LANCZOS)

class VideoProcessor:
    def __init__(self, 
                 model,
                 device: str = 'cuda',
                 image_size: int = 512):
        self.model = model
        self.device = device
        self.image_size = image_size
        
        # Set up transforms
        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()
        self.resize = transforms.Resize((image_size, image_size), Image.LANCZOS)
        
        # Initialize tracking states
        self.source_processed = False
        self.source_data = None
    
    def _prepare_image(self, image: Union[PIL.Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert image to tensor and ensure RGB format."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize and convert to tensor
        image = self.resize(image)
        tensor = self.to_tensor(image).unsqueeze(0)
        
        # Move to device
        return tensor.to(self.device)

    def process_source(self, source_image: Union[PIL.Image.Image, np.ndarray]) -> None:
        """Process source identity image."""
        # Prepare source image
        source_tensor = self._prepare_image(source_image)
        
        # Create mask (same spatial size but single channel)
        source_mask = torch.ones((1, 1, self.image_size, self.image_size), 
                               device=self.device)
        
        # Create data dictionary for source processing
        data_dict = {
            'source_img': source_tensor,
            'source_mask': source_mask
        }
        
        # Process through model
        with torch.no_grad():
            _, _, _, processed_dict = self.model(data_dict, phase='inference')
        
        # Store source data for reuse
        self.source_data = processed_dict
        self.source_processed = True


    def process_frame(self, frame: Union[PIL.Image.Image, np.ndarray]) -> PIL.Image.Image:
        """Process a single frame using the source identity."""
        assert self.source_processed, "Must process source image first"
        
        # Process as a batch of size 1
        frames = self.process_batch([frame])
        return frames[0]

    def process_batch(self, frames: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
        """Process a batch of frames together with proper background handling."""
        assert self.source_processed, "Must process source image first"
        
        try:
            # Prepare batch tensors
            driver_tensors = torch.cat([self._prepare_image(f) for f in frames])
            
            # Create batch mask using face parsing
            batch_size = len(frames)
            driver_masks = []
            
            # Get face masks for each frame
            for frame_tensor in driver_tensors:
                _, _, mask, _ = self.model.face_idt.forward(frame_tensor.unsqueeze(0))
                driver_masks.append(mask)
            
            driver_mask = torch.cat(driver_masks)
            
            # Create data dictionary with masks
            data_dict = {
                'source_processed': True,
                **self.source_data,
                'target_img': driver_tensors,
                'target_mask': driver_mask
            }
            
            # Process through model
            with torch.no_grad():
                _, _, _, processed_dict = self.model(data_dict, phase='inference')
                
                if 'pred_target_img' not in processed_dict:
                    raise ValueError("Model did not generate target images")
                
                # Convert outputs to PIL images
                outputs = processed_dict['pred_target_img'].cpu()
                return [self.to_image(out.clamp(0, 1)) for out in outputs]
                
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            print("Batch size:", batch_size)
            print("Driver tensor shape:", driver_tensors.shape)
            raise

    def save_video(self,
            source_img: PIL.Image.Image,
            generated_frames: List[PIL.Image.Image], 
            driving_frames: List[PIL.Image.Image],
            output_path: str,
            fps: float = 30.0,
            size: Tuple[int, int] = (512, 512)):
        """Save video with source | driving | generated frames side by side."""
        # Debug info
        print(f"Frames to process: driving={len(driving_frames)}, generated={len(generated_frames)}")
        
        # Setup video writer
        out_size = (size[0] * 3, size[1])  # 3 images side by side
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
        
        # Convert and resize source image once
        source_array = np.array(source_img.convert('RGB').resize(size))
        
        try:
            for i, (drive_frame, gen_frame) in enumerate(zip(driving_frames, generated_frames)):
                # Convert driving frame
                drive_array = np.array(drive_frame.convert('RGB').resize(size))
                
                # Convert generated frame
                gen_array = np.array(gen_frame.convert('RGB').resize(size))
                
                # Combine horizontally: source | driving | generated
                composite = np.concatenate([source_array, drive_array, gen_array], axis=1)
                
                # Write frame
                out.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                
                if i % 50 == 0:
                    print(f"Processed frame {i}")
                    
        finally:
            out.release()
            
    def process_video(self,
                 source_image: Union[PIL.Image.Image, str],
                 video_path: str,
                 output_path: str,
                 max_frames: Optional[int] = None,
                 fps: float = 30.0,
                 batch_size: int = 1) -> None:
        """Process entire video file."""
        # Process source image
        if not self.source_processed:
            print("Processing source image...")
            self.process_source(source_image)
        
        # Initialize frame storage
        generated_frames = []
        driving_frames = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Process frames
        batch_frames = []
        processed_frames = 0
        start_time = time.time()
        
        print(f"Processing {total_frames} frames...")
        
        try:
            while processed_frames < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Add to batch
                batch_frames.append(frame_pil)
                driving_frames.append(frame_pil)  # Store original frame
                
                # Process batch when ready
                if len(batch_frames) == batch_size or processed_frames + len(batch_frames) == total_frames:
                    try:
                        # Generate new frames
                        new_frames = self.process_batch(batch_frames)
                        generated_frames.extend(new_frames)
                        
                        # Update progress
                        processed_frames += len(batch_frames)
                        batch_frames = []  # Clear batch
                        
                        # Show progress
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            fps_rate = processed_frames / elapsed
                            eta = (total_frames - processed_frames) / fps_rate
                            print(f"Frame {processed_frames}/{total_frames} | "
                                f"FPS: {fps_rate:.1f} | "
                                f"ETA: {eta:.1f}s")
                    
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        batch_frames = []
        
        finally:
            cap.release()
        
        # Save video
        print("\nSaving video...")
        print(f"Total frames: driving={len(driving_frames)}, generated={len(generated_frames)}")
        
        self.save_video(
            source_img=source_image,
            generated_frames=generated_frames,
            driving_frames=driving_frames,
            output_path=output_path,
            fps=fps
        )



import sys
sys.path.append('.')
from omegaconf import OmegaConf
def create_driven_video(
    source_path: str,
    video_path: str, 
    output_path: str,
    model_path: str,
    device: str = 'cuda',
    max_frames: Optional[int] = None,
    fps: float = 30.0,
    batch_size: int = 1
):
    """Create driven video using the model directly."""
    # Initialize model
    #model = Model(args=None, training=False)  # Configure args as needed
    # from models.volumetric_avatar.stage_1.volumetric_avatar.va import Model
    args = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    model = importlib.import_module(f'models.stage_1.volumetric_avatar.va').Model(args, training=False)

    # Load model weights
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Create video processor
    processor = VideoProcessor(
        model=model,
        device=device,
        image_size=512  # Adjust as needed
    )
    
    # Process video
    source_img = to_512(Image.open(source_path))
    processor.process_video(
        source_image=source_img,
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames,
        fps=fps,
        batch_size=batch_size
    )



# Usage example
if __name__ == "__main__":
    create_driven_video(
        source_path='data/IMG_1.png',
        video_path='data/VID_1.mp4',
        output_path='data/result-main.mp4',
        model_path='logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth',
        max_frames=None,  # Process all frames
        fps=30.0,
        batch_size=16  # Adjust based on GPU memory
    )


 
