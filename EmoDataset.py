from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
from decord import VideoReader, cpu
from rembg import remove
import io
import numpy as np
import decord
import subprocess
from tqdm import tqdm
import cv2
from pathlib import Path
from torchvision.transforms.functional import to_pil_image, to_tensor
import random
from skimage.transform import PiecewiseAffineTransform, warp
import gc
from memory_profiler import profile

class EMODataset(Dataset):
    def __init__(self, use_gpu: False, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False, use_greenscreen=False, apply_warping=False):
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.transform = transform
        self.stage = stage
        self.pixel_transform = transform
        self.drop_ratio = drop_ratio
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.apply_warping = apply_warping
        
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu
        decord.bridge.set_bridge('torch')
        self.ctx = cpu()

        self.video_ids = list(self.celebvhq_info['clips'].keys())

        # Load videos with proper cleanup
        random_video_id = random.choice(self.video_ids)
        driving = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        print("driving:", driving)
        
        self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        torch.cuda.empty_cache()
        gc.collect()
        
        self.video_ids_star = list(self.celebvhq_info['clips'].keys())
        random_video_id = random.choice(self.video_ids_star)
        driving_star = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        print("driving_star:", driving_star)
        
        self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)
        torch.cuda.empty_cache()
        gc.collect()

    def __len__(self) -> int:
        return len(self.video_ids)

    def apply_warp_transform(self, image_tensor, warp_strength=0.01):
        # Convert tensor to numpy array for warping
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        
        image = to_pil_image(image_tensor)
        image_array = np.array(image)
        
        # Generate random control points for warping
        rows, cols = image_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * warp_strength)
        
        # Create and apply the warping transform
        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)
        
        # Apply warping to each channel separately to handle RGB
        warped_array = np.zeros_like(image_array)
        for i in range(image_array.shape[2]):
            warped_array[..., i] = warp(image_array[..., i], tps, output_shape=(rows, cols))
        
        # Convert back to PIL Image and then to tensor
        warped_image = Image.fromarray((warped_array * 255).astype(np.uint8))
        return to_tensor(warped_image)

    def load_and_process_video(self, video_path: str) -> List[torch.Tensor]:
        video_id = Path(video_path).stem
        output_dir = Path(self.video_dir + "/" + video_id)
        output_dir.mkdir(exist_ok=True)
        
        tensor_file_path = output_dir / f"{video_id}_tensors.npz"
        
        if tensor_file_path.exists():
            print(f"Loading processed tensors from file: {tensor_file_path}")
            with np.load(tensor_file_path) as data:
                tensor_frames = [torch.tensor(data[key]) for key in data]
                del data
                gc.collect()
                return tensor_frames
        
        processed_frames = []
        tensor_frames = []
        
        try:
            video_reader = VideoReader(video_path, ctx=self.ctx)
            total_frames = len(video_reader)
            
            for frame_idx in tqdm(range(total_frames), desc="Processing Video Frames"):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                state = torch.get_rng_state()
                tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform, state)
                
                if self.apply_warping:
                    tensor_frame = self.apply_warp_transform(tensor_frame)
                    image_frame = to_pil_image(tensor_frame)
                
                image_frame.save(output_dir / f"{frame_idx:06d}.png")
                tensor_frames.append(tensor_frame)
                
                del frame, tensor_frame, image_frame
                if frame_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            del video_reader
            gc.collect()
            
            np.savez_compressed(tensor_file_path, *[tensor_frame.numpy() for tensor_frame in tensor_frames])
            print(f"Processed tensors saved to file: {tensor_file_path}")
            
            return tensor_frames
            
        finally:
            del processed_frames
            gc.collect()
            torch.cuda.empty_cache()

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)

        if isinstance(images, list):
            if self.remove_background:
                images = [self.remove_bg(img) for img in images]
            transformed_images = [transform(img) for img in tqdm(images, desc="Augmenting Images")]
            ret_tensor = torch.stack(transformed_images, dim=0)
        else:
            if self.remove_background:
                images = self.remove_bg(images)
            ret_tensor = transform(images)

        return ret_tensor, images

    def remove_bg(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")

        if self.use_greenscreen:
            green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))
            final_image = Image.alpha_composite(green_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        final_image = final_image.convert("RGB")
        return final_image

    def save_video(self, frames, output_path, fps=30):
        print(f"Saving video with {len(frames)} frames to {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = np.array(frames[0]).shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"Video saved to {output_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                video_id = self.video_ids[index]
                video_id_star = self.video_ids_star[(index + 1) % len(self.video_ids_star)]
                
                vid_pil_image_list = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id}.mp4"))
                gc.collect()
                torch.cuda.empty_cache()
                
                vid_pil_image_list_star = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id_star}.mp4"))
                gc.collect()
                torch.cuda.empty_cache()
                
                sample = {
                    "video_id": video_id,
                    "source_frames": vid_pil_image_list,
                    "driving_frames": self.driving_vid_pil_image_list,
                    "video_id_star": video_id_star,
                    "source_frames_star": vid_pil_image_list_star,
                    "driving_frames_star": self.driving_vid_pil_image_list_star,
                }
                return sample
                
            except Exception as e:
                print(f"Error loading video {index}: {e}")
                gc.collect()
                torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup method called when the dataset object is destroyed"""
        del self.driving_vid_pil_image_list
        del self.driving_vid_pil_image_list_star
        gc.collect()
        torch.cuda.empty_cache()