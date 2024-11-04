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
# face warp
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
import gc

from memory_profiler import profile


class EMODataset(Dataset):
    def __init__(self, use_gpu: False, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False, use_greenscreen=False, apply_crop_warping=False):
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
        self.apply_crop_warping = apply_crop_warping
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = cpu()

        self.video_ids = list(self.celebvhq_info['clips'].keys())

            # Load videos with proper cleanup
        random_video_id = random.choice(self.video_ids)
        driving = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        print("driving:", driving)
        
        self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
        gc.collect()  # Force garbage collection
        
        self.video_ids_star = list(self.celebvhq_info['clips'].keys())
        random_video_id = random.choice(self.video_ids_star)
        driving_star = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        print("driving_star:", driving_star)
        
        self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)
        torch.cuda.empty_cache()
        gc.collect()

    def __len__(self) -> int:
        return len(self.video_ids)

    def warp_and_crop_face(self, image_tensor, video_name, frame_idx, transform=None, output_dir="output_images", warp_strength=0.01, apply_warp=False):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the file path
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")
        
        # Check if the file already exists
        if os.path.exists(output_path):
            # Load and return the existing image as a tensor
            existing_image = Image.open(output_path).convert("RGBA")
            return to_tensor(existing_image)
        
        # Check if the input tensor has a batch dimension and handle it
        if image_tensor.ndim == 4:
            # Assuming batch size is the first dimension, process one image at a time
            image_tensor = image_tensor.squeeze(0)
        
        # Convert the single image tensor to a PIL Image
        image = to_pil_image(image_tensor)
        
        # Remove the background from the image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
        
        # Convert the image to RGB format to make it compatible with face_recognition
        bg_removed_image_rgb = bg_removed_image.convert("RGB")
        
        # Detect the face in the background-removed RGB image using the numpy array
        face_locations = face_recognition.face_locations(np.array(bg_removed_image_rgb))
        
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            
            # Automatically choose sweet spot to crop.
            # https://github.com/tencent-ailab/V-Express/blob/main/assets/crop_example.jpeg
            
            face_width = right - left
            face_height = bottom - top

            # Calculate the padding amount based on face size and output dimensions
            pad_width = int(face_width * 0.5)
            pad_height = int(face_height * 0.5)

            # Expand the cropping coordinates with the calculated padding
            left_with_pad = max(0, left - pad_width)
            top_with_pad = max(0, top - pad_height)
            right_with_pad = min(bg_removed_image.width, right + pad_width)
            bottom_with_pad = min(bg_removed_image.height, bottom + pad_height)

            # Crop the face region from the image with padding
            face_image_with_pad = bg_removed_image.crop((left_with_pad, top_with_pad, right_with_pad, bottom_with_pad))

            # Crop the face region from the image without padding
            face_image_no_pad = bg_removed_image.crop((left, top, right, bottom))
            
            if apply_warp:
                # Convert the face image to a numpy array
                face_array_with_pad = np.array(face_image_with_pad)
                face_array_no_pad = np.array(face_image_no_pad)
                
                # Generate random control points for thin-plate-spline warping
                rows_with_pad, cols_with_pad = face_array_with_pad.shape[:2]
                rows_no_pad, cols_no_pad = face_array_no_pad.shape[:2]
                src_points_with_pad = np.array([[0, 0], [cols_with_pad-1, 0], [0, rows_with_pad-1], [cols_with_pad-1, rows_with_pad-1]])
                src_points_no_pad = np.array([[0, 0], [cols_no_pad-1, 0], [0, rows_no_pad-1], [cols_no_pad-1, rows_no_pad-1]])
                dst_points_with_pad = src_points_with_pad + np.random.randn(4, 2) * (rows_with_pad * warp_strength)
                dst_points_no_pad = src_points_no_pad + np.random.randn(4, 2) * (rows_no_pad * warp_strength)
                
                # Create a PiecewiseAffineTransform object
                tps_with_pad = PiecewiseAffineTransform()
                tps_with_pad.estimate(src_points_with_pad, dst_points_with_pad)
                tps_no_pad = PiecewiseAffineTransform()
                tps_no_pad.estimate(src_points_no_pad, dst_points_no_pad)
                
                # Apply the thin-plate-spline warping to the face images
                warped_face_array_with_pad = warp(face_array_with_pad, tps_with_pad, output_shape=(rows_with_pad, cols_with_pad))
                warped_face_array_no_pad = warp(face_array_no_pad, tps_no_pad, output_shape=(rows_no_pad, cols_no_pad))
                
                # Convert the warped face arrays back to PIL images
                warped_face_image_with_pad = Image.fromarray((warped_face_array_with_pad * 255).astype(np.uint8))
                warped_face_image_no_pad = Image.fromarray((warped_face_array_no_pad * 255).astype(np.uint8))
            else:
                warped_face_image_with_pad = face_image_with_pad
                warped_face_image_no_pad = face_image_no_pad
            
            # Apply the transform if provided
            if transform:
                warped_face_image_with_pad = warped_face_image_with_pad.convert("RGB")
                warped_face_image_no_pad = warped_face_image_no_pad.convert("RGB")
                warped_face_tensor_with_pad = transform(warped_face_image_with_pad)
                warped_face_tensor_no_pad = transform(warped_face_image_no_pad)
                return warped_face_tensor_with_pad, warped_face_tensor_no_pad
            
            # Convert the warped PIL images back to tensors
            warped_face_image_with_pad = warped_face_image_with_pad.convert("RGB")
            warped_face_image_no_pad = warped_face_image_no_pad.convert("RGB")
            return to_tensor(warped_face_image_with_pad), to_tensor(warped_face_image_no_pad)

        else:
            return None, None

    @profile
    def load_and_process_video(self, video_path: str) -> List[torch.Tensor]:
        video_id = Path(video_path).stem
        output_dir = Path(self.video_dir + "/" + video_id)
        output_dir.mkdir(exist_ok=True)
        
        tensor_file_path = output_dir / f"{video_id}_tensors.npz"
        
        if tensor_file_path.exists():
            print(f"Loading processed tensors from file: {tensor_file_path}")
            with np.load(tensor_file_path) as data:
                tensor_frames = [torch.tensor(data[key]) for key in data]
                del data  # Explicitly delete the loaded data
                gc.collect()  # Force garbage collection
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
                
                if self.apply_crop_warping:
                    transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])
                    video_name = Path(video_path).stem
                    
                    _, sweet_tensor_frame1 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=False)
                    if sweet_tensor_frame1 is not None:
                        img = to_pil_image(sweet_tensor_frame1)
                        img.save(output_dir / f"s_{frame_idx:06d}.png")
                        tensor_frames.append(sweet_tensor_frame1)
                        del img
                    
                    _, sweet_tensor_frame2 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=True)
                    if sweet_tensor_frame2 is not None:
                        img = to_pil_image(sweet_tensor_frame2)
                        img.save(output_dir / f"sw_{frame_idx:06d}.png")
                        tensor_frames.append(sweet_tensor_frame2)
                        del img
                else:
                    image_frame.save(output_dir / f"{frame_idx:06d}.png")
                    tensor_frames.append(tensor_frame)
                
                # Clean up intermediate objects
                del frame, tensor_frame, image_frame
                if frame_idx % 10 == 0:  # Periodic cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Clean up video reader
            del video_reader
            gc.collect()
            
            # Save processed tensors
            np.savez_compressed(tensor_file_path, *[tensor_frame.numpy() for tensor_frame in tensor_frames])
            print(f"Processed tensors saved to file: {tensor_file_path}")
            
            return tensor_frames
            
        finally:
            # Clean up all intermediate objects
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
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")  # Use RGBA to keep transparency

        if self.use_greenscreen:
            # Create a green screen background
            green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))  # Green color

            # Composite the image onto the green screen
            final_image = Image.alpha_composite(green_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        final_image = final_image.convert("RGB")  # Convert to RGB format
        return final_image

    def save_video(self, frames, output_path, fps=30):
        print(f"Saving video with {len(frames)} frames to {output_path}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to other codecs if needed
        height, width, _ = np.array(frames[0]).shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR format

        out.release()
        print(f"Video saved to {output_path}")

    def process_video(self, video_path):
        processed_frames = self.process_video_frames(video_path)
        return processed_frames

    def process_video_frames(self, video_path: str) -> List[torch.Tensor]:
        video_reader = VideoReader(video_path, ctx=self.ctx)
        processed_frames = []
        for frame_idx in tqdm(range(len(video_reader)), desc="Processing Video Frames"):
            frame = Image.fromarray(video_reader[frame_idx].numpy())
            state = torch.get_rng_state()
            tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform, state)
            processed_frames.append(image_frame)
        return processed_frames



    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                video_id = self.video_ids[index]
                video_id_star = self.video_ids_star[(index + 1) % len(self.video_ids_star)]
                
                # Load videos with cleanup
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

