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
from decord import VideoReader,AVReader


# face warp
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition

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

        self.feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')


        # TODO - make this more dynamic
        driving = os.path.join(self.video_dir, "-2KGPYEFnsU_11.mp4")
        self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        self.video_ids = ["M2Ohb0FAaJU_1"]  # list(self.celebvhq_info['clips'].keys())
        self.video_ids_star = ["-1eKufUP5XQ_4"]  # list(self.celebvhq_info['clips'].keys())
        driving_star = os.path.join(self.video_dir, "-2KGPYEFnsU_8.mp4")
        self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)

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
            
            # Crop the face region from the image
            face_image = bg_removed_image.crop((left, top, right, bottom))
            
            if apply_warp:
                # Convert the face image to a numpy array
                face_array = np.array(face_image)
                
                # Generate random control points for thin-plate-spline warping
                rows, cols = face_array.shape[:2]
                src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
                dst_points = src_points + np.random.randn(4, 2) * (rows * warp_strength)
                
                # Create a PiecewiseAffineTransform object
                tps = PiecewiseAffineTransform()
                tps.estimate(src_points, dst_points)
                
                # Apply the thin-plate-spline warping to the face image
                warped_face_array = warp(face_array, tps, output_shape=(rows, cols))
                
                # Convert the warped face array back to a PIL image
                warped_face_image = Image.fromarray((warped_face_array * 255).astype(np.uint8))
            else:
                warped_face_image = face_image
            
            # Apply the transform if provided
            if transform:
                warped_face_image = warped_face_image.convert("RGB")
                warped_face_tensor = transform(warped_face_image)
                return warped_face_tensor
            
            # Convert the warped PIL image back to a tensor
            # Convert the warped PIL image to RGB format before converting to a tensor
            warped_face_image = warped_face_image.convert("RGB")
            return to_tensor(warped_face_image)

        else:
            return None

    def load_and_process_video(self, video_path: str) -> List[torch.Tensor]:
        # Extract video ID from the path
        video_id = Path(video_path).stem
        output_dir =  Path(self.video_dir + "/" + video_id)
        output_dir.mkdir(exist_ok=True)
        
        processed_frames = []
        tensor_frames = []

        tensor_file_path = output_dir / f"{video_id}_tensors.npz"

        # Check if the tensor file exists
        if tensor_file_path.exists():
            print(f"Loading processed tensors from file: {tensor_file_path}")
            with np.load(tensor_file_path) as data:
                tensor_frames = [torch.tensor(data[key]) for key in data]
        else:
            if self.apply_crop_warping:
                print(f"Warping + Processing and saving video frames to directory: {output_dir}")
            else:
                print(f"Processing and saving video frames to directory: {output_dir}")
            video_reader = VideoReader(video_path, ctx=self.ctx)
            for frame_idx in tqdm(range(len(video_reader)), desc="Processing Video Frames"):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                state = torch.get_rng_state()
                # here we run the color jitter / random flip
                tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform, state)
                processed_frames.append(image_frame)

                if self.apply_crop_warping:
                    transform = transforms.Compose([
                        transforms.Resize((512, 512)), # get the cropped image back to this size - TODO support 256
                        transforms.ToTensor(),
                    ])
                    video_name = Path(video_path).stem

                    # vanilla crop                    
                    tensor_frame1 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=False)
                    # Save frame as PNG image
                    img = to_pil_image(tensor_frame1)
                    img.save(output_dir / f"{frame_idx:06d}.png")
                    tensor_frames.append(tensor_frame1)

                    # vanilla crop + warp                  
                    tensor_frame2 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=True)
                    # Save frame as PNG image
                    img = to_pil_image(tensor_frame2)
                    img.save(output_dir / f"w_{frame_idx:06d}.png")
                    tensor_frames.append(tensor_frame2)
                else:
                    # Save frame as PNG image
                    image_frame.save(output_dir / f"{frame_idx:06d}.png")
                    tensor_frames.append(tensor_frame)

            # Convert tensor frames to numpy arrays and save them
            np.savez_compressed(tensor_file_path, *[tensor_frame.numpy() for tensor_frame in tensor_frames])
            print(f"Processed tensors saved to file: {tensor_file_path}")

        return tensor_frames

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

        if  self.stage == 'stage1':
            video_id = self.video_ids[index]
            # Use next item in the list for video_id_star, wrap around if at the end
            video_id_star = self.video_ids_star[(index + 1) % len(self.video_ids_star)]
            vid_pil_image_list = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id}.mp4"))
            vid_pil_image_list_star = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id_star}.mp4"))
            sample = {
                "video_id": video_id,
                "source_frames": vid_pil_image_list,
                "driving_frames": self.driving_vid_pil_image_list,
                "video_id_star": video_id_star,
                "source_frames_star": vid_pil_image_list_star,
                "driving_frames_star": self.driving_vid_pil_image_list_star,
            }
            return sample
        elif self.stage == 'stage2':
            av_reader = AVReader(mp4_path, ctx=self.ctx)
            av_length = len(av_reader)
            transform_to_tensor = ToTensor()
            
            # Read frames and generate masks
            vid_pil_image_list = []
            audio_frame_tensor_list = []
            
            for frame_idx in range(av_length):
                audio_frame, video_frame = av_reader[frame_idx]
                
                # Read frame and convert to PIL Image
                frame = Image.fromarray(video_frame.numpy())
                
                # Transform the frame
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                vid_pil_image_list.append(pixel_values_frame)
                
                # Convert audio frame to tensor
                audio_frame_tensor = transform_to_tensor(audio_frame.asnumpy())
                audio_frame_tensor_list.append(audio_frame_tensor)
            
            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list,
                "audio_frames": audio_frame_tensor_list,
            }
            return sample