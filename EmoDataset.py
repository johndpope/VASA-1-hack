from PIL import Image
from torch.utils.data import Dataset

from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


import json
import os
from math import cos, sin, pi
from typing import List, Tuple, Dict, Any
from camera import Camera
import cv2
import decord
import librosa
import mediapipe as mp
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader,AVReader


from moviepy.editor import VideoFileClip


class Wav2VecFeatureExtractor:
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    def extract_features_from_wav(self, audio_path, m=2, n=2):
            """
            Extract audio features from a WAV file using Wav2Vec 2.0.

            Args:
                audio_path (str): Path to the WAV audio file.
                m (int): The number of frames before the current frame to include.
                n (int): The number of frames after the current frame to include.

            Returns:
                torch.Tensor: Features extracted from the audio for each frame.
            """
            # Load the audio file
            waveform, sample_rate = sf.read(audio_path)

            # Check if we need to resample
            if sample_rate != self.processor.feature_extractor.sampling_rate:
                waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
                sample_rate = self.processor.feature_extractor.sampling_rate

            # Ensure waveform is a 1D array for a single-channel audio
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

            # Process the audio to extract features
            input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
            input_values = input_values.to(self.device)

            # Pass the input_values to the model
            with torch.no_grad():
                hidden_states = self.model(input_values).last_hidden_state

            num_frames = hidden_states.shape[1]
            feature_dim = hidden_states.shape[2]

            # Concatenate nearby frame features
            all_features = []
            for f in range(num_frames):
                start_frame = max(f - m, 0)
                end_frame = min(f + n + 1, num_frames)
                frame_features = hidden_states[0, start_frame:end_frame, :].flatten()

                # Add padding if necessary
                if f - m < 0:
                    front_padding = torch.zeros((m - f) * feature_dim, device=self.device)
                    frame_features = torch.cat((front_padding, frame_features), dim=0)
                if f + n + 1 > num_frames:
                    end_padding = torch.zeros(((f + n + 1 - num_frames) * feature_dim), device=self.device)
                    frame_features = torch.cat((frame_features, end_padding), dim=0)

                all_features.append(frame_features)

            all_features = torch.stack(all_features, dim=0)
            return all_features


    def extract_features_from_mp4(self, video_path, m=2, n=2):
        """
        Extract audio features from an MP4 file using Wav2Vec 2.0.

        Args:
            video_path (str): Path to the MP4 video file.
            m (int): The number of frames before the current frame to include.
            n (int): The number of frames after the current frame to include.

        Returns:
            torch.Tensor: Features extracted from the audio for each frame.
        """
        # Create the audio file path from the video file path
        audio_path = os.path.splitext(video_path)[0] + '.wav'

        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            # Extract audio from video
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path)

        # Load the audio file
        waveform, sample_rate = sf.read(audio_path)

        # Check if we need to resample
        if sample_rate != self.processor.feature_extractor.sampling_rate:
            waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
            sample_rate = self.processor.feature_extractor.sampling_rate

        # Ensure waveform is a 1D array for a single-channel audio
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

        # Process the audio to extract features
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Pass the input_values to the model
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state

        num_frames = hidden_states.shape[1]
        feature_dim = hidden_states.shape[2]

        # Concatenate nearby frame features
        all_features = []
        for f in range(num_frames):
            start_frame = max(f - m, 0)
            end_frame = min(f + n + 1, num_frames)
            frame_features = hidden_states[0, start_frame:end_frame, :].flatten()

            # Add padding if necessary
            if f - m < 0:
                front_padding = torch.zeros((m - f) * feature_dim, device=self.device)
                frame_features = torch.cat((front_padding, frame_features), dim=0)
            if f + n + 1 > num_frames:
                end_padding = torch.zeros(((f + n + 1 - num_frames) * feature_dim), device=self.device)
                frame_features = torch.cat((frame_features, end_padding), dim=0)

            all_features.append(frame_features)

        all_features = torch.stack(all_features, dim=0)
        return all_features
    



    def extract_features_for_frame(self, video_path, frame_index, m=2):
        """
        Extract audio features for a specific frame from an MP4 file using Wav2Vec 2.0.

        Args:
            video_path (str): Path to the MP4 video file.
            frame_index (int): The index of the frame to extract features for.
            m (int): The number of frames before and after the current frame to include.

        Returns:
            torch.Tensor: Features extracted from the audio for the specified frame.
        """
        # Create the audio file path from the video file path
        audio_path = os.path.splitext(video_path)[0] + '.wav'

        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            # Extract audio from video
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path)

        # Load the audio file
        waveform, sample_rate = sf.read(audio_path)

        # Check if we need to resample
        if sample_rate != self.processor.feature_extractor.sampling_rate:
            waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
            sample_rate = self.processor.feature_extractor.sampling_rate

        # Ensure waveform is a 1D array for a single-channel audio
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

        # Process the audio to extract features
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Pass the input_values to the model
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state

        num_frames = hidden_states.shape[1]
        feature_dim = hidden_states.shape[2]

        # Concatenate nearby frame features
        all_features = []
        start_frame = max(frame_index - m, 0)
        end_frame = min(frame_index + m + 1, num_frames)
        frame_features = hidden_states[0, start_frame:end_frame, :].flatten()
        
        # Add padding if necessary
        if frame_index - m < 0:
            front_padding = torch.zeros((m - frame_index) * feature_dim, device=self.device)
            frame_features = torch.cat((front_padding, frame_features), dim=0)
        if frame_index + m + 1 > num_frames:
            end_padding = torch.zeros(((frame_index + m + 1) - num_frames) * feature_dim, device=self.device)
            frame_features = torch.cat((frame_features, end_padding), dim=0)
        
        all_features.append(frame_features)
        
        return torch.stack(all_features)
    


class EMODataset(Dataset):
    def __init__(self, use_gpu:False,data_dir: str, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None):
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.data_dir = data_dir
        self.transform = transform
        self.stage = stage
        self.feature_extractor = Wav2VecFeatureExtractor(model_name='facebook/wav2vec2-base-960h', device='cuda')

        self.face_mask_generator = FaceHelper()
        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.video_ids = list(self.celebvhq_info['clips'].keys())
        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()


    def __len__(self) -> int:
        
        return len(self.video_ids)

    def augmentation(self, images, transform, state=None):
            if state is not None:
                torch.set_rng_state(state)
            if isinstance(images, List):
                transformed_images = [transform(img) for img in images]
                ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
            else:
                ret_tensor = transform(images)  # (c, h, w)
            return ret_tensor
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        mp4_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        

        if  self.stage == 'stage0-facelocator':
            video_reader = VideoReader(mp4_path, ctx=self.ctx)
            video_length = len(video_reader)
            
            transform_to_tensor = ToTensor()
            # Read frames and generate masks
            vid_pil_image_list = []
            mask_tensor_list = []
            face_locator = FaceHelper()
        
            speeds_tensor_list = []
            for frame_idx in range(video_length):
                # Read frame and convert to PIL Image
                frame = Image.fromarray(video_reader[frame_idx].numpy())

                # Transform the frame
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                vid_pil_image_list.append(pixel_values_frame)


                # Convert the transformed frame back to NumPy array in RGB format
                transformed_frame_np = np.array(pixel_values_frame.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
                transformed_frame_np = cv2.cvtColor(transformed_frame_np, cv2.COLOR_RGB2BGR)

                # Generate the mask using the face mask generator
                mask_np = self.face_mask_generator.generate_face_region_mask_np_image(transformed_frame_np, video_id, frame_idx)

                    # Convert the mask from numpy array to PIL Image
                mask_pil = Image.fromarray(mask_np)

                # Transform the PIL Image mask to a PyTorch tensor
                mask_tensor = transform_to_tensor(mask_pil)
                mask_tensor_list.append(mask_tensor)
            
            # Convert list of lists to a tensor
   
            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list,
                "masks": mask_tensor_list,
            }
        elif  self.stage == 'stage1-0-framesencoder': # so when can freeze this https://github.com/johndpope/Emote-hack/issues/25
            video_reader = VideoReader(mp4_path, ctx=self.ctx)
            video_length = len(video_reader)
            

            vid_pil_image_list = []
         
            
            for frame_idx in range(video_length):
                # Read frame and convert to PIL Image
                frame = Image.fromarray(video_reader[frame_idx].numpy())

                # Transform the frame
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                vid_pil_image_list.append(pixel_values_frame)

            # Convert list of lists to a tensor
            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list
            }
        elif self.stage == 'stage1-vae':
            video_reader = VideoReader(mp4_path, ctx=self.ctx)
            video_length = len(video_reader)
            
            # Read frames and generate masks
            vid_pil_image_list = []
            speeds_tensor_list = []
            face_locator = FaceHelper()
            
            for frame_idx in range(video_length):
                # Read frame and convert to PIL Image
                frame = Image.fromarray(video_reader[frame_idx].numpy())

                # Transform the frame
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                vid_pil_image_list.append(pixel_values_frame)

                # Calculate head rotation speeds at the current frame (previous 1 frames)
                head_rotation_speeds = face_locator.get_head_pose_velocities_at_frame(video_reader, frame_idx, 1)

                # Check if head rotation speeds are successfully calculated
                if head_rotation_speeds:
                    head_tensor = torch.tensor(head_rotation_speeds[0], dtype=torch.float32)  # Convert tuple to tensor
                    speeds_tensor_list.append(head_tensor)
                else:
                    # Provide a default value if no speeds were calculated
                    default_speeds = torch.zeros(3, dtype=torch.float32)  # Create a tensor of shape [3]
                    speeds_tensor_list.append(default_speeds)

            # Convert list of lists to a tensor
            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list,
                "motion_frames": vid_pil_image_list[1:],  # Exclude the first frame as motion frame
                "speeds": speeds_tensor_list
            }


       


        elif self.stage == 'stage2-temporal-audio':
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
        
        elif self.stage == 'stage3-speedlayers':
            av_reader = AVReader(mp4_path, ctx=self.ctx)
            av_length = len(av_reader)
            transform_to_tensor = ToTensor()
            
            # Read frames and generate masks
            vid_pil_image_list = []
            audio_frame_tensor_list = []
            head_rotation_speeds = []
            face_locator = FaceHelper()
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

                 # Calculate head rotation speeds at the current frame (previous 1 frames)
                head_rotation_speeds = face_locator.get_head_pose_velocities_at_frame(video_reader, frame_idx,1)

                # Check if head rotation speeds are successfully calculated
                if head_rotation_speeds:
                    head_tensor = transform_to_tensor(head_rotation_speeds)
                    speeds_tensor_list.append(head_tensor)
                else:
                    # Provide a default value if no speeds were calculated
                    #expected_speed_vector_length = 3
                    #default_speeds = torch.zeros(1, expected_speed_vector_length)  # Shape [1, 3]
                    default_speeds = (0.0, 0.0, 0.0)  # List containing one tuple with three elements
                    head_tensor = transform_to_tensor(default_speeds)
                    speeds_tensor_list.append(head_tensor)
            
            sample = {
                "video_id": video_id,
                "images": vid_pil_image_list,
                "audio_frames": audio_frame_tensor_list,
                "speeds": head_rotation_speeds
            }
        


        return sample

