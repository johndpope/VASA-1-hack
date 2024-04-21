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
from PIL import Image
from torch.utils.data import Dataset

from transformers import Wav2Vec2Model, Wav2Vec2Processor


from modules.real3d.facev2v_warp.layers import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from modules.real3d.facev2v_warp.func_utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)


class Canonical3DVolumeEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 64, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        return x

class IdentityEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 512, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1)
        return x
    
class HeadPoseEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 3, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1)
        return x

class FacialDynamicsEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 256, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1) 
        return x


class ExpressiveDisentangledFaceLatentSpace(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.canonical_3d_volume_encoder = Canonical3DVolumeEncoder()
        self.identity_encoder = IdentityEncoder()
        self.head_pose_encoder = HeadPoseEncoder()
        self.facial_dynamics_encoder = FacialDynamicsEncoder()
        
        # Decoder
        self.decoder = Decoder()
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.pairwise_transfer_loss = nn.L1Loss()
        self.identity_similarity_loss = nn.CosineSimilarity()

    def forward(self, img1, img2):
        # Extract latent variables for img1
        V_a1 = self.canonical_3d_volume_encoder(img1)
        z_id1 = self.identity_encoder(img1)
        z_pose1 = self.head_pose_encoder(img1)
        z_dyn1 = self.facial_dynamics_encoder(img1)
        
        # Extract latent variables for img2
        V_a2 = self.canonical_3d_volume_encoder(img2)
        z_id2 = self.identity_encoder(img2)
        z_pose2 = self.head_pose_encoder(img2)
        z_dyn2 = self.facial_dynamics_encoder(img2)
        
        # Reconstruct images
        img1_recon = self.decoder(V_a1, z_id1, z_pose1, z_dyn1)
        img2_recon = self.decoder(V_a2, z_id2, z_pose2, z_dyn2)
        
        # Pairwise head pose and facial dynamics transfer
        img1_pose_transfer = self.decoder(V_a1, z_id1, z_pose2, z_dyn1)
        img2_dyn_transfer = self.decoder(V_a2, z_id2, z_pose2, z_dyn1)
        
        # Cross-identity pose and facial motion transfer
        img1_cross_id_transfer = self.decoder(V_a1, z_id2, z_pose1, z_dyn1)
        img2_cross_id_transfer = self.decoder(V_a2, z_id1, z_pose2, z_dyn2)
        
        return img1_recon, img2_recon, img1_pose_transfer, img2_dyn_transfer, img1_cross_id_transfer, img2_cross_id_transfer

    def training_step(self, img1, img2):
        # Forward pass
        img1_recon, img2_recon, img1_pose_transfer, img2_dyn_transfer, img1_cross_id_transfer, img2_cross_id_transfer = self.forward(img1, img2)
        
        # Reconstruction loss
        loss_recon = self.reconstruction_loss(img1_recon, img1) + self.reconstruction_loss(img2_recon, img2)
        
        # Pairwise transfer loss
        loss_pairwise_transfer = self.pairwise_transfer_loss(img1_pose_transfer, img2_dyn_transfer)
        
        # Identity similarity loss
        id_feat1 = extract_identity_features(img1)
        id_feat1_cross_id_transfer = extract_identity_features(img1_cross_id_transfer)
        id_feat2 = extract_identity_features(img2)
        id_feat2_cross_id_transfer = extract_identity_features(img2_cross_id_transfer)
        loss_id_sim = 1 - self.identity_similarity_loss(id_feat1, id_feat1_cross_id_transfer) + 1 - self.identity_similarity_loss(id_feat2, id_feat2_cross_id_transfer)
        
        # Total loss
        total_loss = loss_recon + loss_pairwise_transfer + loss_id_sim
        
        return total_loss

class FaceEncoder(nn.Module):
    def __init__(self, use_weight_norm=False):
        super(FaceEncoder, self).__init__()
        # Define encoder architecture for extracting latent variables
        # - Canonical 3D appearance volume (V_can)
        # - Identity code (z_id)
        # - 3D head pose (z_pose)
        # - Facial dynamics code (z_dyn)
        # ...
    
        self.in_conv = ConvBlock2D("CNA", 3, 64, 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(
            DownBlock2D(64, 128, use_weight_norm),
            DownBlock2D(128, 256, use_weight_norm),
            DownBlock2D(256, 512, use_weight_norm)
        )
        self.mid_conv = nn.Conv2d(512, 32 * 16, 1, 1, 0)
        self.res = nn.Sequential(
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm)
        )
        self.identity_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        self.head_pose_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )
        self.facial_dynamics_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, 32, 16, H, W)
        appearance_volume = self.res(x)
        
        # Extract identity code, head pose, and facial dynamics code
        x = x.view(N, -1)
        identity_code = self.identity_encoder(x)
        head_pose = self.head_pose_encoder(x)
        facial_dynamics = self.facial_dynamics_encoder(x)
        
        return appearance_volume, identity_code, head_pose, facial_dynamics

class FaceDecoder(nn.Module):
    def __init__(self, use_weight_norm=True):
        super(FaceDecoder, self).__init__()
        self.in_conv = ConvBlock2D("CNA", 32 * 16, 256, 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.res = nn.Sequential(
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 256, 128, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 128, 64, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 64, 3, 7, 1, 3, use_weight_norm, activation_type="tanh")
        )

    def forward(self, appearance_volume, identity_code, head_pose, facial_dynamics):
        N, _, D, H, W = appearance_volume.shape
        x = appearance_volume.view(N, -1, H, W)
        x = self.in_conv(x)
        x = self.res(x)
        
        # Apply 3D warping based on head pose and facial dynamics
        # ...
        
        face_image = self.up(x)
        return face_image

# Define the diffusion transformer for holistic facial dynamics generation
class DiffusionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout=0.1):
        super(DiffusionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, audio_features, gaze_direction, head_distance, emotion_offset):
        # Concatenate input features
        input_features = torch.cat([x, audio_features, gaze_direction, head_distance, emotion_offset], dim=-1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(input_features)
        
        x = self.norm(x)
        return x


# The HeadPoseExpressionEstimator takes an input face image and estimates the head pose (yaw, pitch, roll) 
# and expression deformation parameters. It uses a series of residual bottleneck blocks followed by fully connected layers 
# to predict the head pose angles, translation vector, and deformation parameters.
class HeadPoseExpressionEstimator(nn.Module):
    def __init__(self, use_weight_norm=False):
        super(HeadPoseExpressionEstimator, self).__init__()
        self.pre_layers = nn.Sequential(
            ConvBlock2D("CNA", 3, 64, 7, 2, 3, use_weight_norm),
            nn.MaxPool2d(3, 2, 1)
        )
        self.res_layers = nn.Sequential(
            self._make_layer(0, 64, 256, 3, use_weight_norm),
            self._make_layer(1, 256, 512, 4, use_weight_norm),
            self._make_layer(2, 512, 1024, 6, use_weight_norm),
            self._make_layer(3, 1024, 2048, 3, use_weight_norm)
        )
        self.fc_yaw = nn.Linear(2048, 66)
        self.fc_pitch = nn.Linear(2048, 66)
        self.fc_roll = nn.Linear(2048, 66)
        self.fc_t = nn.Linear(2048, 3)
        self.fc_delta = nn.Linear(2048, 3 * 15)

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return nn.Sequential(
            ResBottleneck(in_channels, out_channels, stride, use_weight_norm),
            *[ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)]
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        t = self.fc_t(x)
        delta = self.fc_delta(x)
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta





class FaceHelper:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize FaceDetection once here
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        self.HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
    def __del__(self):
        self.face_detection.close()
        self.face_mesh.close()

    def generate_face_region_mask(self,frame_image, video_id=0,frame_idx=0):
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(video_id,frame_idx,frame_np)

    def generate_face_region_mask_np_image(self,frame_np, video_id=0,frame_idx=0, padding=10):
        # Convert from RGB to BGR for MediaPipe processing
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Optionally save a debug image
        debug_image = mask
        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Draw a rectangle on the debug image for each detection
                cv2.rectangle(debug_image, (xmin, ymin), (xmin + bbox_width, ymin + bbox_height), (0, 255, 0), 2)
        # Check that detections are not None
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Calculate padded coordinates
                pad_xmin = max(0, xmin - padding)
                pad_ymin = max(0, ymin - padding)
                pad_xmax = min(width, xmin + bbox_width + padding)
                pad_ymax = min(height, ymin + bbox_height + padding)

                # Draw a white padded rectangle on the mask
                mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = 255

               
                # cv2.rectangle(debug_image, (pad_xmin, pad_ymin), 
                            #   (pad_xmax, pad_ymax), (255, 255, 255), thickness=-1)
                # cv2.imwrite(f'./temp/debug_face_mask_{video_id}-{frame_idx}.png', debug_image)

        return mask

    
    def generate_face_region_mask_pil_image(self,frame_image,video_id=0, frame_idx=0):
        # Convert from PIL Image to NumPy array in BGR format
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(frame_np,video_id,frame_idx,)
    


    def calculate_pose(self, face2d):
            """Calculates head pose from detected facial landmarks using 
            Perspective-n-Point (PnP) pose computation:
            
            https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
            """
            # print('Computing head pose from tracking data...')
            # for idx, time in enumerate(self.face2d['time']):
            #     # print(time)
            #     self.pose['time'].append(time)
            #     self.pose['frame'].append(self.face2d['frame'][idx])
            #     face2d = self.face2d['key landmark positions'][idx]
            face3d = [[0, -1.126865, 7.475604], # 1
                        [-4.445859, 2.663991, 3.173422], # 33
                        [-2.456206,	-4.342621, 4.283884], # 61
                        [0, -9.403378, 4.264492], # 199
                        [4.445859, 2.663991, 3.173422], # 263
                        [2.456206, -4.342621, 4.283884]] # 291
            face2d = np.array(face2d, dtype=np.float64)
            face3d = np.array(face3d, dtype=np.float64)

            camera = Camera()
            success, rot_vec, trans_vec = cv2.solvePnP(face3d,
                                                        face2d,
                                                        camera.internal_matrix,
                                                        camera.distortion_matrix,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
            
            rmat = cv2.Rodrigues(rot_vec)[0]

            P = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
            eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
            yaw = eulerAngles[1, 0]
            pitch = eulerAngles[0, 0]
            roll = eulerAngles[2,0]
            
            if pitch < 0:
                pitch = - 180 - pitch
            elif pitch >= 0: 
                pitch = 180 - pitch
            
            yaw *= -1
            pitch *= -1
            
            # if nose2d:
            #     nose2d = nose2d
            #     p1 = (int(nose2d[0]), int(nose2d[1]))
            #     p2 = (int(nose2d[0] - yaw * 2), int(nose2d[1] - pitch * 2))
            
            return yaw, pitch, roll 

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
        return img

    def get_head_pose(self, image_path):
        """
        Given an image, estimate the head pose (roll, pitch, yaw angles).

        Args:
            image: Image to estimate head pose.

        Returns:
            tuple: Roll, Pitch, Yaw angles if face landmarks are detected, otherwise None.
        """


    # Define the landmarks that represent the head pose.

        image = cv2.imread(image_path)
        # Convert the image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect face landmarks.
        results = self.mp_face_mesh.process(image_rgb)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []


        if results.multi_face_landmarks:       
            for face_landmarks in results.multi_face_landmarks:
                key_landmark_positions=[]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.HEAD_POSE_LANDMARKS:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                        landmark_position = [x,y]
                        key_landmark_positions.append(landmark_position)
                # Convert to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix
                focal_length = img_w  # Assuming fx = fy
                cam_matrix = np.array(
                    [[focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]]
                )

                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to get rotation vector
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
                self.draw_axis(image, yaw, pitch, roll)
                debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
                cv2.imwrite(debug_image_path, image)
                print(f'Debug image saved to {debug_image_path}')
                
                return roll, pitch, yaw 

        return None




    def get_head_pose_velocities_at_frame(self, video_reader: VideoReader, frame_index, n_previous_frames=2):

        # Adjust frame_index if it's larger than the total number of frames
        total_frames = len(video_reader)
        frame_index = min(frame_index, total_frames - 1)

        # Calculate starting index for previous frames
        start_index = max(0, frame_index - n_previous_frames)

        head_poses = []
        for idx in range(start_index, frame_index + 1):
            # idx is the frame index you want to access
            frame_tensor = video_reader[idx]

            #  check emodataset decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
            # Assert that frame_tensor is a PyTorch tensor
            assert isinstance(frame_tensor, torch.Tensor), "Expected a PyTorch tensor"

            image = video_reader[idx].numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:       
                for face_landmarks in results.multi_face_landmarks:
                    key_landmark_positions=[]
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in self.HEAD_POSE_LANDMARKS:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                            landmark_position = [x,y]
                            key_landmark_positions.append(landmark_position)
                    # Convert to numpy arrays
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera matrix
                    focal_length = img_w  # Assuming fx = fy
                    cam_matrix = np.array(
                        [[focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]]
                    )

                    # Distortion matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP to get rotation vector
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix
                    )
                    yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                    head_poses.append((roll, pitch, yaw))

        # Calculate velocities
        head_velocities = []
        for i in range(len(head_poses) - 1):
            roll_diff = head_poses[i + 1][0] - head_poses[i][0]
            pitch_diff = head_poses[i + 1][1] - head_poses[i][1]
            yaw_diff = head_poses[i + 1][2] - head_poses[i][2]
            head_velocities.append((roll_diff, pitch_diff, yaw_diff))

        return head_velocities




