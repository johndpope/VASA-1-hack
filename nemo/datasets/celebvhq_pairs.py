import os
import json
import torch
import lmdb
import pickle
import numpy as np
from torch.utils import data
from torchvision import transforms
import albumentations as A
import cv2
from PIL import Image
import io
import random
from pathlib import Path
from face3d.morphable_model import MorphabelModel
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_landmark_detection import FaceAlignment

class CelebVHQDataset(data.Dataset):
    """Integrated dataset class that handles both preprocessing and data loading"""
    def __init__(self,
                 data_root,
                 json_path,
                 output_lmdb_path,
                 num_source_frames=1,
                 num_target_frames=1,
                 image_size=256,
                 image_additional_size=None,
                 phase='train',
                 device='cuda',
                 augment_geometric=False,
                 augment_color=True,
                 output_aug_warp=False,
                 output_aug_warp_out=True,
                 use_masked_aug=False,
                 aug_warp_size=256,
                 warp_aug_color_coef=1.0,
                 aug_color_coef=1.0,
                 gray_source_prob=0.0,
                 shift_limit=0.1,
                 scale_cond_tr=0.8,
                 rot_aug_angle=0,
                 rand_crop_prob=0.0,
                 rand_crop_scale=0.9,
                 preprocess_batch_size=32,
                 bfm_path='/media/oem/12TB/FaceFitting/models/BFM.mat'):
        
        super().__init__()
        self.data_root = data_root
        self.output_lmdb_path = output_lmdb_path
        self.image_size = image_size
        self.device = device
        self.phase = phase
        self.preprocess_batch_size = preprocess_batch_size
        
        # Initialize face processing models
        self.init_face_models()
        
        # Initialize dataset parameters
        self.init_dataset_params(
            num_source_frames, num_target_frames, image_size,
            image_additional_size, augment_geometric, augment_color,
            output_aug_warp, output_aug_warp_out, use_masked_aug,
            aug_warp_size, warp_aug_color_coef, aug_color_coef,
            gray_source_prob, shift_limit, scale_cond_tr,
            rot_aug_angle, rand_crop_prob, rand_crop_scale
        )
        
        # Process videos and create LMDB if needed
        if not os.path.exists(output_lmdb_path):
            self.preprocess_videos(json_path)
            
        # Initialize LMDB environment
        self.env = lmdb.open(output_lmdb_path, 
                           max_readers=32,
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False)
        
        # Load dataset keys
        with open(f"{output_lmdb_path}/keys_best.pkl", 'rb') as f:
            self.keys = pickle.load(f)[phase]
            
        # Initialize augmentations
        self.init_augmentations()

    def init_face_models(self):
        """Initialize face detection, parsing, and landmark models"""
        self.face_detector = RetinaFacePredictor(
            threshold=0.8,
            device=self.device,
            model=RetinaFacePredictor.get_model('mobilenet0.25')
        )
        
        self.face_parser = RTNetPredictor(
            device=self.device,
            encoder='rtnet50',
            decoder='fcn',
            num_classes=14
        )
        
        # self.landmark_detector = FaceAlignment(
        #     device=self.device,
        #     model_type='2D'
        # )
        
        self.bfm = MorphabelModel(bfm_path)

    def init_dataset_params(self, *args):
        """Initialize dataset parameters"""
        (self.num_source_frames, self.num_target_frames, self.image_size,
         self.image_additional_size, self.augment_geometric, self.augment_color,
         self.output_aug_warp, self.output_aug_warp_out, self.use_masked_aug,
         self.aug_warp_size, self.warp_aug_color_coef, self.aug_color_coef,
         self.gray_source_prob, self.shift_limit, self.scale_cond_tr,
         self.rot_aug_angle, self.rand_crop_prob, self.rand_crop_scale) = args
        
        self.image_additional_size = self.image_additional_size or self.image_size

    def init_augmentations(self):
        """Initialize augmentation pipelines"""
        if self.augment_color:
            self.color_aug = A.Compose([
                A.ColorJitter(
                    brightness=0.06 * max(1, self.aug_color_coef/2),
                    contrast=0.03 * self.aug_color_coef,
                    saturation=0.03 * self.aug_color_coef,
                    hue=0.03 * self.aug_color_coef,
                    p=0.8
                ),
                A.ToGray(p=self.gray_source_prob)
            ])
            
            self.rot_aug = A.Compose([
                A.Rotate(limit=self.rot_aug_angle, value=0)
            ], additional_targets={'mask': 'image', 'mask1': 'image'})
            
            self.rand_crop = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=self.shift_limit,
                    scale_limit=0.0,
                    rotate_limit=0,
                    p=1.0
                ),
                A.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=(self.rand_crop_scale, 1.0),
                    ratio=(1, 1),
                    p=self.rand_crop_prob
                )
            ], additional_targets={'mask': 'image', 'mask1': 'image'})
            
            self.flip = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
               additional_targets={'image1': 'image', 'mask': 'image', 'mask1': 'image',
                                 'keypoints': 'keypoints', 'keypoints1': 'keypoints'})
                                 
        self.to_tensor = transforms.ToTensor()
        
        if self.output_aug_warp:
            grid = torch.linspace(0, 1, self.aug_warp_size)
            v, u = torch.meshgrid(grid, grid)
            self.grid = torch.stack([u, v, torch.zeros_like(u)], dim=2)
            self.grid = (self.grid * 255).numpy().astype('uint8')

    def preprocess_videos(self, json_path):
        """Preprocess videos and create LMDB database"""
        os.makedirs(self.output_lmdb_path, exist_ok=True)
        env = lmdb.open(self.output_lmdb_path, map_size=1024**4)
        
        with open(json_path) as f:
            metadata = json.load(f)
            
        keys_dict = {'train': [], 'test': []}
        
        for clip_id, clip_info in metadata['clips'].items():
            video_path = os.path.join(self.data_root, f"{clip_info['ytb_id']}.mp4")
            if not os.path.exists(video_path):
                continue
                
            # Process video frames in batches
            processed_frames = self.process_video_frames(
                video_path, clip_info, self.preprocess_batch_size
            )
            
            if not processed_frames:
                continue
                
            # Store processed frames in LMDB
            with env.begin(write=True) as txn:
                for i, frame_data in enumerate(processed_frames):
                    key = f"{clip_id}_{i:06d}".encode()
                    txn.put(key, pickle.dumps(frame_data))
                    
                    # Split into train/test (80/20)
                    if random.random() < 0.8:
                        keys_dict['train'].append([key])
                    else:
                        keys_dict['test'].append([key])
        
        # Save keys
        with open(os.path.join(self.output_lmdb_path, 'keys_best.pkl'), 'wb') as f:
            pickle.dump(keys_dict, f)
            
        env.close()

    def process_video_frames(self, video_path, clip_info, batch_size):
        """Process frames from a video in batches"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(clip_info['duration']['start_sec'] * fps)
        end_frame = int(clip_info['duration']['end_sec'] * fps)
        
        frame_indices = np.linspace(start_frame, end_frame, 
                                  self.num_source_frames + self.num_target_frames,
                                  dtype=int)
        
        processed_frames = []
        current_batch = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            current_batch.append((frame, frame_idx))
            
            if len(current_batch) == batch_size:
                batch_results = self.process_frame_batch(current_batch)
                processed_frames.extend(batch_results)
                current_batch = []
        
        if current_batch:
            batch_results = self.process_frame_batch(current_batch)
            processed_frames.extend(batch_results)
            
        cap.release()
        return processed_frames

    def process_frame_batch(self, frame_batch):
        """Process a batch of frames"""
        processed_frames = []
        
        for frame, frame_idx in frame_batch:
            # Face detection
            faces = self.face_detector(frame, rgb=True)
            if len(faces) == 0:
                continue
            
            # Take largest face
            face_areas = [(f[2]-f[0])*(f[3]-f[1]) for f in faces]
            face = faces[np.argmax(face_areas)]
            
            # Process frame
            frame_data = self.process_single_frame(frame, face)
            if frame_data is not None:
                processed_frames.append(frame_data)
                
        return processed_frames

    def process_single_frame(self, frame, face):
        """Process a single frame"""
        x1, y1, x2, y2 = map(int, face[:4])
        bbox = np.array([x1, x2, y1, y2])
        
        # Get landmarks
        landmarks_2d = self.landmark_detector(frame, face)
        
        # Estimate 3DMM parameters
        landmarks_3d = self.estimate_3dmm(frame, landmarks_2d)
        
        # Face parsing
        face_parsing = self.face_parser.predict_img(
            torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(self.device),
            torch.from_numpy(face).unsqueeze(0).to(self.device)
        )
        mask = face_parsing[0]
        
        # Align face
        face_scale = max((x2-x1)/frame.shape[1], (y2-y1)/frame.shape[0])
        aligned_face, aligned_mask, transform_matrix = self.align_face(
            frame, mask, landmarks_2d
        )
        
        # Get FFHQ parameters
        theta = self.get_ffhq_params(transform_matrix)
        
        return {
            'image': self.pil_to_bytes(Image.fromarray(aligned_face)),
            'mask': self.pil_to_bytes(Image.fromarray(aligned_mask)),
            'size': self.image_size,
            'face_scale': face_scale,
            'keypoints_3d': landmarks_3d,
            '3dmm': {
                'param': self.estimate_3dmm_params(landmarks_2d),
                'bbox': bbox
            },
            'transform_ffhq': {
                'theta': theta
            }
        }

    def __getitem__(self, index):
        """Get a data sample"""
        frames_data = self.load_video_sequence(self.keys[index])
        
        source_frames = frames_data[:self.num_source_frames]
        target_frames = frames_data[-self.num_target_frames:]
        
        # Process frames
        source_data = [self.process_frame_data(frame) for frame in source_frames]
        target_data = [self.process_frame_data(frame) for frame in target_frames]
        
        # Apply augmentations and create output dict
        return self.create_output_dict(source_data, target_data)

    def __len__(self):
        return len(self.keys)

    # Utility methods from CelebVHQPreprocessor
    def estimate_3dmm(self, image, landmarks_2d):
        vertices, landmarks_3d, alpha_shp, alpha_exp = self.bfm.fit(landmarks_2d, image)
        return landmarks_3d

    def estimate_3dmm_params(self, landmarks_2d):
        return self.bfm.estimate_parameters(landmarks_2d, return_pose=True)

    def align_face(self, image, mask, landmarks, output_size=256):
        src_points = landmarks[[36, 45, 48, 54]]
        dst_points = np.array([
            [0.3 * output_size, 0.3 * output_size],
            [0.7 * output_size, 0.3 * output_size],
            [0.3 * output_size, 0.7 * output_size],
            [0.7 * output_size, 0.7 * output_size]
        ])
        
        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
        
        aligned_face = cv2.warpAffine(image, transform_matrix, (output_size, output_size))
        aligned_mask = cv2.warpAffine(mask, transform_matrix, (output_size, output_size))
        
        return aligned_face, aligned_mask, transform_matrix

    def get_ffhq_params(self, transform_matrix):
        """Convert alignment matrix to FFHQ theta parameters"""
        scale_x = np.sqrt(transform_matrix[0,0]**2 + transform_matrix[0,1]**2)
        scale_y = np.sqrt(transform_matrix[1,0]**2 + transform_matrix[1,1]**2)
        theta = np.arctan2(transform_matrix[0,1], transform_matrix[0,0])
        
        tx = transform_matrix[0,2]
        ty = transform_matrix[1,2]
        
        theta_matrix = np.array([
            [scale_x * np.cos(theta), -scale_x * np.sin(theta), tx],
            [scale_y * np.sin(theta), scale_y * np.cos(theta), ty]
        ])
        
        return theta_matrix

    @staticmethod
    def pil_to_bytes(pil_img):
        """Convert PIL image to bytes for LMDB storage"""
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    def load_video_sequence(self, video_keys):
        """Load a sequence of frames from a video"""
        with self.env.begin(write=False) as txn:
            # Ensure we have enough frames
            if len(video_keys) < (self.num_source_frames + self.num_target_frames):
                video_keys = random.choice(self.keys)
            
            # Sample frames
            if len(video_keys) == (self.num_source_frames + self.num_target_frames):
                selected_keys = video_keys
            else:
                total_frames = self.num_source_frames + self.num_target_frames
                selected_indices = sorted(random.sample(range(len(video_keys)), total_frames))
                selected_keys = [video_keys[i] for i in selected_indices]
            
            # Load frame data
            frames_data = []
            for key in selected_keys:
                frame_data = pickle.loads(txn.get(key))
                frames_data.append(frame_data)
                
            return frames_data

    def process_frame_data(self, frame_data):
        """Process individual frame data"""
        # Load image and mask
        image = Image.open(io.BytesIO(frame_data['image'])).convert('RGB')
        mask = Image.open(io.BytesIO(frame_data['mask']))
        
        # Convert to tensors
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        
        # Get other parameters
        keypoints = torch.tensor(frame_data['keypoints_3d']).float()
        size = frame_data['size']
        face_scale = frame_data['face_scale']
        
        return {
            'image': image,
            'mask': mask,
            'keypoints': keypoints,
            'size': size,
            'face_scale': face_scale,
            'params_3dmm': frame_data['3dmm'],
            'params_ffhq': frame_data['transform_ffhq']
        }

    def apply_augmentations(self, source_data, target_data):
        """Apply augmentations to source and target data"""
        if self.augment_color:
            # Color augmentations
            augmented = self.color_aug(image=source_data['image'].permute(1,2,0).numpy())
            source_data['image'] = torch.from_numpy(augmented['image']).permute(2,0,1)
            
            if self.rot_aug_angle > 0:
                augmented = self.rot_aug(
                    image=source_data['image'].permute(1,2,0).numpy(),
                    image1=target_data['image'].permute(1,2,0).numpy(),
                    mask=source_data['mask'].permute(1,2,0).numpy(),
                    mask1=target_data['mask'].permute(1,2,0).numpy()
                )
                source_data['image'] = torch.from_numpy(augmented['image']).permute(2,0,1)
                target_data['image'] = torch.from_numpy(augmented['image1']).permute(2,0,1)
                source_data['mask'] = torch.from_numpy(augmented['mask']).permute(2,0,1)
                target_data['mask'] = torch.from_numpy(augmented['mask1']).permute(2,0,1)
            
            # Apply random crop if conditions met
            if (self.rand_crop_prob > 0.0 or self.rand_crop_scale < 1.0) and \
               (source_data['face_scale'] >= self.scale_cond_tr):
                augmented = self.rand_crop(
                    image=source_data['image'].permute(1,2,0).numpy(),
                    image1=target_data['image'].permute(1,2,0).numpy(),
                    mask=source_data['mask'].permute(1,2,0).numpy(),
                    mask1=target_data['mask'].permute(1,2,0).numpy()
                )
                source_data['image'] = torch.from_numpy(augmented['image']).permute(2,0,1)
                target_data['image'] = torch.from_numpy(augmented['image1']).permute(2,0,1)
                source_data['mask'] = torch.from_numpy(augmented['mask']).permute(2,0,1)
                target_data['mask'] = torch.from_numpy(augmented['mask1']).permute(2,0,1)
            
            # Apply flip augmentation with keypoint handling
            if random.random() < 0.5:
                augmented = self.flip(
                    image=source_data['image'].permute(1,2,0).numpy(),
                    image1=target_data['image'].permute(1,2,0).numpy(),
                    mask=source_data['mask'].permute(1,2,0).numpy(),
                    mask1=target_data['mask'].permute(1,2,0).numpy(),
                    keypoints=source_data['keypoints'].numpy(),
                    keypoints1=target_data['keypoints'].numpy()
                )
                source_data['image'] = torch.from_numpy(augmented['image']).permute(2,0,1)
                target_data['image'] = torch.from_numpy(augmented['image1']).permute(2,0,1)
                source_data['mask'] = torch.from_numpy(augmented['mask']).permute(2,0,1)
                target_data['mask'] = torch.from_numpy(augmented['mask1']).permute(2,0,1)
                source_data['keypoints'] = torch.from_numpy(augmented['keypoints'])
                target_data['keypoints'] = torch.from_numpy(augmented['keypoints1'])

        return source_data, target_data

    def create_output_dict(self, source_data, target_data):
        """Create the final output dictionary"""
        # Combine source and target data
        source_data = {
            k: torch.stack([d[k] for d in source_data]) 
            for k in source_data[0].keys()
        }
        target_data = {
            k: torch.stack([d[k] for d in target_data])
            for k in target_data[0].keys()
        }
        
        # Apply augmentations
        source_data, target_data = self.apply_augmentations(source_data, target_data)
        
        output_dict = {
            'source_img': source_data['image'],
            'source_mask': source_data['mask'],
            'source_keypoints': source_data['keypoints'],
            'source_params_3dmm': {
                'R': torch.tensor(source_data['params_3dmm']['param']).float(),
                'offset': source_data['params_3dmm']['bbox'],
                'roi_box': source_data['size'],
                'crop_box': torch.zeros(4)
            },
            'source_params_ffhq': {
                'theta': torch.tensor(source_data['params_ffhq']['theta']).float()
            },
            'target_img': target_data['image'],
            'target_mask': target_data['mask'],
            'target_keypoints': target_data['keypoints'],
            'target_params_3dmm': {
                'R': torch.tensor(target_data['params_3dmm']['param']).float(),
                'offset': target_data['params_3dmm']['bbox'],
                'roi_box': target_data['size'],
                'crop_box': torch.zeros(4)
            },
            'target_params_ffhq': {
                'theta': torch.tensor(target_data['params_ffhq']['theta']).float()
            }
        }

        # Add warping augmentations if required
        if self.output_aug_warp and self.output_aug_warp_out:
            source_warp = self.augment_via_warp(source_data['image'])
            target_warp = self.augment_via_warp(target_data['image'])
            
            if self.use_masked_aug:
                source_warp = self.apply_masked_augmentation(source_warp, source_data['mask'])
                target_warp = self.apply_masked_augmentation(target_warp, target_data['mask'])
            
            output_dict['source_warp_aug'] = source_warp
            output_dict['target_warp_aug'] = target_warp
        
        return output_dict

    def augment_via_warp(self, images):
        """Apply elastic deformation warping"""
        if not isinstance(images, list):
            images = [images]
        
        image_aug = []
        for image in images:
            if torch.is_tensor(image):
                image = image.permute(1, 2, 0).numpy()
            
            cell_count = 8 + 1
            cell_size = self.aug_warp_size // (cell_count - 1)
            
            grid_points = np.linspace(0, self.aug_warp_size, cell_count)
            mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
            mapy = mapx.T
            
            mapx[1:-1, 1:-1] += np.random.normal(size=(cell_count-2, cell_count-2)) * cell_size * 0.1
            mapy[1:-1, 1:-1] += np.random.normal(size=(cell_count-2, cell_count-2)) * cell_size * 0.1
            
            half_cell_size = cell_size // 2
            
            mapx = cv2.resize(mapx, (self.aug_warp_size + cell_size,) * 2)[
                   half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size
            ].astype(np.float32)
            
            mapy = cv2.resize(mapy, (self.aug_warp_size + cell_size,) * 2)[
                   half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size
            ].astype(np.float32)
            
            warped_image = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)
            warped_tensor = torch.from_numpy(warped_image).permute(2, 0, 1)
            image_aug.append(warped_tensor)
        
        return image_aug[0] if len(image_aug) == 1 else torch.stack(image_aug)

    def apply_masked_augmentation(self, image, mask):
        """Apply augmentation only to masked regions"""
        if self.use_masked_aug:
            mask_expanded = mask > 0.9
            masked_image = image * mask_expanded
            return masked_image
        return image

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for handling variable length sequences"""
        batch_size = len(batch)
        max_source_len = max([b['source_img'].size(0) for b in batch])
        max_target_len = max([b['target_img'].size(0) for b in batch])
        
        output_dict = {
            'source_img': [],
            'source_mask': [],
            'source_keypoints': [],
            'target_img': [],
            'target_mask': [],
            'target_keypoints': [],
            'source_params_3dmm': {
                'R': [],
                'offset': [],
                'roi_box': [],
                'crop_box': []
            },
            'target_params_3dmm': {
                'R': [],
                'offset': [],
                'roi_box': [],
                'crop_box': []
            },
            'source_params_ffhq': {
                'theta': []
            },
            'target_params_ffhq': {
                'theta': []
            }
        }
        
        # Pad sequences and combine batch
        for b in batch:
            # Pad source sequences
            pad_size = max_source_len - b['source_img'].size(0)
            if pad_size > 0:
                b['source_img'] = torch.cat([b['source_img'], 
                    torch.zeros(pad_size, *b['source_img'].shape[1:])], dim=0)
                b['source_mask'] = torch.cat([b['source_mask'],
                    torch.zeros(pad_size, *b['source_mask'].shape[1:])], dim=0)
                b['source_keypoints'] = torch.cat([b['source_keypoints'],
                    torch.zeros(pad_size, *b['source_keypoints'].shape[1:])], dim=0)
            
            # Pad target sequences
            pad_size = max_target_len - b['target_img'].size(0)
            if pad_size > 0:
                b['target_img'] = torch.cat([b['target_img'],
                    torch.zeros(pad_size, *b['target_img'].shape[1:])], dim=0)
                b['target_mask'] = torch.cat([b['target_mask'],
                    torch.zeros(pad_size, *b['target_mask'].shape[1:])], dim=0)
                b['target_keypoints'] = torch.cat([b['target_keypoints'],
                    torch.zeros(pad_size, *b['target_keypoints'].shape[1:])], dim=0)
            
            # Add to output dictionary
            for k in output_dict.keys():
                if k not in ['source_params_3dmm', 'target_params_3dmm', 
                           'source_params_ffhq', 'target_params_ffhq']:
                    output_dict[k].append(b[k])
            
            # Handle nested parameters
            for param_type in ['source_params_3dmm', 'target_params_3dmm',
                             'source_params_ffhq', 'target_params_ffhq']:
                for k in output_dict[param_type].keys():
                    output_dict[param_type][k].append(b[param_type][k])
        
        # Stack all tensors
        for k in output_dict.keys():
            if k not in ['source_params_3dmm', 'target_params_3dmm',
                        'source_params_ffhq', 'target_params_ffhq']:
                output_dict[k] = torch.stack(output_dict[k])
        
        # Stack nested parameters
        for param_type in ['source_params_3dmm', 'target_params_3dmm',
                          'source_params_ffhq', 'target_params_ffhq']:
            for k in output_dict[param_type].keys():
                output_dict[param_type][k] = torch.stack(output_dict[param_type][k])
        
        return output_dict

    @staticmethod
    def prepare_keypoints(keypoints, crop_box, size):
        """Prepare keypoints based on crop box and normalization"""
        keypoints = keypoints.clone()
        size_box = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
        
        # Normalize xy coordinates
        keypoints[..., 0] = (keypoints[..., 0] - crop_box[0]) / size_box[0] - 0.5
        keypoints[..., 1] = (keypoints[..., 1] - crop_box[1]) / size_box[1] - 0.5
        
        # Normalize z coordinate
        keypoints[..., 2] = keypoints[..., 2] / (size_box[0] + size_box[1]) * 2
        
        # Scale to [-1, 1]
        keypoints *= 2
        return keypoints

# Helper class for data loading
class DataModule:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.data_root = args.data_root
        self.json_path = args.json_path
        self.output_lmdb_path = args.output_lmdb_path
        self.device = args.device
        self.bfm_path = args.bfm_path 
        # Dataset parameters
        self.dataset_params = {
            'num_source_frames': args.num_source_frames,
            'num_target_frames': args.num_target_frames,
            'image_size': args.image_size,
            'image_additional_size': args.image_additional_size,
            'augment_geometric': args.augment_geometric_train,
            'augment_color': args.augment_color_train,
            'output_aug_warp': args.output_aug_warp,
            'output_aug_warp_out': args.output_aug_warp_out,
            'use_masked_aug': args.use_masked_aug,
            'aug_warp_size': args.aug_warp_size,
            'warp_aug_color_coef': args.warp_aug_color_coef,
            'aug_color_coef': args.aug_color_coef,
            'gray_source_prob': args.gray_source_prob,
            'shift_limit': args.shift_limit,
            'scale_cond_tr': args.scale_cond_tr,
            'rot_aug_angle': args.rot_aug_angle,
            'rand_crop_prob': args.rand_crop_prob,
            'rand_crop_scale': args.rand_crop_scale,
        }

    def train_dataloader(self):
        """Create training data loader"""
        train_dataset = CelebVHQDataset(
            data_root=self.data_root,
            json_path=self.json_path,
            output_lmdb_path=self.output_lmdb_path,
            phase='train',
            device=self.device,
            bfm_path=self.bfm_path
            **self.dataset_params
            
        )
        
        return data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=CelebVHQDataset.collate_fn
        )

    def val_dataloader(self):
        """Create validation data loader"""
        val_dataset = CelebVHQDataset(
            data_root=self.data_root,
            json_path=self.json_path,
            output_lmdb_path=self.output_lmdb_path,
            phase='test',
            device=self.device,

            **{**self.dataset_params, 
               'augment_geometric': False,
               'augment_color': False}
        )
        
        return data.DataLoader(
            val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=CelebVHQDataset.collate_fn
        )

def add_dataset_args(parser):
    """Add dataset-specific arguments to argument parser"""
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing video files')
                        
    parser.add_argument('--json_path', type=str, required=True, 
                        help='Path to metadata JSON file')
    parser.add_argument('--output_lmdb_path', type=str, required=True,
                        help='Output path for LMDB database')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='Testing batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_source_frames', type=int, default=1,
                        help='Number of source frames to sample')
    parser.add_argument('--num_target_frames', type=int, default=1,
                        help='Number of target frames to sample')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of output images')
    parser.add_argument('--image_additional_size', type=int, default=None,
                        help='Additional image size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for processing')
    
    # Augmentation parameters
    parser.add_argument('--augment_geometric_train', type=bool, default=False,
                        help='Use geometric augmentations during training')
    parser.add_argument('--augment_color_train', type=bool, default=True,
                        help='Use color augmentations during training')
    parser.add_argument('--output_aug_warp', type=bool, default=True,
                        help='Output warping augmentations')
    parser.add_argument('--output_aug_warp_out', type=bool, default=True,
                        help='Output warping augmentations in results')
    parser.add_argument('--use_masked_aug', type=bool, default=False,
                        help='Use masked augmentations')
    parser.add_argument('--aug_warp_size', type=int, default=256,
                        help='Size of warping augmentations')
    parser.add_argument('--warp_aug_color_coef', type=float, default=1.0,
                        help='Color coefficient for warp augmentations')
    parser.add_argument('--aug_color_coef', type=float, default=1.0,
                        help='Color augmentation coefficient')
    parser.add_argument('--gray_source_prob', type=float, default=0.0,
                        help='Probability of converting source to grayscale')
    parser.add_argument('--shift_limit', type=float, default=0.1,
                        help='Maximum shift in augmentations')
    parser.add_argument('--scale_cond_tr', type=float, default=0.8,
                        help='Scale condition threshold')
    parser.add_argument('--rot_aug_angle', type=float, default=0,
                        help='Maximum rotation angle for augmentation')
    parser.add_argument('--rand_crop_prob', type=float, default=0.0,
                        help='Random crop probability')
    parser.add_argument('--rand_crop_scale', type=float, default=0.9,
                        help='Random crop scale factor')
    
    return parser


