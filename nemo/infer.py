import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import pathlib
import numpy as np
import importlib
import math
from scipy import linalg
import apex
import sys
sys.path.append('/media/oem/12TB/EMOPortraits/')
import utils.args as args_utils
from utils import spectral_norm, stats_calc
from datasets.voxceleb2hq_pairs import LMDBDataset
from repos.MODNet.src.models.modnet import MODNet
from networks import volumetric_avatar
from torch.nn.modules.module import _addindent
import mediapipe as mp
from facenet_pytorch import MTCNN
import pickle
from utils import point_transforms
# from bilayer_model.external.Graphonomy.wrapper import SegmentationWrapper
import contextlib
none_context = contextlib.nullcontext()
from PIL import Image
import cv2
import PIL
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
import time 
from typing import Optional, List, Dict, Union


to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1) 
to_512 = lambda x: x.resize((512, 512), Image.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.LANCZOS)
mp_face_detection = mp.solutions.face_detection


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


class InferenceWrapper(nn.Module):
    def __init__(self, experiment_name, which_epoch='latest', model_file_name='', use_gpu=True, num_gpus=1,
                 fixed_bounding_box=False, project_dir='./', folder= 'mp_logs', model_ = 'va',
                 torch_home='', debug=False, print_model=False, print_params=True, args_overwrite={}, state_dict=None, pose_momentum=0.5, rank=0, args_path=None):
        super(InferenceWrapper, self).__init__()
        self.use_gpu = use_gpu
        self.debug = debug
        self.num_gpus = num_gpus

        self.modnet_pass = '/media/2TB/EMOPortraits/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'

        # Get a config for the network
        # args_path = pathlib.Path(project_dir) / folder / experiment_name / 'args.txt' if args_path is None else args_path
        self.args = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')

        # self.args = OmegaConf.load("config.yaml")
        # Add args from args_overwrite dict that overwrite the default ones
        self.args.project_dir = project_dir
        if args_overwrite is not None:
            for k, v in args_overwrite.items():
                setattr(self.args, k, v)
        self.face_idt = volumetric_avatar.FaceParsing(None, 'cuda')
#         self.graph = SegmentationWrapper()

        if torch_home:
            os.environ['TORCH_HOME'] = torch_home

        if self.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.random_seed)

        self.check_grads = self.args.check_grads_of_every_loss
        self.print_model = print_model
        # Set distributed training options
        if self.num_gpus <= 1:
            self.rank = 0

        elif self.num_gpus > 1 and self.num_gpus <= 8:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)

        elif self.num_gpus > 8:
            raise

        #print(self.args)
        # Initialize model

        self.model = importlib.import_module(f'models.stage_1.volumetric_avatar.va').Model(self.args, training=False)

        if rank == 0 and print_params:
            for n, p in self.model.net_param_dict.items():
                print(f'Number of perameters in {n}: {p}')
        #         self.model = importlib.import_module(f'models.__volumetric_avatar').Model(self.args, training=False)
        if self.use_gpu:
            self.model.cuda()

        if self.rank == 0 and self.print_model:
            print(self.model)
            # ms = torch_summarize(self.model)

        # Load pre-trained weights
        self.model_checkpoint = pathlib.Path(project_dir) / folder / experiment_name / 'checkpoints' / model_file_name
        # print(self.model_checkpoint, args_path)
        if self.args.model_checkpoint:
            if self.rank == 0:
                # print(f'Loading model from {self.model_checkpoint}')
                pass
            self.model_dict = torch.load(self.model_checkpoint, map_location='cpu') if state_dict==None else state_dict
            self.model.load_state_dict(self.model_dict, strict=False)

        # Initialize distributed training
        if self.num_gpus > 1:
            self.model = apex.parallel.convert_syncbn_model(self.model)
            self.model = apex.parallel.DistributedDataParallel(self.model)

        self.model.eval()

        self.modnet = MODNet(backbone_pretrained=False)

        if self.num_gpus > 0:
            self.modnet = nn.DataParallel(self.modnet).cuda()

        if self.use_gpu:
            self.modnet = self.modnet.cuda()

        self.modnet.load_state_dict(torch.load(self.modnet_pass, map_location='cpu'))
        self.modnet.eval()

        # Face detection is required as pre-processing
        device = 'cuda' if use_gpu else 'cpu'
        self.device = device
        face_detector = 'sfd'
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=False)

        # Face tracking and bounding box smoothing parameters
        self.fixed_bounding_box = fixed_bounding_box  # no tracking is performed, first bounding box in driver is used for all frames
        self.momentum = 0.01  # if bounding box is not fixed, it is updated with momentum
        self.center = None
        self.size = None

        # Head pose smoother
        self.pose_momentum = pose_momentum
        self.theta = None

        # Head normalization params
        self.norm_momentum = 0.1
        self.delta_yaw = None
        self.delta_pitch = None

        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()
        self.resize_warp = self.args.warp_output_size != self.args.gen_latent_texture_size
        self.use_seg = self.args.use_seg

    @torch.no_grad()
    def calculate_standing_stats(self, data_root, num_iters):
        self.identity_embedder.train().apply(stats_calc.stats_calculation)
        self.pose_embedder.train().apply(stats_calc.stats_calculation)
        self.generator.train().apply(stats_calc.stats_calculation)

        # Initialize train dataset
        dataset = LMDBDataset(
            data_root,
            'train',
            self.args.num_source_frames,
            self.args.num_target_frames,
            self.args.image_size,
            False)

        dataset.names = dataset.names[:self.args.batch_size * num_iters]

        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=self.args.num_workers_per_process)

        for data_dict in dataloader:
            source_img_crop = data_dict['source_img']
            driver_img_crop = data_dict['target_img']

            source_img_crop = source_img_crop.view(-1, *source_img_crop.shape[2:])
            driver_img_crop = driver_img_crop.view(-1, *driver_img_crop.shape[2:])

            if self.use_gpu:
                source_img_crop = source_img_crop.cuda()
                driver_img_crop = driver_img_crop.cuda()

            idt_embed = self.identity_embedder.forward_image(source_img_crop)

            # During training, pose embedder accepts concatenated data, so we need to imitate it during stats calculation
            img_crop = torch.cat([source_img_crop, driver_img_crop])
            pose_embed, pred_theta = self.pose_embedder.forward_image(img_crop)

            source_pose_embed, driver_pose_embed = pose_embed.split(
                [source_img_crop.shape[0], driver_img_crop.shape[0]])
            pred_source_theta, pred_driver_theta = pred_theta.split(
                [source_img_crop.shape[0], driver_img_crop.shape[0]])

            latent_texture, embed_dict = self.generator.forward_source(source_img_crop, idt_embed, source_pose_embed,
                                                                       pred_source_theta)
            pred_target_img = self.generator.forward_driver(idt_embed, driver_pose_embed, embed_dict, pred_source_theta,
                                                            pred_driver_theta, latent_texture)

    def convert_to_tensor(self, image):
        if isinstance(image, list):
            image_tensor = [self.to_tensor(img) for img in image]
            image_tensor = torch.stack(image_tensor)  # all images have to be the same size
        else:
            image_tensor = self.to_tensor(image)

        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor[None]

        if self.use_gpu:
            image_tensor = image_tensor.cuda()

        return image_tensor

    @staticmethod
    def remove_overflow(center, size, w, h):
        bbox = np.asarray([center[0] - size / 2, center[1] - size / 2, center[0] + size / 2, center[1] + size / 2])

        shift_l = 0 if bbox[0] >= 0 else -bbox[0]
        shift_u = 0 if bbox[1] >= 0 else -bbox[1]
        shift_r = 0 if bbox[2] <= w else bbox[2] - w
        shift_d = 0 if bbox[3] <= h else bbox[3] - h

        shift = max(shift_l, shift_u, shift_r, shift_d)

        bbox[[0, 1]] += shift
        bbox[[2, 3]] -= shift

        center = np.asarray([bbox[[0, 2]].mean(), bbox[[1, 3]].mean()]).astype(int)
        size_overflow = int((bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2)
        size_overflow = size_overflow - size_overflow % 2

        return size_overflow


    def crop_image(self, image, faces, use_smoothed_crop=False, scale=1):
        imgs_crop = []
        face_check = np.ones(len(image), dtype=bool)
        face_scale_stats = []

        for b, face in enumerate(faces):

            if face is None:
                face_check[b] = False
                imgs_crop.append(torch.zeros((1, 3, self.args.image_size, self.args.image_size)))
                face_scale_stats.append(0)
                continue

            center = np.asarray([(face[2] + face[0]) // 2, (face[3] + face[1]) // 2])
            size = (face[2] - face[0] + face[3] - face[1])*scale

            if use_smoothed_crop:
                if self.center is None:
                    self.center = center
                    self.size = size

                elif not self.fixed_bounding_box:
                    self.center = center * self.momentum + self.center * (1 - self.momentum)
                    self.size = size * self.momentum + self.size * (1 - self.momentum)

                center = self.center
                size = self.size

            center = center.round().astype(int)
            size = int(round(size))
            size = size - size % 2

            if isinstance(image, list):
                size_overflow = self.remove_overflow(center, size, image[b].shape[2], image[b].shape[1])
                face_scale = size_overflow / size
                size = size_overflow
                img_crop = image[b][:, center[1] - size // 2: center[1] + size // 2,
                           center[0] - size // 2: center[0] + size // 2]
            else:
                size_overflow = self.remove_overflow(center, size, image.shape[3], image.shape[2])
                face_scale = size_overflow / size
                size = size_overflow
                img_crop = image[b, :, center[1] - size // 2: center[1] + size // 2,
                           center[0] - size // 2: center[0] + size // 2]

            img_crop = F.interpolate(img_crop[None], size=(self.args.image_size, self.args.image_size), mode='bicubic')
            imgs_crop.append(img_crop)
            face_scale_stats.append(face_scale)

        imgs_crop = torch.cat(imgs_crop).clip(0, 1)

        return imgs_crop, face_check, face_scale_stats


    def forward(self,
                source_image: Optional[PIL.Image.Image] = None,
                driver_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
                source_mask: Optional[torch.Tensor] = None,
                source_mask_add: int = 0,
                driver_mask: Optional[torch.Tensor] = None,
                crop: bool = True,
                reset_tracking: bool = False,
                smooth_pose: bool = False,
                hard_normalize: bool = False,
                soft_normalize: bool = False,
                delta_yaw: Optional[float] = None,
                delta_pitch: Optional[float] = None,
                cloth: bool = False,
                thetas_pass: str = '',
                theta_n: int = 0,
                target_theta: bool = True,
                mix: bool = False,
                mix_old: bool = True,
                c_source_latent_volume: Optional[torch.Tensor] = None,
                c_target_latent_volume: Optional[torch.Tensor] = None,
                custome_target_pose_embed: Optional[torch.Tensor] = None,
                custome_target_theta_embed: Optional[torch.Tensor] = None,
                no_grad_infer: bool = True,
                modnet_mask: bool = False) -> Tuple[Optional[List[PIL.Image.Image]], Optional[torch.Tensor]]:
        """
        Forward pass of the inference wrapper.
        
        Args: [see param descriptions below function]
        
        Returns:
            Tuple containing:
            - List of generated images (or None)
            - Generated image tensor (or None)
        """
        self._initialize_inference_state(
            no_grad_infer, target_theta, mix, mix_old, reset_tracking,
            delta_yaw, delta_pitch
        )
        
        # Process source image if provided
        if source_image is not None:
            source_result = self._process_source_image(
                source_image=source_image,
                crop=crop,
                source_mask=source_mask,
                source_mask_add=source_mask_add,
                modnet_mask=modnet_mask,
                c_source_latent_volume=c_source_latent_volume,
                c_target_latent_volume=c_target_latent_volume
            )
            if source_result is None:
                return None, None
        
        # Process driver image if provided
        if driver_image is not None:
            return self._process_driver_image(
                driver_image=driver_image,
                crop=crop,
                driver_mask=driver_mask,
                smooth_pose=smooth_pose,
                custome_target_pose_embed=custome_target_pose_embed,
                custome_target_theta_embed=custome_target_theta_embed
            )
            
        return None, None

    def _initialize_inference_state(self, no_grad_infer: bool, target_theta: bool,
                                  mix: bool, mix_old: bool, reset_tracking: bool,
                                  delta_yaw: Optional[float], delta_pitch: Optional[float]) -> None:
        """Initialize inference state variables."""
        self.no_grad_infer = no_grad_infer
        self.target_theta = target_theta
        self.mix = mix
        self.mix_old = mix_old
        
        if reset_tracking:
            self.center = None
            self.size = None
            self.theta = None
            self.delta_yaw = None
            self.delta_pitch = None
            
        if delta_yaw is not None:
            self.delta_yaw = delta_yaw
        if delta_pitch is not None:
            self.delta_pitch = delta_pitch

    def _process_source_image(self, source_image: PIL.Image.Image, crop: bool,
                            source_mask: Optional[torch.Tensor],
                            source_mask_add: int, modnet_mask: bool,
                            c_source_latent_volume: Optional[torch.Tensor],
                            c_target_latent_volume: Optional[torch.Tensor]) -> bool:
        """
        Process source identity image.
        
        Args:
            source_image: Source identity image
            crop: Whether to crop face
            source_mask: Optional mask for source image
            source_mask_add: Additional mask value
            modnet_mask: Use MODNet masking
            c_source_latent_volume: Custom source latent volume
            c_target_latent_volume: Custom target latent volume
            
        Returns:
            True if processing succeeded, False otherwise
        """
        with torch.no_grad():
            # Store original image
            self.source_image = self.convert_to_tensor(source_image)[:, :3]
            
            # Process image based on crop setting
            if crop:
                source_img_crop = self._crop_source_image(source_image)
            else:
                source_img_crop = self._prepare_uncropped_image(source_image)
            
            # Store cropped image and move to device
            self.source_image_crop = source_img_crop
            source_img_crop = source_img_crop.to(self.device)
            
            # Generate masks
            face_mask_source, source_mask_modnet = self._generate_source_masks(source_img_crop, modnet_mask)
            
            # Apply mask to source image
            source_img_crop = (source_img_crop * face_mask_source).float()
            self.source_img_crop_m = source_img_crop
            
            # Prepare final source mask
            source_img_mask = self._prepare_source_mask(
                source_mask=source_mask,
                face_mask=face_mask_source,
                modnet_mask=source_mask_modnet,
                use_modnet=modnet_mask,
                mask_add=source_mask_add
            )
            self.source_img_mask = source_img_mask
            
            # Process source features
            return self._process_source_features(
                source_img_crop=source_img_crop,
                source_img_mask=source_img_mask,
                c_source_latent_volume=c_source_latent_volume,
                c_target_latent_volume=c_target_latent_volume
            )

    def _prepare_source_mask(self, source_mask: Optional[torch.Tensor],
                           face_mask: torch.Tensor,
                           modnet_mask: torch.Tensor,
                           use_modnet: bool,
                           mask_add: int) -> torch.Tensor:
        """
        Prepare the final source mask combining different mask inputs.
        
        Args:
            source_mask: Optional external mask
            face_mask: Face parsing mask
            modnet_mask: MODNet generated mask
            use_modnet: Whether to use MODNet mask
            mask_add: Additional mask value
            
        Returns:
            Combined mask tensor
        """
        # Select base mask
        if source_mask is not None:
            final_mask = source_mask
        elif use_modnet:
            final_mask = modnet_mask
        else:
            final_mask = face_mask
            
        # Apply additional mask value if specified
        if mask_add:
            final_mask = final_mask.clamp_(min=0, max=1)
            
        return final_mask.to(self.device)

    def convert_to_tensor(self, image: Union[PIL.Image.Image, List[PIL.Image.Image]]) -> torch.Tensor:
        """
        Convert PIL image(s) to tensor.
        
        Args:
            image: Single PIL image or list of images
            
        Returns:
            Tensor of shape (B, C, H, W)
        """
        if isinstance(image, list):
            image_tensor = torch.stack([self.to_tensor(img) for img in image])
        else:
            image_tensor = self.to_tensor(image)[None]
            
        if self.use_gpu:
            image_tensor = image_tensor.cuda()
            
        return image_tensor

    def _process_latent_volumes(self, source_latents: torch.Tensor,
                              source_rotation_warp: torch.Tensor,
                              source_xy_warp: torch.Tensor,
                              embed_dict: Dict[str, torch.Tensor],
                              c: int, s: int, d: int,
                              c_source_latent_volume: Optional[torch.Tensor],
                              c_target_latent_volume: Optional[torch.Tensor]) -> None:
        """
        Process and store latent volumes.
        """
        # Reshape source latents
        source_latent_volume = source_latents.view(1, c, d, s, s)
        if self.args.source_volume_num_blocks > 0:
            source_latent_volume = self.model.volume_source_nw(source_latent_volume)
        
        # Store source volume
        self.source_latent_volume = (
            source_latent_volume if c_source_latent_volume is None 
            else c_source_latent_volume
        )
        
        # Store warping fields
        self.source_rotation_warp = source_rotation_warp
        source_xy_warp_resize = source_xy_warp
        if self.resize_warp:
            source_xy_warp_resize = self.model.resize_warp_func(source_xy_warp_resize)
        self.source_xy_warp_resize = source_xy_warp_resize
        
        # Generate target volumes
        target_latent_volume = self.model.grid_sample(
            self.model.grid_sample(self.source_latent_volume, source_rotation_warp),
            source_xy_warp_resize
        )
        
        # Store target volumes
        self.target_latent_volume_1 = (
            target_latent_volume if c_target_latent_volume is None 
            else c_target_latent_volume
        )
        self.target_latent_volume = self.model.volume_process_nw(
            self.target_latent_volume_1, embed_dict
        )
    def _prepare_uncropped_image(self, source_image: PIL.Image.Image) -> torch.Tensor:
        """
        Prepare source image without cropping.
        
        Args:
            source_image: Source identity image
            
        Returns:
            Preprocessed tensor of shape (1, 3, H, W)
        """
        source_tensor = self.convert_to_tensor(source_image)[:, :3]
        return F.interpolate(
            source_tensor,
            size=(self.args.image_size, self.args.image_size),
            mode='bicubic'
        )

    def _crop_source_image(self, source_image: PIL.Image.Image) -> torch.Tensor:
        """Crop source image using face detection."""
        source_faces = []
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            np_s = np.array(source_image)
            results = face_detection.process(np_s)
            
            if results.detections is None:
                source_faces.append(None)
            else:
                r = results.detections[0].location_data.relative_bounding_box
                source_faces.append(np.array([
                    source_image.size[0] * r.xmin,
                    source_image.size[1] * r.ymin * 0.9,
                    source_image.size[0] * (r.xmin + r.width),
                    min(source_image.size[1] * (r.ymin + r.height * 1.2), 
                        source_image.size[1] - 1)
                ]))
        
        source_img_crop, _, _ = self.crop_image(
            [self.to_tensor(source_image)], source_faces)
        return source_img_crop

    def _generate_source_masks(self, source_img_crop: torch.Tensor,
                             modnet_mask: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate face parsing and MODNet masks."""
        trashhold = 0.6
        face_mask_source, _, _, _ = self.model.face_idt.forward(source_img_crop)
        face_mask_source = (face_mask_source > trashhold).float()
        source_mask_modnet = self.get_mask(source_img_crop)
        return face_mask_source, source_mask_modnet


    def _process_source_features(self, source_img_crop: torch.Tensor,
                            source_img_mask: torch.Tensor,
                            c_source_latent_volume: Optional[torch.Tensor],
                            c_target_latent_volume: Optional[torch.Tensor]) -> bool:
        """
        Process source image features and embeddings.
        """
        try:
            # Get model dimensions
            c = self.args.latent_volume_channels
            s = self.args.latent_volume_size
            d = self.args.latent_volume_depth
            
            # Generate embeddings using model's embedder
            self.idt_embed = self.model.idt_embedder_nw.forward_image(
                source_img_crop * source_img_mask)
            
            # Generate source latents 
            source_latents = self.model.local_encoder_nw(
                source_img_crop * source_img_mask)
                
            # Get source pose
            with torch.no_grad():
                pred_source_theta = self.model.head_pose_regressor.forward(source_img_crop)
            self.pred_source_theta = pred_source_theta
            
            # Create transformation grid
            grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
            inv_source_theta = pred_source_theta.float().inverse().type(pred_source_theta.type())
            source_rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
            
            # Prepare data dictionary
            data_dict = {
                'source_img': source_img_crop,
                'source_mask': source_img_mask,
                'source_theta': pred_source_theta,
                'target_img': source_img_crop,
                'target_mask': source_img_mask,
                'target_theta': pred_source_theta,
                'idt_embed': self.idt_embed
            }
            
            # Process expression embedding
            data_dict = self.model.expression_embedder_nw(data_dict, True, False)
            source_pose_embed = data_dict['source_pose_embed']
            
            # Store important data
            self.pred_source_pose_embed = source_pose_embed
            self.source_img_align = data_dict.get('source_img_align')
            self.source_img = source_img_crop
            self.align_warp = data_dict.get('align_warp')
            
            # Generate warping
            source_warp_embed_dict, _, _, embed_dict = self.model.predict_embed(data_dict)
            xy_gen_outputs = self.model.xy_generator_nw(source_warp_embed_dict)
            data_dict['source_delta_xy'] = xy_gen_outputs[0]
            
            # Process source warping
            source_xy_warp = xy_gen_outputs[0]
            source_xy_warp_resize = source_xy_warp
            if self.resize_warp:
                source_xy_warp_resize = self.model.resize_warp_func(source_xy_warp_resize)
            
            # Process source volume
            source_latents_face = source_latents
            source_latent_volume = source_latents_face.view(1, c, d, s, s)
            
            if self.args.source_volume_num_blocks > 0:
                source_latent_volume = self.model.volume_source_nw(source_latent_volume)
            
            # Store processed volumes
            self.source_latent_volume = (source_latent_volume if c_source_latent_volume is None 
                                    else c_source_latent_volume)
            self.source_rotation_warp = source_rotation_warp
            self.source_xy_warp_resize = source_xy_warp_resize
            
            # Generate target volume
            target_latent_volume = self.model.grid_sample(
                self.model.grid_sample(self.source_latent_volume, source_rotation_warp),
                source_xy_warp_resize
            )
            
            # Store target volumes
            self.target_latent_volume_1 = (target_latent_volume if c_target_latent_volume is None 
                                        else c_target_latent_volume)
            self.target_latent_volume = self.model.volume_process_nw(
                self.target_latent_volume_1, embed_dict)
            
            return True
            
        except Exception as e:
            print(f"Error processing source features: {str(e)}")
            return False

    def _generate_target_features(self, driver_img_crop: torch.Tensor,
                                pred_target_theta: torch.Tensor,
                                driver_mask: Optional[torch.Tensor],
                                custome_target_pose_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Generate target features and final image.
        """
        # Set up transformation grid
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth
        
        grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
        
        # Apply target theta transformation
        if self.target_theta:
            target_rotation_warp = grid.bmm(
                pred_target_theta[:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
        else:
            target_rotation_warp = grid.bmm(
                self.pred_source_theta[:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
        
        # Get driver mask
        driver_img_mask = (driver_mask if driver_mask is not None 
                        else self.get_mask(driver_img_crop))
        driver_img_mask = driver_img_mask.to(driver_img_crop.device)
        
        # Prepare data dictionary
        data_dict = {
            'source_img': driver_img_crop,
            'source_mask': driver_img_mask,
            'source_theta': pred_target_theta,
            'target_img': driver_img_crop,
            'target_mask': driver_img_mask,
            'target_theta': pred_target_theta,
            'idt_embed': self.idt_embed
        }
        
        # Process expression embedding
        data_dict = self.model.expression_embedder_nw(data_dict, True, False)
        
        # Use custom pose embedding if provided
        if custome_target_pose_embed is not None:
            data_dict['target_pose_embed'] = custome_target_pose_embed
        
        # Store pose embedding and aligned image
        self.target_pose_embed = data_dict['target_pose_embed']
        self.target_img_align = data_dict.get('target_img_align')
        
        # Generate target features
        _, target_warp_embed_dict, _, embed_dict = self.model.predict_embed(data_dict)
        target_uv_warp, data_dict['target_delta_uv'] = self.model.uv_generator_nw(
            target_warp_embed_dict)
        
        # Apply resize if needed
        target_uv_warp_resize = target_uv_warp
        if self.resize_warp:
            target_uv_warp_resize = self.model.resize_warp_func(target_uv_warp_resize)
        
        # Generate aligned target volume
        aligned_target_volume = self.model.grid_sample(
            self.model.grid_sample(self.target_latent_volume, target_uv_warp_resize),
            target_rotation_warp
        )
        
        # Process final features
        target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
        
        # Generate final image
        img, _, _, _ = self.model.decoder_nw(
            data_dict,
            embed_dict,
            target_latent_feats,
            False,
            stage_two=True
        )
        
        return img

    def _process_source_volume(self, source_latents: torch.Tensor,
                             c: int, d: int, s: int) -> torch.Tensor:
        """
        Process source latents into volume.
        
        Args:
            source_latents: Source latent features
            c: Number of channels
            d: Depth dimension
            s: Spatial dimension
            
        Returns:
            Processed source volume
        """
        # Reshape source latents into volume
        source_latent_volume = source_latents.view(1, c, d, s, s)
        
        # Apply additional processing if configured
        if self.args.source_volume_num_blocks > 0:
            source_latent_volume = self.model.volume_source_nw(source_latent_volume)
            
        return source_latent_volume

    def _process_source_pose(self, source_img_crop: torch.Tensor,
                           source_img_mask: torch.Tensor,
                           source_latent_volume: torch.Tensor,
                           c_source_latent_volume: Optional[torch.Tensor],
                           c_target_latent_volume: Optional[torch.Tensor],
                           c: int, d: int, s: int) -> Dict[str, torch.Tensor]:
        """
        Process source pose and transformations.
        
        Args:
            source_img_crop: Cropped source image
            source_img_mask: Source image mask
            source_latent_volume: Source latent volume
            c_source_latent_volume: Optional custom source volume
            c_target_latent_volume: Optional custom target volume
            c: Number of channels
            d: Depth dimension
            s: Spatial dimension
            
        Returns:
            Dictionary containing processed data
        """
        # Get source pose
        with torch.no_grad():
            pred_source_theta = self.model.head_pose_regressor.forward(source_img_crop)
        self.pred_source_theta = pred_source_theta
        
        # Create transformation grid
        grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
        inv_source_theta = pred_source_theta.float().inverse().type(pred_source_theta.type())
        source_rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
        
        # Prepare data dictionary
        data_dict = {
            'source_img': source_img_crop,
            'source_mask': source_img_mask,
            'source_theta': pred_source_theta,
            'target_img': source_img_crop,
            'target_mask': source_img_mask,
            'target_theta': pred_source_theta,
            'idt_embed': self.idt_embed
        }
        
        # Process expression embedding
        data_dict = self.model.pred_source_pose_embed(data_dict, True, False)
        
        # Store important data
        self.pred_source_pose_embed = data_dict['source_pose_embed']
        self.source_img_align = data_dict['source_img_align']
        self.source_img = source_img_crop
        self.align_warp = data_dict['align_warp']
        
        # Process warping
        source_warp_embed_dict, _, _, embed_dict = self.model.predict_embed(data_dict)
        xy_gen_outputs = self.model.xy_generator_nw(source_warp_embed_dict)
        data_dict['source_delta_xy'] = xy_gen_outputs[0]
        
        # Process and store warping
        source_xy_warp = xy_gen_outputs[0]
        source_xy_warp_resize = source_xy_warp
        if self.resize_warp:
            source_xy_warp_resize = self.model.resize_warp_func(source_xy_warp_resize)
            
        # Store warping fields and volumes
        self.source_rotation_warp = source_rotation_warp
        self.source_xy_warp_resize = source_xy_warp_resize
        self.source_latent_volume = (source_latent_volume if c_source_latent_volume is None 
                                   else c_source_latent_volume)
        
        # Generate target volumes
        target_latent_volume = self.model.grid_sample(
            self.model.grid_sample(self.source_latent_volume, source_rotation_warp),
            source_xy_warp_resize
        )
        
        self.target_latent_volume_1 = (target_latent_volume if c_target_latent_volume is None 
                                     else c_target_latent_volume)
        self.target_latent_volume = self.model.volume_process_nw(
            self.target_latent_volume_1, embed_dict)
        
        return data_dict
    
    def _process_driver_image(self, driver_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                         crop: bool, driver_mask: Optional[torch.Tensor],
                         smooth_pose: bool,
                         custome_target_pose_embed: Optional[torch.Tensor],
                         custome_target_theta_embed: Optional[torch.Tensor]) -> Tuple[List[PIL.Image.Image], torch.Tensor]:
        """
        Process driver image to generate animated output.
        
        Args:
            driver_image: Single or batch of driver images
            crop: Whether to crop faces
            driver_mask: Optional mask for driver image
            smooth_pose: Whether to apply pose smoothing
            custome_target_pose_embed: Optional custom pose embedding
            custome_target_theta_embed: Optional custom theta embedding
            
        Returns:
            Tuple of (generated images list, generated image tensor)
        """
        context = torch.no_grad() if self.no_grad_infer else contextlib.nullcontext()
        with context:
            # Prepare driver image
            driver_img_crop = self._prepare_driver_images(driver_image, crop)
            driver_img_crop = driver_img_crop.to(self.device)
            
            # Process driver pose
            pred_target_theta = self._process_driver_pose(
                driver_img_crop, smooth_pose, custome_target_theta_embed)
            
            # Generate target features
            img = self._generate_target_features(
                driver_img_crop, pred_target_theta, driver_mask,
                custome_target_pose_embed
            )
            
            # Format output
            pred_target_img = img.detach().cpu().clamp(0, 1)
            pred_target_img = [self.to_image(img_) for img_ in pred_target_img]
            
            return pred_target_img, img

    def _prepare_driver_images(self, driver_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                            crop: bool) -> torch.Tensor:
        """
        Prepare driver images for processing.
        
        Args:
            driver_image: Single or batch of driver images
            crop: Whether to crop faces
            
        Returns:
            Processed image tensor
        """
        if crop:
            if not isinstance(driver_image, list):
                driver_image = [driver_image]
                
            # Detect faces in all images
            driver_faces = [self._detect_face(img) for img in driver_image]
            driver_tensors = [self.to_tensor(img) for img in driver_image]
            
            # Crop faces
            driver_img_crop, face_check, face_scale_stats = self.crop_image(
                driver_tensors, driver_faces)
        else:
            # Handle uncropped images
            if not isinstance(driver_image, list):
                driver_image = self.convert_to_tensor(driver_image)[:, :3]
            driver_img_crop = F.interpolate(
                driver_image,
                size=(self.args.image_size, self.args.image_size),
                mode='bicubic'
            )
        
        return driver_img_crop

    def _process_driver_pose(self, driver_img_crop: torch.Tensor,
                            smooth_pose: bool,
                            custome_target_theta_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Process driver pose and handle pose smoothing.
        
        Args:
            driver_img_crop: Processed driver image tensor
            smooth_pose: Whether to apply pose smoothing
            custome_target_theta_embed: Optional custom theta embedding
            
        Returns:
            Processed pose tensor
        """
        with torch.no_grad():
            pred_target_theta, scale, rotation, translation = self.model.head_pose_regressor.forward(
                driver_img_crop, return_srt=True)
        
        self.pred_target_theta = pred_target_theta
        self.pred_target_srt = (scale, rotation, translation)
        
        # Use custom theta if provided
        if custome_target_theta_embed is not None:
            pred_target_theta = point_transforms.get_transform_matrix(*custome_target_theta_embed)
        
        # Apply mixing if enabled
        if self.mix:
            pred_target_theta = self.get_mixing_theta(self.pred_source_theta, pred_target_theta)
        
        # Apply pose smoothing if enabled
        if smooth_pose:
            pred_target_theta = self._smooth_pose(pred_target_theta)
        
        return pred_target_theta

    def _smooth_pose(self, pred_target_theta: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to predicted pose.
        
        Args:
            pred_target_theta: Predicted pose tensor
            
        Returns:
            Smoothed pose tensor
        """
        if self.theta is None:
            self.theta = pred_target_theta[0].clone()
        
        smooth_driver_theta = []
        for i in range(pred_target_theta.shape[0]):
            self.theta = (pred_target_theta[i] * self.pose_momentum + 
                        self.theta * (1 - self.pose_momentum))
            smooth_driver_theta.append(self.theta.clone())
        
        return torch.stack(smooth_driver_theta)

    def _generate_target_features(self, driver_img_crop: torch.Tensor,
                                pred_target_theta: torch.Tensor,
                                driver_mask: Optional[torch.Tensor],
                                custome_target_pose_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Generate target features and final image.
        
        Args:
            driver_img_crop: Processed driver image
            pred_target_theta: Predicted pose
            driver_mask: Optional driver mask
            custome_target_pose_embed: Optional custom pose embedding
            
        Returns:
            Generated image tensor
        """
        # Set up transformation grid
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth
        
        grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
        
        # Apply target theta transformation
        if self.target_theta:
            target_rotation_warp = grid.bmm(
                pred_target_theta[:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
        else:
            target_rotation_warp = grid.bmm(
                self.pred_source_theta[:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
        
        # Generate or get driver mask
        driver_img_mask = (driver_mask if driver_mask is not None 
                        else self.get_mask(driver_img_crop))
        driver_img_mask = driver_img_mask.to(driver_img_crop.device)
        
        # Prepare data dictionary
        data_dict = {
            'source_img': driver_img_crop,
            'source_mask': driver_img_mask,
            'source_theta': pred_target_theta,
            'target_img': driver_img_crop,
            'target_mask': driver_img_mask,
            'target_theta': pred_target_theta,
            'idt_embed': self.idt_embed
        }
        
        # Process expression embedding
        data_dict = self.model.expression_embedder_nw(data_dict, True, False)
        
        # Use custom pose embedding if provided
        if custome_target_pose_embed is not None:
            data_dict['target_pose_embed'] = custome_target_pose_embed
        
        # Store pose embedding and aligned image
        self.target_pose_embed = data_dict['target_pose_embed']
        self.target_img_align = data_dict['target_img_align']
        
        # Generate target features
        _, target_warp_embed_dict, _, embed_dict = self.model.predict_embed(data_dict)
        target_uv_warp, data_dict['target_delta_uv'] = self.model.uv_generator_nw(
            target_warp_embed_dict)
        
        # Process warping
        target_uv_warp_resize = target_uv_warp
        if self.resize_warp:
            target_uv_warp_resize = self.model.resize_warp_func(target_uv_warp_resize)
        
        # Generate aligned target volume
        aligned_target_volume = self.model.grid_sample(
            self.model.grid_sample(self.target_latent_volume, target_uv_warp_resize),
            target_rotation_warp
        )
        
        # Process final features
        target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
        
        # Generate final image
        img, _, deep_f, img_f = self.model.decoder_nw(
            data_dict,
            embed_dict,
            target_latent_feats,
            False,
            stage_two=True
        )
        
        return img
    def _prepare_driver_image(self, driver_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                            crop: bool) -> torch.Tensor:
        """Prepare driver image for processing."""
        if crop:
            if not isinstance(driver_image, list):
                driver_image = [driver_image]
            
            driver_faces = self._detect_driver_faces(driver_image)
            driver_image = [self.to_tensor(img) for img in driver_image]
            driver_img_crop, face_check, face_scale_stats = self.crop_image(
                driver_image, driver_faces)
        else:
            if not isinstance(driver_image, list):
                driver_image = self.convert_to_tensor(driver_image)[:, :3]
            driver_img_crop = F.interpolate(
                driver_image,
                size=(self.args.image_size, self.args.image_size),
                mode='bicubic'
            )
        
        return driver_img_crop

    def _detect_driver_faces(self, driver_images: List[PIL.Image.Image]) -> List[Optional[np.ndarray]]:
        """Detect faces in driver images."""
        driver_faces = []
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            for img in driver_images:
                np_d = np.array(img)
                results = face_detection.process(np_d)
                
                if results.detections is None:
                    driver_faces.append(None)
                else:
                    r = results.detections[0].location_data.relative_bounding_box
                    driver_faces.append(np.array([
                        img.size[0] * r.xmin,
                        img.size[1] * r.ymin * 0.9,
                        img.size[0] * (r.xmin + r.width),
                        min(img.size[1] * (r.ymin + r.height * 1.2),
                            img.size[1] - 1)
                    ]))
        
        return driver_faces
    
    def get_mask(self, img):

        im_transform = transforms.Compose(
            [

                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        im = im_transform(img)
        ref_size = 512
        # add mini-batch dim

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
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

        # inference
        _, _, matte = self.modnet(im.cuda(), True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

        return matte

    def get_mixing_theta(self, source_theta, target_theta):
        source_theta = source_theta[:, :3, :]
        target_theta = target_theta[:, :3, :]
        N = 1
        B = source_theta.shape[0] // N
        T = target_theta.shape[0] // B

        source_theta_ = np.stack([np.eye(4) for i in range(B)])
        target_theta_ = np.stack([np.eye(4) for i in range(B * T)])

        source_theta = source_theta.view(B, N, *source_theta.shape[1:])[:, 0]  # take theta from the first source image
        target_theta = target_theta.view(B, T, 3, 4).roll(1, dims=0).view(B * T, 3, 4)  # shuffle target poses

        source_theta_[:, :3, :] = source_theta.detach().cpu().numpy()
        target_theta_[:, :3, :] = target_theta.detach().cpu().numpy()

        # Extract target translation
        target_translation = np.stack([np.eye(4) for i in range(B * T)])
        target_translation[:, :3, 3] = target_theta_[:, :3, 3]

        # Extract linear components
        source_linear_comp = source_theta_.copy()
        source_linear_comp[:, :3, 3] = 0

        target_linear_comp = target_theta_.copy()
        target_linear_comp[:, :3, 3] = 0

        pred_mixing_theta = []
        for b in range(B):
            # Sometimes the decomposition is not possible, hense try-except blocks
            try:
                source_rotation, source_stretch = linalg.polar(source_linear_comp[b])
            except:
                pred_mixing_theta += [target_theta_[b * T + t] for t in range(T)]
            else:
                for t in range(T):
                    try:
                        target_rotation, target_stretch = linalg.polar(target_linear_comp[b * T + t])
                    except:
                        pred_mixing_theta.append(source_stretch)
                    else:
                        if self.mix_old:
                            pred_mixing_theta.append(target_translation[b * T + t] @ target_rotation @ source_stretch)
                        else:
                            pred_mixing_theta.append(source_stretch * target_stretch.mean() / source_stretch.mean() @ target_rotation @ target_translation[b * T + t])

                        # pred_mixing_theta.append(source_stretch * target_stretch.mean() / source_stretch.mean() @ target_rotation @ target_translation[b * T + t])
                        
        pred_mixing_theta = np.stack(pred_mixing_theta)

        return torch.from_numpy(pred_mixing_theta)[:, :3].type(source_theta.type()).to(source_theta.device)

    def process_video(self, 
                source_img: PIL.Image.Image,
                video_path: str,
                max_frames: Optional[int] = None) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
        """Process video frames using source image.
        
        Args:
            source_img: Source identity image
            video_path: Path to driving video
            max_frames: Optional limit on number of frames to process
            
        Returns:
            Tuple of (generated frames, driving frames)
        """
        import time
        
        # Get video frames
        cap = cv2.VideoCapture(video_path)
        driving_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and len(driving_frames) >= max_frames):
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = to_512(Image.fromarray(frame_rgb))
            driving_frames.append(frame_pil)
            
        cap.release()
        
        print(f"Total frames to process: {len(driving_frames)}")
        
        # Process first frame to initialize source
        start_time = time.time()
        results = self.forward(
            source_image=source_img,
            driver_image=driving_frames[0],
            crop=False,
            smooth_pose=False, 
            target_theta=True,
            mix=True,
            mix_old=False,
            modnet_mask=False
        )
        
        generated_frames = []
        if results:
            generated_frames.append(results[0][0])
        
        # Process remaining frames with FPS tracking
        frame_times = []
        
        for i, frame in enumerate(driving_frames[1:], 1):
            frame_start = time.time()
            
            results = self.forward(
                source_image=None,  # Source already processed
                driver_image=frame,
                crop=False,
                smooth_pose=False,
                target_theta=True, 
                mix=True,
                mix_old=False,
                modnet_mask=False
            )
            
            if results:
                generated_frames.append(results[0][0])
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            # Calculate and display running statistics
            if i % 10 == 0:  # Update every 10 frames
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                eta = (len(driving_frames) - i) * (sum(frame_times) / len(frame_times))
                print(f"Processed {i}/{len(driving_frames)} frames | "
                    f"Average FPS: {avg_fps:.2f} | "
                    f"ETA: {eta:.2f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = len(driving_frames) / total_time
        print(f"\nGeneration complete:")
        print(f"Total frames: {len(driving_frames)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average time per frame: {1000 * total_time / len(driving_frames):.2f}ms")
        
        return generated_frames, driving_frames
    

    def batch_process_video(self, 
                source_img: PIL.Image.Image,
                video_path: str,
                max_frames: Optional[int] = None,
                batch_size: int = 8) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
        """Process video frames using source image with batch processing.
        
        Args:
            source_img: Source identity image
            video_path: Path to driving video
            max_frames: Optional limit on number of frames to process
            batch_size: Number of frames to process simultaneously
            
        Returns:
            Tuple of (generated frames, driving frames)
        """
        import time
        
        # Get video frames
        cap = cv2.VideoCapture(video_path)
        driving_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and len(driving_frames) >= max_frames):
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = to_512(Image.fromarray(frame_rgb))
            driving_frames.append(frame_pil)
            
        cap.release()
        
        print(f"Total frames to process: {len(driving_frames)}")
        
        # Process first frame to initialize source
        start_time = time.time()
        results = self.forward(
            source_image=source_img,
            driver_image=driving_frames[0],
            crop=False,
            smooth_pose=False, 
            target_theta=True,
            mix=True,
            mix_old=False,
            modnet_mask=False
        )
        
        generated_frames = []
        if results:
            generated_frames.append(results[0][0])
        
        # Process remaining frames in batches with FPS tracking
        frame_times = []
        remaining_frames = driving_frames[1:]
        
        for i in range(0, len(remaining_frames), batch_size):
            batch_start = time.time()
            
            # Get current batch
            batch_frames = remaining_frames[i:i + batch_size]
            
            # Forward pass with batch
            results = self.forward(
                source_image=None,  # Source already processed
                driver_image=batch_frames,  # Pass batch of frames
                crop=False,
                smooth_pose=False,
                target_theta=True, 
                mix=True,
                mix_old=False,
                modnet_mask=False
            )
            
            if results:
                generated_frames.extend(results[0])  # Extend with batch results
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            frame_times.append(batch_time / len(batch_frames))  # Per-frame time
            
            # Calculate and display running statistics
            frames_processed = i + len(batch_frames)
            if frames_processed % (batch_size * 2) == 0:  # Update every few batches
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                eta = (len(remaining_frames) - frames_processed) * (sum(frame_times) / len(frame_times))
                print(f"Processed {frames_processed}/{len(driving_frames)} frames | "
                    f"Average FPS: {avg_fps:.2f} | "
                    f"ETA: {eta:.2f}s | "
                    f"Batch time: {batch_time:.3f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = len(driving_frames) / total_time
        print(f"\nGeneration complete:")
        print(f"Total frames: {len(driving_frames)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average time per frame: {1000 * total_time / len(driving_frames):.2f}ms")
        print(f"Average time per batch: {1000 * total_time * batch_size / len(driving_frames):.2f}ms")
        
        return generated_frames, driving_frames
    def save_video(self,
            source_img: PIL.Image.Image,
            generated_frames: List[PIL.Image.Image], 
            driving_frames: List[PIL.Image.Image],
            output_path: str,
            fps: float = 30.0,
            size: Tuple[int, int] = (512, 512)):
        """Save video with source, driving and generated frames.
        
        Args:
            source_img: Source identity image
            generated_frames: List of generated frame images
            driving_frames: List of driving frame images
            output_path: Path to save video
            fps: Frames per second
            size: Size of output frames
        """
        # Setup video writer
        out_size = (size[0] * 3, size[1])  # 3 images side by side
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
        
        # Convert source image to RGB and resize
        source_img_rgb = source_img.convert('RGB')
        source_array = np.array(source_img_rgb.resize(size))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for gen_frame in generated_frames:
            # Ensure RGB format and resize
            gen_frame_rgb = gen_frame.convert('RGB')
            gen_array = np.array(gen_frame_rgb.resize(size))
        
        # Convert to BGR for OpenCV
        out.write(cv2.cvtColor(gen_array, cv2.COLOR_RGB2BGR))

        # for gen_frame, drive_frame in zip(generated_frames, driving_frames):
        #     # Ensure RGB format and resize
        #     gen_frame_rgb = gen_frame.convert('RGB')
        #     drive_frame_rgb = drive_frame.convert('RGB')
            
        #     gen_array = np.array(gen_frame_rgb.resize(size))
        #     drive_array = np.array(drive_frame_rgb.resize(size))
            
        #     # Concatenate frames
        #     composite = np.concatenate([source_array, drive_array, gen_array], axis=1)
        #     out.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            
        # out.release()

        
from typing import Optional, List, Dict, Union
# Usage example:
def create_driven_video(
    source_path: str,
    video_path: str,
    output_path: str,
    inferer: InferenceWrapper,
    max_frames: Optional[int] = None,
    fps: float = 30.0
):
    """Create driven video from source image and driving video.
    
    Args:
        source_path: Path to source image
        video_path: Path to driving video
        output_path: Path for output video
        inferer: Initialized InferenceWrapper
        max_frames: Optional maximum number of frames to process
        fps: Output video frame rate
    """
    # Load source image
    source_img = to_512(Image.open(source_path))
    
    # Process video
    generated_frames, driving_frames = inferer.process_video(
        source_img=source_img,
        video_path=video_path,
        max_frames=max_frames
    )
    
    # Save video
    inferer.save_video(
        source_img=source_img,
        generated_frames=generated_frames,
        driving_frames=driving_frames,
        output_path=output_path,
        fps=fps
    )

threshold = 0.8
device = 'cuda'
# face_detector = RetinaFacePredictor(threshold=threshold, device=device, model=(RetinaFacePredictor.get_model('mobilenet0.25')))
args_overwrite = {'l1_vol_rgb':0}
inferer = InferenceWrapper(experiment_name = 'speedup', model_file_name = '328_model.pth',
                           project_dir = '/media/2TB/EMOPortraits', folder = 'logs', state_dict = None,
                           args_overwrite=args_overwrite, pose_momentum = 0.1, print_model=False, print_params = True)

# inferer = InferenceWrapper()
create_driven_video(
    source_path='data/IMG_1.png',
    video_path='data/VID_1.mp4',
    output_path='data/result3.mp4',
    inferer=inferer,
    max_frames=None,  # Process all frames
    fps=30.0
)