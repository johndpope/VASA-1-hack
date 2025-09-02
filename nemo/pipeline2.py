
import argparse
import os
import cv2
import torch
import numpy as np
from torch import nn
from glob import glob
from PIL import Image

from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm.notebook import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage

from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1) 
to_512 = lambda x: x.resize((512, 512), Image.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.LANCZOS)



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
from typing import *
from logger import logger
from PIL import Image
from vis_helper import save_expression_embed



def log_tensor_state(name: str, tensor: Optional[Union[torch.Tensor, Image.Image, list]], detailed: bool = False) -> None:
    """Log tensor state with shape and value range if available."""
    if tensor is None:
        logger.debug(f"{name}: None")
    elif isinstance(tensor, Image.Image):
        logger.debug(f"{name}: PIL Image size={tensor.size}")
    elif isinstance(tensor, list):
        logger.debug(f"{name}: List of {len(tensor)} items")
        if len(tensor) > 0:
            if isinstance(tensor[0], Image.Image):
                logger.debug(f"{name} (first item): PIL Image size={tensor[0].size}")
            elif isinstance(tensor[0], torch.Tensor):
                logger.debug(f"{name} (first item): Tensor shape={tensor[0].shape}")
    elif isinstance(tensor, torch.Tensor):
        info = f"{name}: shape={tensor.shape}"
        if detailed and torch.is_floating_point(tensor):
            info += f", range=[{tensor.min():.3f}, {tensor.max():.3f}]"
        logger.debug(info)
    else:
        logger.debug(f"{name}: type={type(tensor)}")

def log_data_dict(prefix: str, data_dict: Dict[str, torch.Tensor]) -> None:
    """Log the shapes of tensors in a data dictionary."""
    logger.debug(f"\n{prefix} data dictionary contents:")
    for k, v in data_dict.items():
        if isinstance(v, (torch.Tensor, Image.Image, list)):
            log_tensor_state(f"  {k}", v)

def log_processing_step(step_name: str) -> None:
    """Log a processing step header."""
    logger.debug(f"\n=== {step_name} ===")


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
        args_path = pathlib.Path(project_dir) / folder / experiment_name / 'args.txt' if args_path is None else args_path

        self.args = args_utils.parse_args(args_path)
        # Add args from args_overwrite dict that overwrite the default ones
        self.args.project_dir = project_dir
        if args_overwrite is not None:
            for k, v in args_overwrite.items():
                setattr(self.args, k, v)

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

        self.model = importlib.import_module(f'models.stage_1.volumetric_avatar.{model_}').Model(self.args, training=False)

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

    # @torch.no_grad()
    # def calculate_standing_stats(self, data_root, num_iters):
    #     self.identity_embedder.train().apply(stats_calc.stats_calculation)
    #     self.pose_embedder.train().apply(stats_calc.stats_calculation)
    #     self.generator.train().apply(stats_calc.stats_calculation)

    #     # Initialize train dataset
    #     dataset = LMDBDataset(
    #         data_root,
    #         'train',
    #         self.args.num_source_frames,
    #         self.args.num_target_frames,
    #         self.args.image_size,
    #         False)

    #     dataset.names = dataset.names[:self.args.batch_size * num_iters]

    #     dataloader = data.DataLoader(
    #         dataset,
    #         batch_size=self.args.batch_size,
    #         drop_last=True,
    #         num_workers=self.args.num_workers_per_process)

    #     for data_dict in dataloader:
    #         source_img_crop = data_dict['source_img']
    #         driver_img_crop = data_dict['target_img']

    #         source_img_crop = source_img_crop.view(-1, *source_img_crop.shape[2:])
    #         driver_img_crop = driver_img_crop.view(-1, *driver_img_crop.shape[2:])

    #         if self.use_gpu:
    #             source_img_crop = source_img_crop.cuda()
    #             driver_img_crop = driver_img_crop.cuda()

    #         idt_embed = self.identity_embedder.forward_image(source_img_crop)

    #         # During training, pose embedder accepts concatenated data, so we need to imitate it during stats calculation
    #         img_crop = torch.cat([source_img_crop, driver_img_crop])
    #         pose_embed, pred_theta = self.pose_embedder.forward_image(img_crop)

    #         source_pose_embed, driver_pose_embed = pose_embed.split(
    #             [source_img_crop.shape[0], driver_img_crop.shape[0]])
    #         pred_source_theta, pred_driver_theta = pred_theta.split(
    #             [source_img_crop.shape[0], driver_img_crop.shape[0]])

    #         latent_texture, embed_dict = self.generator.forward_source(source_img_crop, idt_embed, source_pose_embed,
    #                                                                    pred_source_theta)
    #         pred_target_img = self.generator.forward_driver(idt_embed, driver_pose_embed, embed_dict, pred_source_theta,
    #                                                         pred_driver_theta, latent_texture)

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

    def crop_image_old(self, image, faces, use_smoothed_crop=False):
        imgs_crop = []

        for b, face in enumerate(faces):

            assert face is not None, 'Face not found!'

            center = np.asarray([(face[2] + face[0]) // 2, (face[3] + face[1]) // 2])
            size = face[2] - face[0] + face[3] - face[1]

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
            size = self.remove_overflow(center, size, image.shape[3], image.shape[2])

            img_crop = image[b, :, center[1] - size // 2: center[1] + size // 2,
                       center[0] - size // 2: center[0] + size // 2]
            img_crop = F.interpolate(img_crop[None], size=(self.args.image_size, self.args.image_size), mode='bicubic')

            imgs_crop += [img_crop]

        imgs_crop = torch.cat(imgs_crop)

        return imgs_crop

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


    
    def forward(self, source_image=None, driver_image=None, source_mask=None, source_mask_add=0, 
         driver_mask=None, crop=True, reset_tracking=False, smooth_pose=False,
         hard_normalize=False, soft_normalize=False, delta_yaw=None, delta_pitch=None, cloth=False,
         thetas_pass='', theta_n=0, target_theta=True, mix=False, mix_old=True, 
         c_source_latent_volume=None, c_target_latent_volume=None, custome_target_pose_embed=None, 
         custome_target_theta_embed=None, no_grad_infer=True, modnet_mask=False,frame_idx=0):
    
        log_processing_step("Forward Pass Initialization")
        log_tensor_state("Input source image", source_image)
        log_tensor_state("Input driver image", driver_image)
        
        self.no_grad_infer = no_grad_infer
        self.target_theta = target_theta
        
        with torch.no_grad():
            if reset_tracking:
                log_processing_step("Resetting Tracking Parameters")
                self.center = None
                self.size = None
                self.theta = None
                self.delta_yaw = None
                self.delta_pitch = None
            
            self.mix = mix
            self.mix_old = mix_old
            if delta_yaw is not None:
                self.delta_yaw = delta_yaw
            if delta_pitch is not None:
                self.delta_pitch = delta_pitch

            if source_image is not None:
                log_processing_step("Source Image Processing")
                
                if crop:
                    log_processing_step("Face Detection and Cropping")
                    source_faces = []
                    
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                        np_s = np.array(source_image)
                        results = face_detection.process(np_s)
                        if results.detections is None:
                            logger.debug("No face detected in source image")
                            source_faces.append(None)
                        else:
                            r = results.detections[0].location_data.relative_bounding_box
                            bbox = np.array([
                                source_image.size[0] * r.xmin,
                                source_image.size[1] * r.ymin * 0.9,
                                source_image.size[0] * (r.xmin + r.width),
                                min(source_image.size[1] * (r.ymin + r.height * 1.2), source_image.size[1] - 1)])
                            source_faces.append(bbox)
                            logger.debug(f"Face detected at bbox: {bbox}")

                    source_img_crop, _, _ = self.crop_image([self.to_tensor(source_image)], source_faces)
                    log_tensor_state("Cropped source image", source_img_crop)
                else:
                    log_processing_step("Image Resizing (No Crop)")
                    if source_image is not None:
                        source_image = self.convert_to_tensor(source_image)[:, :3]
                    source_image = F.interpolate(source_image, size=(self.args.image_size, self.args.image_size),
                                                mode='bicubic')
                    source_img_crop = source_image
                    log_tensor_state("Resized source image", source_img_crop)

                self.source_image = source_image
                self.source_image_crop = source_img_crop
                source_img_crop = source_img_crop.to(self.device)
                
                log_processing_step("Face Mask Generation")
                trashhold = 0.6
                face_mask_source, _, _, cloth_s = self.model.face_idt.forward(source_img_crop)
                log_tensor_state("Raw face mask", face_mask_source)
                face_mask_source = (face_mask_source > trashhold).float()
                log_tensor_state("Thresholded face mask", face_mask_source)

                log_processing_step("MODNet Mask Generation")
                source_mask_modnet = self.get_mask(source_img_crop)
                log_tensor_state("MODNet mask", source_mask_modnet)

                face_mask_source = (face_mask_source).float()
                source_img_crop = (source_img_crop * face_mask_source).float()
                log_tensor_state("Masked source image", source_img_crop)

                self.source_img_crop_m = source_img_crop

                log_processing_step("Final Mask Processing")
                source_img_mask = source_mask if source_mask is not None else face_mask_source
                source_img_mask = source_mask_modnet if modnet_mask else source_img_mask
                if source_mask_add:
                    source_img_mask = source_img_mask.clamp_(max=1, min=0)
                log_tensor_state("Final source mask", source_img_mask)

                self.source_img_mask = source_img_mask
                c = self.args.latent_volume_channels
                s = self.args.latent_volume_size
                d = self.args.latent_volume_depth
                logger.debug(f"Volume dimensions: channels={c}, size={s}, depth={d}")

                source_img_mask = source_img_mask.to(self.device)
                
                log_processing_step("Feature Extraction")
                self.idt_embed = self.model.idt_embedder_nw.forward_image(source_img_crop * source_img_mask)
                log_tensor_state("Identity embedding", self.idt_embed)
                
                source_latents = self.model.local_encoder_nw(source_img_crop * source_img_mask)
                log_tensor_state("Source latents", source_latents)

                log_processing_step("Head Pose Regression")
                with torch.no_grad():
                    pred_source_theta = self.model.head_pose_regressor.forward(source_img_crop)
                log_tensor_state("Source theta", pred_source_theta)
                
                self.pred_source_theta = pred_source_theta

                log_processing_step("3D Grid Processing")
                grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
                log_tensor_state("Identity grid", grid)

                inv_source_theta = pred_source_theta.float().inverse().type(pred_source_theta.type())
                source_rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                log_tensor_state("Source rotation warp", source_rotation_warp)

                log_processing_step("Data Dictionary Creation")
                data_dict = {
                    'source_img': source_img_crop, 
                    'source_mask': source_img_mask,
                    'source_theta': pred_source_theta,
                    'target_img': source_img_crop, 
                    'target_mask': source_img_mask,
                    'target_theta': pred_source_theta
                }
                data_dict['idt_embed'] = self.idt_embed
                log_data_dict("Initial data dictionary", data_dict)

                log_processing_step("Expression Embedding")
                data_dict = self.model.expression_embedder_nw(data_dict, True, False)
                log_data_dict("After expression embedding", data_dict)

                save_expression_embed(data_dict['source_pose_embed'], frame_idx, 'wrapper')

                self.pred_source_pose_embed = data_dict['source_pose_embed']
                source_pose_embed = data_dict['source_pose_embed']
                self.source_img_align = data_dict['source_img_align']
                self.source_img = source_img_crop
                self.align_warp = data_dict['align_warp']

                log_processing_step("Embedding Prediction")
                source_warp_embed_dict, _, _, embed_dict = self.model.predict_embed(data_dict)
                log_tensor_state("Source warp embed dict", source_warp_embed_dict)

                log_processing_step("XY Generation")
                xy_gen_outputs = self.model.xy_generator_nw(source_warp_embed_dict)
                data_dict['source_delta_xy'] = xy_gen_outputs[0]
                log_tensor_state("XY generator outputs", xy_gen_outputs[0])

                source_xy_warp = xy_gen_outputs[0]
                source_xy_warp_resize = source_xy_warp
                if self.resize_warp:
                    source_xy_warp_resize = self.model.resize_warp_func(source_xy_warp_resize)
                log_tensor_state("Source XY warp resize", source_xy_warp_resize)

                source_latents_face = source_latents
                source_latent_volume = source_latents_face.view(1, c, d, s, s)
                log_tensor_state("Source latent volume", source_latent_volume)

                if self.args.source_volume_num_blocks > 0:
                    source_latent_volume = self.model.volume_source_nw(source_latent_volume)
                    log_tensor_state("Processed source latent volume", source_latent_volume)

                self.source_latent_volume = source_latent_volume if c_source_latent_volume is None else c_source_latent_volume
                self.source_rotation_warp = source_rotation_warp 
                self.source_xy_warp_resize = source_xy_warp_resize

                log_processing_step("Target Volume Generation")
                target_latent_volume = self.model.grid_sample(
                    self.model.grid_sample(self.source_latent_volume, source_rotation_warp), 
                    source_xy_warp_resize)
                log_tensor_state("Target latent volume", target_latent_volume)

                self.target_latent_volume_1 = target_latent_volume if c_target_latent_volume is None else c_target_latent_volume
                self.target_latent_volume = self.model.volume_process_nw(self.target_latent_volume_1, embed_dict)
                log_tensor_state("Processed target latent volume", self.target_latent_volume)

        with torch.no_grad() if self.no_grad_infer else none_context:
            if driver_image is not None:
                log_processing_step("Driver Image Processing")
                c = self.args.latent_volume_channels
                s = self.args.latent_volume_size
                d = self.args.latent_volume_depth
                
                if crop:
                    log_processing_step("Driver Face Detection")
                    if not isinstance(driver_image, list):
                        driver_image = [driver_image]

                    driver_faces = []
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                        for img in driver_image:
                            np_d = np.array(img)
                            results = face_detection.process(np_d)
                            if results.detections is None:
                                logger.debug("No face detected in driver image")
                                driver_faces.append(None)
                            else:
                                r = results.detections[0].location_data.relative_bounding_box
                                bbox = np.array([
                                    img.size[0] * r.xmin,
                                    img.size[1] * r.ymin * 0.9,
                                    img.size[0] * (r.xmin + r.width),
                                    min(img.size[1] * (r.ymin + r.height * 1.2), img.size[1] - 1)])
                                driver_faces.append(bbox)
                                logger.debug(f"Driver face detected at bbox: {bbox}")

                    driver_image = [self.to_tensor(img) for img in driver_image]
                    driver_img_crop, face_check, face_scale_stats = self.crop_image(driver_image, driver_faces)
                    log_tensor_state("Cropped driver image", driver_img_crop)
                else:
                    log_processing_step("Driver Image Resizing (No Crop)")
                    if driver_image is not None:
                        driver_image = self.convert_to_tensor(driver_image)[:, :3]
                    driver_image = F.interpolate(driver_image, size=(self.args.image_size, self.args.image_size),
                                                mode='bicubic')
                    driver_img_crop = driver_image
                    log_tensor_state("Resized driver image", driver_img_crop)

                driver_img_crop = driver_img_crop.to(self.device)
                
                log_processing_step("Driver Head Pose Regression")
                with torch.no_grad():
                    pred_target_theta, scale, rotation, translation = self.model.head_pose_regressor.forward(driver_img_crop, True)
                log_tensor_state("Target theta", pred_target_theta)
                logger.debug(f"Scale: {scale}, Rotation: {rotation}, Translation: {translation}")
                
                self.pred_target_theta = pred_target_theta
                self.pred_target_srt = (scale, rotation, translation)
                
                if custome_target_theta_embed is not None:
                    pred_target_theta = point_transforms.get_transform_matrix(*custome_target_theta_embed)
                    log_tensor_state("Custom target theta", pred_target_theta)

                if self.mix:
                    log_processing_step("Theta Mixing")
                    pred_target_theta = self.get_mixing_theta(self.pred_source_theta, pred_target_theta)
                    log_tensor_state("Mixed target theta", pred_target_theta)

                if smooth_pose:
                    log_processing_step("Pose Smoothing")
                    if self.theta is None:
                        self.theta = pred_target_theta[0].clone()

                    smooth_driver_theta = []
                    for i in range(len(driver_image)):
                        self.theta = pred_target_theta[i] * self.pose_momentum + self.theta * (1 - self.pose_momentum)
                        smooth_driver_theta.append(self.theta.clone())
                        logger.debug(f"Smoothed theta mean at index {i}: {torch.mean(self.theta)}")

                    pred_target_theta = torch.stack(smooth_driver_theta)
                    log_tensor_state("Smoothed target theta", pred_target_theta)

                grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
                self.pred_target_theta = pred_target_theta
                
                log_processing_step("Target Rotation Warp")
                if self.target_theta:
                    target_rotation_warp = grid.bmm(pred_target_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                else:
                    target_rotation_warp = grid.bmm(pred_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                log_tensor_state("Target rotation warp", target_rotation_warp)

                log_processing_step("Driver Mask Generation")
                driver_img_mask = driver_mask if driver_mask is not None else self.get_mask(driver_img_crop)
                driver_img_mask = driver_img_mask.to(driver_img_crop.device)
                log_tensor_state("Driver mask", driver_img_mask)

                log_processing_step("Driver Data Dictionary Creation")
                data_dict = {
                    'source_img': driver_img_crop, 
                    'source_mask': driver_img_mask,
                    'source_theta': pred_target_theta,
                    'target_img': driver_img_crop, 
                    'target_mask': driver_img_mask,
                    'target_theta': pred_target_theta
                }
                data_dict['idt_embed'] = self.idt_embed
                log_data_dict("Initial driver data dictionary", data_dict)

                log_processing_step("Driver Expression Embedding")
                data_dict = self.model.expression_embedder_nw(data_dict, True, False)
                log_data_dict("After driver expression embedding", data_dict)
                save_expression_embed(data_dict['target_pose_embed'], frame_idx, 'wrapper_target')

                if custome_target_pose_embed is not None:
                    log_processing_step("Custom Target Pose Embedding")
                    data_dict['target_pose_embed'] = custome_target_pose_embed
                    log_tensor_state("Custom target pose embed", custome_target_pose_embed)

                target_pose_embed = data_dict['target_pose_embed']
                self.target_pose_embed = target_pose_embed
                self.target_img_align = data_dict['target_img_align']
                
                log_processing_step("Driver Embedding Prediction")
                bla, target_warp_embed_dict, bla1, embed_dict = self.model.predict_embed(data_dict)

        
                
                # Log detailed info about target_warp_embed_dict
                logger.debug("Target warp embed dict contents:")
                if isinstance(target_warp_embed_dict, dict):
                    for key, value in target_warp_embed_dict.items():
                        if isinstance(value, torch.Tensor):
                            logger.debug(f"  {key}: shape={value.shape}")
                        else:
                            logger.debug(f"  {key}: type={type(value)}")
                else:
                    logger.debug(f"  type={type(target_warp_embed_dict)}")
                    
                log_tensor_state("Target warp embed dict", target_warp_embed_dict)
                log_data_dict("Final embed dict", embed_dict)

                log_processing_step("UV Generation")
                target_uv_warp, data_dict['target_delta_uv'] = self.model.uv_generator_nw(target_warp_embed_dict)
                log_tensor_state("Target UV warp", target_uv_warp)
                
                target_uv_warp_resize = target_uv_warp
                if self.resize_warp:
                    target_uv_warp_resize = self.model.resize_warp_func(target_uv_warp_resize)
                target_uv_warp_resize = target_uv_warp_resize
                log_tensor_state("Target UV warp resize", target_uv_warp_resize)

                log_processing_step("Target Volume Alignment")
                aligned_target_volume = self.model.grid_sample(
                    self.model.grid_sample(self.target_latent_volume, target_uv_warp_resize), 
                    target_rotation_warp)
                log_tensor_state("Aligned target volume", aligned_target_volume)

                target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
                log_tensor_state("Target latent features", target_latent_feats)

                log_processing_step("Final Image Generation")

                log_data_dict("data_dict", data_dict)
                log_data_dict("embed_dict", embed_dict)
                log_tensor_state("target_latent_feats", target_latent_feats)

                img, _, deep_f, img_f = self.model.decoder_nw(
                    data_dict, 
                    embed_dict, 
                    target_latent_feats, 
                    False,
                    stage_two=True
                )
                log_tensor_state("Generated image", img)
                log_tensor_state("Deep features", deep_f)
                log_tensor_state("Image features", img_f)

                pred_target_img = img.detach().cpu().clamp(0, 1)
                pred_target_img = [self.to_image(img) for img in pred_target_img]
                logger.debug(f"Generated {len(pred_target_img)} target images")

                return pred_target_img, img
            else:
                logger.debug("No driver image provided, returning None")
                return None
                

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


def get_video_frames_as_images(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертация из BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = to_512(Image.fromarray(frame_rgb))
        frames.append(img)

    cap.release()
    return frames

def make_video(source, drivers, out_frames_b, path, fps=30.0):
    videodims = (3*512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, videodims)
    #draw stuff that goes on every frame here
    for i in tqdm(range(0,len(out_frames_b))):
        our_s2 = out_frames_b[i]
        out_img = np.array(np.concatenate([np.array(source.resize((512, 512), Image.LANCZOS))[:, :, :3], np.array(drivers[i].resize((512, 512), Image.LANCZOS)), np.array(our_s2.resize((512, 512), Image.LANCZOS))], axis=1))
        video.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    video.release()

def get_bg(s_img, mdnt = True):
    gt_img_t = to_tensor(s_img)[:3].unsqueeze(dim=0).cuda()
    m = get_mask(gt_img_t) if mdnt else get_mask_fp(gt_img_t)
    kernel_back = np.ones((21, 21), 'uint8')
    mask = (m >= 0.8).float()
    mask = mask[0].permute(1,2,0)
    dilate_mask = cv2.dilate(mask.cpu().numpy(), kernel_back, iterations=2)
    dilate_mask = torch.FloatTensor(dilate_mask).unsqueeze(0).unsqueeze(0).cuda()
    background = lama(gt_img_t.cuda(), dilate_mask.cuda())
    bg_img = to_image(background[0])
    bg = to_tensor(bg_img.resize((512, 512), Image.BICUBIC))
    return bg, bg_img

def get_modnet_mask(img):
    im_transform = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    im = im_transform(img)
    ref_size = 512
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
    _, _, matte = modnet(im.cuda(), True)

    # resize and save matteget_mask
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

    return matte

@torch.no_grad()
def get_mask(source_img_crop):
    source_img_mask = get_modnet_mask(source_img_crop)
    source_img_mask = source_img_mask
    source_img_mask = source_img_mask.clamp_(max=1, min=0)
    
    return source_img_mask


@torch.no_grad()
def get_mask_fp(source_img_crop):
    face_mask_source, _, _, cloth_s = face_idt.forward(source_img_crop)
    trashhold = 0.6
    face_mask_source = (face_mask_source > trashhold).float()

    source_mask_modnet = get_mask(source_img_crop)

    face_mask_source = (face_mask_source*source_mask_modnet).float()

    return face_mask_source

def make_video_crop(source, drivers, out_frames_b, path, fps=30.0, k=2, size=128):
    videodims = (3*512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, videodims)
    #draw stuff that goes on every frame here
    for i in tqdm(range(0,len(out_frames_b))):
        our_s2 = out_frames_b[i]
        out_img = np.array(np.concatenate([np.array(source.resize((512, 512), Image.LANCZOS))[:, :, :3], np.array(to_512(to_512(drivers[i]).crop((256-size*k, 256-size*k, 256+size*k, 256+size*k))) ), np.array(to_512(to_512(our_s2).crop((256-size*k, 256-size*k, 256+size*k, 256+size*k)))    )], axis=1))
        video.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    video.release()

def connect_img_and_bg(img, bg, mdnt=True):
    pred_img_t = to_tensor(img)[:3].unsqueeze(0).cuda()
    _source_img_mask = get_modnet_mask(pred_img_t) if mdnt else get_mask_fp(pred_img_t)
    mask_sss = torch.where(_source_img_mask>0.3, _source_img_mask, _source_img_mask*0)**8
    out_nn = mask_sss.cpu()*pred_img_t.cpu()+ (1-mask_sss.cpu())*bg.cpu()
    return to_image(out_nn[0])

def drive_image_with_video(source, video_path = '/path/to/your/xxx.mp4', max_len=None):

    all_srs = []
    all_bgs = []
    all_img_bg = []

    all_curr_d = get_video_frames_as_images(video_path)

    all_curr_d = all_curr_d[:max_len]
    
    first_d = all_curr_d[0]
    img = inferer.forward(source, first_d, crop=False, smooth_pose=False, target_theta=True, mix=True, mix_old=False, modnet_mask=False)
    all_srs.append(source)
    
    #make background 
    bg, bg_img = get_bg(source, mdnt=False)
    all_bgs.append(bg_img)
    
    # infer
    img_with_bg = connect_img_and_bg(img[0][0], bg, mdnt=False)
    # sr_img_with_bg = to_512(do_stage_2(img_with_bg))
    all_img_bg.append(img_with_bg)
     
        
    for i, curr_d in enumerate(tqdm(all_curr_d[1:]), 1):  # Start enumerate from 1
        frame_idx = i  # Update frame index
        img = inferer.forward(None, curr_d, crop=False, smooth_pose=False, 
                            target_theta=True, mix=True, mix_old=False, 
                            modnet_mask=False, frame_idx=frame_idx)
        img_with_bg = connect_img_and_bg(img[0][0], bg, mdnt=False)
        all_img_bg.append(img_with_bg)

    return all_img_bg, all_curr_d


def get_custom_crop_first(img, k=1.2, in_s=512, first_frame=True, center=None, size=None):
    mpl = in_s//512
    img_cv2 =  np.asarray(img)*255
    if first_frame:
        _faces = face_detector(img_cv2, rgb=False)
        if _faces.shape[0]==0:
            raise ValueError('Face not found')
        else:
            faces = _faces[0]
            center = (int(faces[0] + (faces[2]-faces[0])/2), int(faces[1]+ (faces[3]-faces[1])/2))
            size = max(int((faces[2]-faces[0])/2), int((faces[3]-faces[1])/2))
            new_img = to_512(img.crop((center[0]-size*k, center[1]-size*k, center[0]+size*k, center[1]+size*k)))
            return new_img, center, size
    else:
        new_img = to_512(img.crop((center[0]-size*k, center[1]-size*k, center[0]+size*k, center[1]+size*k)))
        return new_img


project_dir = os.path.dirname(os.path.abspath(__file__))
args_overwrite = {'l1_vol_rgb':0}
face_idt = FaceParsing(None, 'cuda')

lama = torch.jit.load('repos/jit_lama.pt').cuda()

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load('repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
modnet.eval();

threshold = 0.8
device = 'cuda'
face_detector = RetinaFacePredictor(threshold=threshold, device=device, model=(RetinaFacePredictor.get_model('mobilenet0.25')))

inferer = InferenceWrapper(experiment_name = 'Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1', model_file_name = '328_model.pth',
                           project_dir = project_dir, folder = 'logs', state_dict = None,
                           args_overwrite=args_overwrite, pose_momentum = 0.1, print_model=False, print_params = True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_image_path', type=str, default='data/A.png', help='Path to source image')
    parser.add_argument('--driven_video_path', type=str, default='./junk/overfit/ovs-GiY_848_1.mp4', help='Path to driving video')
    parser.add_argument('--saved_to_path', type=str, default='data/result.mp4', help='Path to save result video')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS of output video')
    parser.add_argument('--max_len', type=int, default=1000, help='Maximum number of frames to process')

    args = parser.parse_args()

    source_img = to_512(Image.open(args.source_image_path))
    driving_video_path = args.driven_video_path

    driven_result, drivers = drive_image_with_video(source_img, driving_video_path, max_len=args.max_len)

    save_path = args.saved_to_path

    make_video_crop(source_img, driven_result, drivers, save_path, fps=args.fps)