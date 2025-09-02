from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import losses

@dataclass
class LossWeights:
    """Configuration for loss weights."""
    adversarial: float = 0.0
    feature_matching: float = 0.0
    l1_weight: float = 0.0
    vgg19: float = 0.0
    vgg19_neutral: float = 0.0
    vgg19_face: float = 0.0
    vgg19_emotions: float = 0.0
    vgg19_cycle_idn: float = 0.0
    vgg19_face_cycle_idn: float = 0.0
    vgg19_cycle_exp: float = 0.0
    vgg19_face_cycle_exp: float = 0.0
    face_resnet: float = 0.0
    landmarks: float = 0.0
    warping_reg: float = 0.0
    cycle_idn: float = 0.0
    cycle_exp: float = 0.0
    l1_vol_rgb: float = 0.0
    l1_vol_rgb_mix: float = 0.0
    gaze: float = 0.0
    resnet18_emotions: float = 0.0
    barlow: float = 0.0
    pull_exp: float = 0.0
    push_exp: float = 0.0
    contrastive_exp: float = 0.0
    contrastive_idt: float = 0.0
    resnet18_fv_mix: float = 0.0
    vgg19_fv_mix: float = 0.0
    volumes_pull: float = 0.0
    volumes_push: float = 0.0
    perc_face_pars: float = 0.0
    mix_gen_adversarial: float = 1.0
    neutral_expr_l1_weight: float = 1.0
    stm: float = 1.0

class LossCalculator:
    """Handles loss calculation for training and testing."""
    
    def __init__(self, args, weights: LossWeights, num_source_frames: int):
        self.args = args
        self.weights = weights
        self.num_source_frames = num_source_frames
        self.prev_targets = None
        self.num_b_negs = getattr(args, 'num_b_negs', 4)
        self.pred_seg = getattr(args, 'pred_seg', False)
        self.visualize = getattr(args, 'visualize', False)
        self.sep_test_losses = getattr(args, 'sep_test_losses', False)
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Register Barlow Twins batch normalization
        self.bn = nn.BatchNorm1d(getattr(args, 'lpe_output_channels_expression', 128), 
                                affine=False, track_running_stats=False)
        
    def _init_loss_functions(self):
        """Initialize all loss functions based on weights configuration."""
        if self.weights.adversarial:
            self.adversarial_loss = losses.AdversarialLoss()
            
        if self.weights.feature_matching:
            self.feature_matching_loss = losses.FeatureMatchingLoss()
            
        if self.weights.gaze:
            self.gaze_loss = losses.GazeLoss(
                device='cuda', 
                gaze_model_types=['vgg16'], 
                project_dir=self.args.project_dir
            )
            
        if self.weights.vgg19:
            self.vgg19_loss = losses.PerceptualLoss(
                num_scales=self.args.vgg19_num_scales, 
                use_fp16=False
            )
            
        if self.weights.vgg19_face:
            self.vgg19_loss_face = losses.PerceptualLoss(
                num_scales=2,
                network='vgg_face_dag',
                layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                resize=True,
                weights=(0.03125, 0.0625, 0.125, 0.25, 1.0),
                use_fp16=False
            )
            
        # Initialize basic loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.cosin_sim_pos = nn.CosineEmbeddingLoss(margin=0.1)
        self.cosin_sim = nn.CosineEmbeddingLoss(margin=0.3)
        self.cosin_sim_2 = nn.CosineEmbeddingLoss(margin=0.5, reduce=False)
        self.cosin_dis = nn.CosineSimilarity()
        
        # Initialize test metrics
        self.ssim = losses.SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim = losses.MS_SSIM(data_range=1, size_average=True, channel=3)
        self.psnr = losses.PSNR()
        self.lpips = losses.LPIPS()

    def calc_train_losses(self, 
                         data_dict: Dict[str, Tensor], 
                         mode: str = 'gen', 
                         epoch: int = 0, 
                         iteration: int = 0) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Calculate training losses.
        
        Args:
            data_dict: Dictionary containing all necessary tensors
            mode: Either 'gen' or 'dis' for generator/discriminator losses
            epoch: Current training epoch
            iteration: Current training iteration
            
        Returns:
            total_loss: Combined loss value
            losses_dict: Dictionary containing individual loss components
        """
        losses_dict = {}
        
        if mode == 'dis':
            return self._calc_discriminator_losses(data_dict, epoch, losses_dict)
        
        return self._calc_generator_losses(data_dict, epoch, iteration, losses_dict)

    def calc_test_losses(self, 
                        data_dict: Dict[str, Tensor], 
                        iteration: int = 0) -> Tuple[Dict[str, Tensor], Optional[Tensor], Optional[Tensor]]:
        """Calculate test metrics.
        
        Args:
            data_dict: Dictionary containing all necessary tensors
            iteration: Current iteration number
            
        Returns:
            losses_dict: Dictionary containing test metrics
            expl_var: Optional explained variance
            expl_var_test: Optional test explained variance
        """
        losses_dict = {
            'ssim': self.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean(),
            'psnr': self.psnr(data_dict['pred_target_img'], data_dict['target_img']),
            'lpips': self.lpips(data_dict['pred_target_img'], data_dict['target_img'])
        }
        
        if self.args.image_size > 160:
            losses_dict['ms_ssim'] = self.ms_ssim(
                data_dict['pred_target_img'], 
                data_dict['target_img']
            ).mean()
        
        if self.sep_test_losses:
            self._add_separate_test_metrics(data_dict, losses_dict)
            
        return losses_dict, None, None

    def _calc_discriminator_losses(self, 
                                 data_dict: Dict[str, Tensor], 
                                 epoch: int,
                                 losses_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Calculate discriminator losses."""
        losses_dict['dis_adversarial'] = (
            self.weights.adversarial *
            self.adversarial_loss(
                real_scores=data_dict['real_score_dis'],
                fake_scores=data_dict['fake_score_dis'],
                mode='dis'
            )
        )
        
        if self.args.use_mix_dis and epoch >= self.args.dis2_train_start:
            losses_dict['dis_adversarial_mix'] = (
                self.weights.adversarial *
                self.adversarial_loss(
                    real_scores=data_dict['real_score_dis_mix'],
                    fake_scores=data_dict['fake_score_dis_mix'],
                    mode='dis'
                )
            )
            
        total_loss = sum(losses_dict.values())
        return total_loss, losses_dict

    def _add_separate_test_metrics(self, 
                                 data_dict: Dict[str, Tensor], 
                                 losses_dict: Dict[str, Tensor]) -> None:
        """Add separate test metrics for person and background."""
        # Person metrics
        losses_dict['ssim_person'] = self.ssim(
            data_dict['pred_target_img'] * data_dict['target_mask'],
            data_dict['target_img'] * data_dict['target_mask']
        ).mean()
        
        losses_dict['psnr_person'] = self.psnr(
            data_dict['pred_target_img'] * data_dict['target_mask'],
            data_dict['target_img'] * data_dict['target_mask']
        )
        
        losses_dict['lpips_person'] = self.lpips(
            data_dict['pred_target_img'] * data_dict['target_mask'],
            data_dict['target_img'] * data_dict['target_mask']
        )
        
        # Background metrics
        losses_dict['ssim_back'] = self.ssim(
            data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
            data_dict['target_img'] * (1 - data_dict['target_mask'])
        ).mean()
        
        losses_dict['psnr_back'] = self.psnr(
            data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
            data_dict['target_img'] * (1 - data_dict['target_mask'])
        )
        
        losses_dict['lpips_back'] = self.lpips(
            data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
            data_dict['target_img'] * (1 - data_dict['target_mask'])
        )
        
        if self.args.image_size > 160:
            self._add_separate_ms_ssim_metrics(data_dict, losses_dict)

    def _add_separate_ms_ssim_metrics(self, 
                                    data_dict: Dict[str, Tensor], 
                                    losses_dict: Dict[str, Tensor]) -> None:
        """Add separate MS-SSIM metrics for person and background."""
        losses_dict['ms_ssim_person'] = self.ms_ssim(
            data_dict['pred_target_img'] * data_dict['target_mask'],
            data_dict['target_img'] * data_dict['target_mask']
        ).mean()
        
        losses_dict['ms_ssim_back'] = self.ms_ssim(
            data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
            data_dict['target_img'] * (1 - data_dict['target_mask'])
        ).mean()