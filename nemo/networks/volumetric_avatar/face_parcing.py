import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

class FaceParsing:
    """
    A class for parsing facial features in images using BiSeNet.
    
    Attributes:
        mask_labels (list): Labels for mask segmentation
        face_labels (list): Labels for facial features
        body_labels (list): Labels for body parts
        cloth_labels (list): Labels for clothing
    """
    
    # Class-level constants for label definitions
    ALL_FACE_LABELS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]  # face features
    EAR_LABELS = [7, 8, 9]  # ears
    NECK_LABELS = [14, 15]  # neck
    HAIR_LABEL = [17]  # hair
    HAT_LABEL = [18]  # hat
    CLOTH_LABEL = [16]  # clothing

    def __init__(
        self,
        mask_type=None,
        device="cuda",
        project_dir='/media/2TB/EMOPortraits'
    ):
        """
        Initialize the FaceParsing model.
        
        Args:
            mask_type (str or None): Type of mask to generate. If None, uses default labels.
            device (str): Device to run the model on ('cuda' or 'cpu').
            project_dir (str): Root directory of the project.
        """
        super().__init__()
        
        # Setup paths and import BiSeNet
        face_parsing_path = os.path.join(project_dir, 'repos/face_par_off')
        sys.path.extend([face_parsing_path, project_dir])
        from repos.face_par_off.model import BiSeNet
        
        # Initialize and load model
        self.net = BiSeNet(n_classes=19).to(device)
        checkpoint_path = os.path.join(face_parsing_path, 'res/cp/79999_iter.pth')
        self.net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.net.eval()
        
        # Setup normalization parameters
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        
        # Initialize label lists
        self._initialize_labels(mask_type)
    
    def _initialize_labels(self, mask_type):
        """
        Initialize the label lists based on mask_type.
        
        Args:
            mask_type (str or None): Type of mask to generate.
        """
        if mask_type is None:
            self.mask_labels = (
                self.ALL_FACE_LABELS + self.EAR_LABELS + 
                [14, 17, 18]  # neck, hair, hat
            )
            self.face_labels = (
                self.ALL_FACE_LABELS + self.EAR_LABELS + 
                self.HAIR_LABEL + self.HAT_LABEL
            )
            self.body_labels = self.HAT_LABEL
            self.cloth_labels = self.CLOTH_LABEL
        else:
            self.mask_labels = []
            if 'face' in mask_type:
                self.mask_labels.extend(self.ALL_FACE_LABELS)
            if 'ears' in mask_type:
                self.mask_labels.extend(self.EAR_LABELS)
            if 'neck' in mask_type:
                self.mask_labels.extend(self.NECK_LABELS)
            if 'hair' in mask_type:
                self.mask_labels.extend(self.HAIR_LABEL)
            if 'hat' in mask_type:
                self.mask_labels.extend(self.HAT_LABEL)
            if 'cloth' in mask_type:
                self.mask_labels.extend(self.CLOTH_LABEL)
    
    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass of the face parsing model.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            tuple: (mask, face_body, mask_body, mask_cloth) tensors.
        """
        # Get original dimensions
        h, w = x.shape[2:]
        
        # Normalize and resize input
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        x = F.interpolate(x, size=(512, 512), mode='bilinear')
        
        # Get model prediction and resize back
        y = self.net(x)[0]
        y = F.interpolate(y, size=(h, w), mode='bilinear')
        
        # Generate label masks
        labels = y.argmax(1, keepdim=True)
        
        # Create masks for different parts
        masks = {
            'main': self._create_mask(labels, self.mask_labels),
            'face_body': self._create_mask(labels, self.face_labels),
            'body': self._create_mask(labels, self.body_labels),
            'cloth': self._create_mask(labels, self.cloth_labels)
        }
        
        return masks['main'], masks['face_body'], masks['body'], masks['cloth']
    
    def _create_mask(self, labels, target_labels):
        """
        Create a binary mask for specified labels.
        
        Args:
            labels (torch.Tensor): Predicted labels tensor.
            target_labels (list): List of label indices to include in mask.
            
        Returns:
            torch.Tensor: Binary mask tensor.
        """
        mask = torch.zeros_like(labels)
        for label in target_labels:
            mask += (labels == label)
        return mask