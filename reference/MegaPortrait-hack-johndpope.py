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

"""
from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torchvision.models as models

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
            layers: List[int] = [3, 4, 6, 3],
            n_class: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            is_remix=False
    ) -> None:
        super(ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, n_class)
        self.fc = nn.Linear(512 * block.expansion, 512)  # Reduce to 512 dimensions

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(2048, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
        self._initialize_weights()

    def _initialize_weights(self):
        pretrained_resnet50 = models.resnet50(pretrained=True)
        
        self.conv1.weight.data = pretrained_resnet50.conv1.weight.data.clone()
        self.bn1.weight.data = pretrained_resnet50.bn1.weight.data.clone()
        self.bn1.bias.data = pretrained_resnet50.bn1.bias.data.clone()
        
        for i in range(1, 5):
            layer = getattr(self, f'layer{i}')
            pretrained_layer = getattr(pretrained_resnet50, f'layer{i}')
            self._initialize_layer_weights(layer, pretrained_layer)
        
        # Comment out the following lines if you don't want to copy the FC layer weights
        # self.fc.weight.data = pretrained_resnet50.fc.weight.data.clone()
        # self.fc.bias.data = pretrained_resnet50.fc.bias.data.clone()

    def _initialize_layer_weights(self, layer, pretrained_layer):
        for block, pretrained_block in zip(layer, pretrained_layer):
            block.conv1.weight.data = pretrained_block.conv1.weight.data.clone()
            block.bn1.weight.data = pretrained_block.bn1.weight.data.clone()
            block.bn1.bias.data = pretrained_block.bn1.bias.data.clone()
            block.conv2.weight.data = pretrained_block.conv2.weight.data.clone()
            block.bn2.weight.data = pretrained_block.bn2.weight.data.clone()
            block.bn2.bias.data = pretrained_block.bn2.bias.data.clone()
            if isinstance(block, Bottleneck):
                block.conv3.weight.data = pretrained_block.conv3.weight.data.clone()
                block.bn3.weight.data = pretrained_block.bn3.weight.data.clone()
                block.bn3.bias.data = pretrained_block.bn3.bias.data.clone()
            if block.downsample is not None:
                block.downsample[0].weight.data = pretrained_block.downsample[0].weight.data.clone()
                block.downsample[1].weight.data = pretrained_block.downsample[1].weight.data.clone()
                block.downsample[1].bias.data = pretrained_block.downsample[1].bias.data.clone()


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Reduce to 512 dimensions
        # out = self.fc(x) # Comment out this line if you don't want to use the FC layer
        if self.is_remix:
            rot_output = self.rot_classifier(x)
            return x, rot_output
        else:
            return x

    def forward(self, x):
        return self._forward_impl(x)


class build_ResNet50:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return ResNet50(n_class=num_classes, is_remix=self.is_remix)


if __name__ == '__main__':
    a = torch.rand(16, 3, 224, 224)
    net = ResNet50(is_remix=True)
    x,y = net(a)
    print(x.shape)
    print(y.shape)
from __future__ import annotations
from typing import List, Set, Tuple, Dict

import time
from collections import OrderedDict
from functools import partial
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
# Calculate the product of the scale factors
from functools import reduce
from operator import mul

Tensor = torch.Tensor

module_flop_count = []
module_mac_count = []
module_profile_lists = []
old_functions = {}
cuda_sync=False
@dataclass
class profileEntry:
    flops: int = 0,
    macs: int = 0,
    duration: float = 0.0

class FlopsProfiler:
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = FlopsProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops(as_string=True)
                    params = prof.get_total_params(as_string=True)
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self, model: nn.Module, ds_engine=None):
        self.model = model
        self.ds_engine = ds_engine
        self.started = False
        self.func_patched = False

        global cuda_sync
        cuda_sync = next(model.parameters()).device.type == 'cuda'

    def start_profile(self, ignore_list: list[nn.Module]| None = None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_functionals()
        _patch_tensor_methods()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                if not hasattr(module, '__flops_handle__'):
                    module.__flops_handle__ = module.register_forward_hook(
                        MODULE_HOOK_MAPPING[type(module)],
                    )
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])
                module_profile_lists.append([])

            if not hasattr(module, '__pre_hook_handle__'):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(
                    pre_hook,
                )

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([
                        elem[1]
                        for elem in module_flop_count[-1]
                    ])
                    module_flop_count.pop()
                    module.__macs__ += sum([
                        elem[1]
                        for elem in module_mac_count[-1]
                    ])
                    module_mac_count.pop()
                if module_profile_lists:
                    lst = module_profile_lists.pop()
                    for name, entry in lst:
                        module.__profile_table__[name].flops += entry.flops
                        module.__profile_table__[name].macs += entry.macs
                        module.__profile_table__[name].duration += entry.duration

            if not hasattr(module, '__post_hook_handle__'):
                module.__post_hook_handle__ = module.register_forward_hook(
                    post_hook,
                )

            def start_time_hook(module, input):
                if cuda_sync:
                    torch.cuda.synchronize()
                module.__start_time__ = time.time()

            if not hasattr(module, '__start_time_hook_handle'):
                module.__start_time_hook_handle__ = module.register_forward_pre_hook(
                    start_time_hook,
                )

            def end_time_hook(module, input, output):
                if cuda_sync:
                    torch.cuda.synchronize()
                module.__duration__ += time.time() - module.__start_time__

            if not hasattr(module, '__end_time_hook_handle__'):
                module.__end_time_hook_handle__ = module.register_forward_hook(
                    end_time_hook,
                )

        self.model.apply(
            partial(register_module_hooks, ignore_list=ignore_list),
        )
        self.started = True
        self.func_patched = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, '__pre_hook_handle__'):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, '__post_hook_handle__'):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, '__flops_handle__'):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, '__start_time_hook_handle__'):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, '__end_time_hook_handle__'):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__

        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())
            module.__start_time__ = 0
            module.__duration__ = 0
            module.__profile_table__ = defaultdict(lambda: profileEntry(0, 0, 0.0))

        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if hasattr(module, '__flops__'):
                del module.__flops__
            if hasattr(module, '__macs__'):
                del module.__macs__
            if hasattr(module, '__params__'):
                del module.__params__
            if hasattr(module, '__start_time__'):
                del module.__start_time__
            if hasattr(module, '__duration__'):
                del module.__duration__
            if hasattr(module, '__profile_table__'):
                del module.__profile_table__

        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string: bool = False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = _get_module_flops(self.model)
        return _num_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string: bool = False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = _get_module_macs(self.model)
        return _macs_to_string(total_macs) if as_string else total_macs

    def get_total_duration(self, as_string: bool = False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The latency of the model forward pass.
        """
        total_duration = _get_module_duration(self.model)
        return _duration_to_string(total_duration) if as_string else total_duration

    def get_total_functional_duration(self, as_string: bool = False):
        """Returns the total duration of the nn.functional calls in the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The total latency of the nn.functional calls.
        """
        table = _get_module_profile_table(self.model)
        total_duration = sum([table[func_name].duration for func_name in table])
        return _duration_to_string(total_duration) if as_string else total_duration

    def get_total_params(self, as_string: bool = False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return _params_to_string(
            self.model.__params__,
        ) if as_string else self.model.__params__

    def print_model_profile(
        self,
        profile_step: int = 1,
        module_depth: int = -1,
        top_modules: int =1,
        detailed: bool = True,
        output_file: str|None = None,
    ):
        """Prints the model graph with the measured profile attached to each module.

        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
        """
        if not self.started:
            return
        import os.path
        import sys
        original_stdout = None
        f = None
        if output_file and output_file != '':
            dir_path = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, 'w')
            sys.stdout = f

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_functation_duration = self.get_total_functional_duration()
        total_params = self.get_total_params()

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        print(
            '\n-------------------------- Flops Profiler --------------------------',
        )
        print(f'Profile on Device: {next(self.model.parameters()).device}')
        print(f'Profile Summary at step {profile_step}:')
        print(
            'Notations:\ndata parallel size (dp_size), model parallel size(mp_size),\nnumber of parameters (params), number of multiply-accumulate operations(MACs),\nnumber of floating-point operations (flops), floating-point operations per second (FLOPS),\nfwd latency (forward propagation latency), bwd latency (backward propagation latency),\nstep (weights update latency), iter latency (sum of fwd, bwd and step latency)\n',
        )
        if self.ds_engine:
            print(
                '{:<60}  {:<8}'.format(
                    'world size: ', self.ds_engine.world_size,
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'data parallel size: ',
                    self.ds_engine.dp_world_size,
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'model parallel size: ',
                    self.ds_engine.mp_world_size,
                ),
            )
            if self.ds_engine and hasattr(self.ds_engine, 'train_micro_batch_size_per_gpu'):
                print(
                    '{:<60}  {:<8}'.format(
                        'batch size per device: ',
                        self.ds_engine.train_micro_batch_size_per_gpu(),
                    ),
                )

        print(
            '{:<60}  {:<8}'.format(
                'params per device: ', _params_to_string(total_params),
            ),
        )
        print(
            '{:<60}  {:<8}'.format(
                'params of model = params per device * mp_size: ',
                _params_to_string(
                    total_params *
                    ((self.ds_engine.mp_world_size) if self.ds_engine else 1),
                ),
            ),
        )

        print(
            '{:<60}  {:<8}'.format(
                'fwd MACs per device: ', _macs_to_string(total_macs),
            ),
        )

        print(
            '{:<60}  {:<8}'.format(
                'fwd flops per device: ', _num_to_string(total_flops),
            ),
        )

        print(
            '{:<60}  {:<8}'.format(
                'fwd flops of model = fwd flops per device * mp_size: ',
                _num_to_string(
                    total_flops *
                    ((self.ds_engine.mp_world_size) if self.ds_engine else 1),
                ),
            ),
        )

        fwd_latency = self.get_total_duration()
        if fwd_latency:
            if self.ds_engine and hasattr(self.ds_engine, 'wall_clock_breakdown') and self.ds_engine.wall_clock_breakdown():
                fwd_latency = self.ds_engine.timers(
                    'forward',
                ).elapsed(False) / 1000.0
            print(
                '{:<60}  {:<8}'.format(
                    'fwd latency: ',
                    _duration_to_string(fwd_latency),
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'fwd FLOPS per device = fwd flops per device / fwd latency: ',
                    _flops_to_string(total_flops / fwd_latency),
                ),
            )

        if self.ds_engine and hasattr(self.ds_engine, 'wall_clock_breakdown') and self.ds_engine.wall_clock_breakdown():
            bwd_latency = self.ds_engine.timers(
                'backward',
            ).elapsed(False) / 1000.0
            step_latency = self.ds_engine.timers(
                'step',
            ).elapsed(False) / 1000.0
            print(
                '{:<60}  {:<8}'.format(
                    'bwd latency: ',
                    _duration_to_string(bwd_latency),
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'bwd FLOPS per device = 2 * fwd flops per device / bwd latency: ',
                    _flops_to_string(2 * total_flops / bwd_latency),
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'fwd+bwd FLOPS per device = 3 * fwd flops per device / (fwd+bwd latency): ',
                    _flops_to_string(
                        3 * total_flops /
                        (fwd_latency + bwd_latency),
                    ),
                ),
            )

            print(
                '{:<60}  {:<8}'.format(
                    'step latency: ',
                    _duration_to_string(step_latency),
                ),
            )

            iter_latency = fwd_latency + bwd_latency + step_latency
            print(
                '{:<60}  {:<8}'.format(
                    'iter latency: ',
                    _duration_to_string(iter_latency),
                ),
            )
            print(
                '{:<60}  {:<8}'.format(
                    'FLOPS per device = 3 * fwd flops per device / iter latency: ',
                    _flops_to_string(3 * total_flops / iter_latency),
                ),
            )

            if self.ds_engine and hasattr(self.ds_engine, 'train_micro_batch_size_per_gpu'):
                samples_per_iter = self.ds_engine.train_micro_batch_size_per_gpu(
                ) * self.ds_engine.world_size
                print(
                    '{:<60}  {:<8.2f}'.format(
                        'samples/second: ',
                        samples_per_iter / iter_latency,
                    ),
                )

        def flops_repr(module):
            mod_params = module.__params__
            mod_flops = _get_module_flops(module)
            mod_macs = _get_module_macs(module)
            mod_duration = _get_module_duration(module)
            items = []

            mod_profile_str = str('module = ')
            mod_profile = {}
            mod_profile.update({
                'param': _params_to_string(mod_params),
                'flops': _num_to_string(mod_flops),
                'macs': _macs_to_string(mod_macs),
                'duration': _duration_to_string(mod_duration),
                'FLOPS': _flops_to_string(mod_flops / mod_duration if mod_duration else 0.0),
                'params%': f'{mod_params / total_params if total_params else 0:.2%}',
                'flops%': f'{mod_flops / total_flops if total_flops else 0:.2%}',
                'macs%': f'{mod_macs / total_macs if total_macs else 0:.2%}',
                'duration%': f'{mod_duration / total_duration if total_duration else 0:.2%}',
                })
            mod_profile_str += f'{str(mod_profile)}'
            items.append(mod_profile_str)

            func_profile_table = _get_module_profile_table(module)
            func_profile_str = str('functionals = ')
            func_profile = {}
            f, m, d = 0, 0, 0.0
            for name, entry in func_profile_table.items():
                f += entry.flops
                m += entry.macs
                d += entry.duration
                func_profile[name] = asdict(entry)
                func_profile[name].update({
                    'flops': _num_to_string(entry.flops),
                    'macs': _macs_to_string(entry.macs),
                    'duration': _duration_to_string(entry.duration),
                    'FLOPS': _flops_to_string(entry.flops / entry.duration if entry.duration else 0.0),
                    'flops%': f'{entry.flops / total_flops if total_flops else 0:.2%}',
                    'macs%': f'{entry.macs / total_macs if total_macs else 0:.2%}',
                    'duration%/allfuncs': f'{entry.duration / total_functation_duration if total_functation_duration else 0:.2%}',
                    'duration%/e2e': f'{entry.duration / total_duration if total_duration else 0:.2%}',
                })
            func_profile_str += str(func_profile)
            assert f == mod_flops, f'module total flops =! functional flops {f} != {mod_flops}'
            assert m == mod_macs, f'module total macs =! functional macs{m} != {mod_macs}'
            items.append(func_profile_str)
            items.append(f'functionals_duration = {_duration_to_string(d)}')

            items.append(module.original_extra_repr())

            return ', '.join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, 'original_extra_repr'):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        print(
            '\n----------------------------- Aggregated Profile per Device -----------------------------',
        )
        self.print_model_aggregated_profile(
            module_depth=module_depth,
            top_modules=top_modules,
        )

        if detailed:
            print(
                '\n------------------------------ Detailed Profile per Device ------------------------------',
            )
            print(
                'Each module profile is listed after its name in the following order: \nparams, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS',
            )
            print(
                "\nNote: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.\n2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.\n3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch.\n",
            )
            print(self.model)

        self.model.apply(del_extra_repr)

        print(
            '------------------------------------------------------------------------------',
        )

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth: int = -1, top_modules: int = 1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 1.
        """
        info = {}
        if not hasattr(self.model, '__flops__'):
            print(
                'no __flops__ attribute in the model, call this function after start_profile and before end_profile',
            )
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                    0,
                ]  # macsparams, time
            info[curr_depth][module.__class__.__name__][0] += module.__params__
            info[curr_depth][module.__class__.__name__][1] += _get_module_flops(
                module,
            )
            info[curr_depth][module.__class__.__name__][2] += _get_module_macs(
                module,
            )
            info[curr_depth][module.__class__.__name__][3] += _get_module_duration(
                module,
            )
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)

        walk_module(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1

        print(
            f'Top {top_modules} modules in terms of params, flops, MACs or duration at different model depths:',
        )

        for d in range(depth):
            num_items = min(top_modules, len(info[d]))
            sort_params = {
                    k: _params_to_string(v[0])
                    for k,
                    v in sorted(
                        info[d].items(),
                        key=lambda item: item[1][0],
                        reverse=True,
                    )[:num_items]
                }
            sort_flops = {
                k: _num_to_string(v[1])
                for k,
                v in sorted(
                    info[d].items(),
                    key=lambda item: item[1][1],
                    reverse=True,
                )[:num_items]
            }
            sort_macs = {
                k: _macs_to_string(v[2])
                for k,
                v in sorted(
                    info[d].items(),
                    key=lambda item: item[1][2],
                    reverse=True,
                )[:num_items]
            }

            sort_time = {
                k: _duration_to_string(v[3])
                for k,
                v in sorted(
                    info[d].items(),
                    key=lambda item: item[1][3],
                    reverse=True,
                )[:num_items]
            }

            print(f'depth {d}:')
            print(f'    params      - {sort_params}')
            print(f'    flops       - {sort_flops}')
            print(f'    MACs        - {sort_macs}')
            print(f'    fwd latency - {sort_time}')


def _prod(dims: int):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs


def _relu_flops_compute(input, inplace=False):
    return input.numel(), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel(), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return input.numel(), 0


def _leaky_relu_flops_compute(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
):
    return input.numel(), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel(), 0


def _gelu_flops_compute(input, **kwargs):
    return input.numel(), 0


def _pool_flops_compute(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=None,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    return_indices=None,
):
    return input.numel(), 0


def _conv_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (
            input_dim + 2 * paddings[idx] -
            (dilations[idx] * (kernel_dims[idx] - 1) + 1)
        ) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(
        _prod(kernel_dims),
    ) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (
            input_dim + 2 * paddings[idx] -
            (dilations[idx] * (kernel_dims[idx] - 1) + 1)
        ) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // grofups
    conv_per_position_macs = int(
        _prod(kernel_dims),
    ) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _group_norm_flops_compute(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _upsample_flops_compute(input, **kwargs):
    size = kwargs.get('size', None)
    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    scale_factor = kwargs.get('scale_factor', None)
    assert scale_factor is not None, 'either size or scale_factor should be defined'
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops, 0


def _interpolate_flops_compute(input, **kwargs):
    size = kwargs.get('size', None)
    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    scale_factor = kwargs.get('scale_factor', None)
    assert scale_factor is not None, 'either size or scale_factor should be defined'
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        if isinstance(scale_factor, tuple):
            scale_factor_product = reduce(mul, scale_factor)
        else:
            scale_factor_product = scale_factor
        
        flops *= scale_factor_product ** len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel(), 0


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(' ', '')
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize='optimal')[1]
    for line in optim.split('\n'):
        if 'optimized flop' in line.lower():
            flop = int(float(line.split(':')[-1]))
            return flop, 0
    raise NotImplementedError('Unsupported einsum operation.')


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def _wrapFunc(func, funcFlopCompute):
    global cuda_sync
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc
    func_name = func.__name__
    def newFunc(*args, **kwds):
        # print(f"args:{args}")
        # print(f"kwds:{kwds}")
        
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        if cuda_sync:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        else:
            start = time.time()
        ret = oldFunc(*args, **kwds)
        if cuda_sync:
            cuda_end.record()
            torch.cuda.synchronize()
            duration = cuda_start.elapsed_time(cuda_end)/1000
        else:
            duration = time.time() - start
        if module_profile_lists:
            module_profile_lists[-1].append((func_name, profileEntry(flops, macs, duration)))
        return ret

    newFunc.__str__ = func.__str__

    return newFunc


def _patch_functionals():
    # FC
    F.linear = _wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = _wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = _wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = _wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = _wrapFunc(
        F.conv_transpose1d, _conv_trans_flops_compute,
    )
    F.conv_transpose2d = _wrapFunc(
        F.conv_transpose2d, _conv_trans_flops_compute,
    )
    F.conv_transpose3d = _wrapFunc(
        F.conv_transpose3d, _conv_trans_flops_compute,
    )

    # activations
    F.relu = _wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = _wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = _wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = _wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = _wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, 'silu'):
        F.silu = _wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = _wrapFunc(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = _wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = _wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = _wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = _wrapFunc(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = _wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = _wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = _wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = _wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = _wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = _wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = _wrapFunc(
        F.adaptive_avg_pool1d, _pool_flops_compute,
    )
    F.adaptive_avg_pool2d = _wrapFunc(
        F.adaptive_avg_pool2d, _pool_flops_compute,
    )
    F.adaptive_avg_pool3d = _wrapFunc(
        F.adaptive_avg_pool3d, _pool_flops_compute,
    )
    F.adaptive_max_pool1d = _wrapFunc(
        F.adaptive_max_pool1d, _pool_flops_compute,
    )
    F.adaptive_max_pool2d = _wrapFunc(
        F.adaptive_max_pool2d, _pool_flops_compute,
    )
    F.adaptive_max_pool3d = _wrapFunc(
        F.adaptive_max_pool3d, _pool_flops_compute,
    )

    # upsample
    F.upsample = _wrapFunc(F.upsample, _upsample_flops_compute)
    # F.interpolate = _wrapFunc(F.interpolate, _interpolate_flops_compute) - has problem https://github.com/cli99/flops-profiler/issues/13

    # softmax
    F.softmax = _wrapFunc(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = _wrapFunc(F.embedding, _embedding_flops_compute)


def _patch_tensor_methods():
    torch.matmul = _wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = _wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = _wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = _wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = _wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = _wrapFunc(torch.Tensor.bmm, _matmul_flops_compute)

    torch.addmm = _wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = _wrapFunc(
        torch.Tensor.addmm, _tensor_addmm_flops_compute,
    )

    torch.mul = _wrapFunc(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = _wrapFunc(torch.Tensor.mul, _mul_flops_compute)

    torch.add = _wrapFunc(torch.add, _add_flops_compute)
    torch.Tensor.add = _wrapFunc(torch.Tensor.add, _add_flops_compute)

    torch.einsum = _wrapFunc(torch.einsum, _einsum_flops_compute)

    torch.baddbmm = _wrapFunc(torch.baddbmm, _tensor_addmm_flops_compute)
    torch.Tensor.baddbmm = _wrapFunc(torch.Tensor.baddbmm, _tensor_addmm_flops_compute)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, 'silu'):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    # F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.embedding = old_functions[F.embedding.__str__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    torch.mm = old_functions[torch.mm.__str__]
    torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    torch.bmm = old_functions[torch.matmul.__str__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]

    torch.einsum = old_functions[torch.einsum.__str__]

    torch.baddbmm = old_functions[torch.baddbmm.__str__]
    torch.Tensor.baddbmm = old_functions[torch.Tensor.baddbmm.__str__]

def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}


def _num_to_string(num, precision=2):
    if num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + ' G'
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + ' M'
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + ' K'
    else:
        return str(num)


def _macs_to_string(macs, units=None, precision=2):
    if units is None:
        if macs // 10**9 > 0:
            return str(round(macs / 10.0**9, precision)) + ' GMACs'
        elif macs // 10**6 > 0:
            return str(round(macs / 10.0**6, precision)) + ' MMACs'
        elif macs // 10**3 > 0:
            return str(round(macs / 10.0**3, precision)) + ' KMACs'
        else:
            return str(macs) + ' MACs'
    else:
        if units == 'GMACs':
            return str(round(macs / 10.0**9, precision)) + ' ' + units
        elif units == 'MMACs':
            return str(round(macs / 10.0**6, precision)) + ' ' + units
        elif units == 'KMACs':
            return str(round(macs / 10.0**3, precision)) + ' ' + units
        else:
            return str(macs) + ' MACs'


def _number_to_string(num, units=None, precision=2):
    if units is None:
        if num // 10**9 > 0:
            return str(round(num / 10.0**9, precision)) + ' G'
        elif num // 10**6 > 0:
            return str(round(num / 10.0**6, precision)) + ' M'
        elif num // 10**3 > 0:
            return str(round(num / 10.0**3, precision)) + ' K'
        else:
            return str(num) + ' '
    else:
        if units == 'G':
            return str(round(num / 10.0**9, precision)) + ' ' + units
        elif units == 'M':
            return str(round(num / 10.0**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num / 10.0**3, precision)) + ' ' + units
        else:
            return str(num) + ' '


def _flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**12 > 0:
            return str(round(flops / 10.0**12, precision)) + ' TFLOPS'
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + ' GFLOPS'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + ' MFLOPS'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + ' KFLOPS'
        else:
            return str(flops) + ' FLOPS'
    else:
        if units == 'TFLOPS':
            return str(round(flops / 10.0**12, precision)) + ' ' + units
        if units == 'GFLOPS':
            return str(round(flops / 10.0**9, precision)) + ' ' + units
        elif units == 'MFLOPS':
            return str(round(flops / 10.0**6, precision)) + ' ' + units
        elif units == 'KFLOPS':
            return str(round(flops / 10.0**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPS'


def _params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + ' M'
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.0**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.0**3, precision)) + ' ' + units
        else:
            return str(params_num)


def _duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + ' s'
        elif duration * 10**3 > 1:
            return str(round(duration * 10**3, precision)) + ' ms'
        elif duration * 10**6 > 1:
            return str(round(duration * 10**6, precision)) + ' us'
        else:
            return str(duration)
    else:
        if units == 'us':
            return str(round(duration * 10.0**6, precision)) + ' ' + units
        elif units == 'ms':
            return str(round(duration * 10.0**3, precision)) + ' ' + units
        else:
            return str(round(duration, precision)) + ' s'

    # can not iterate over all submodules using self.model.modules()
    # since modules() returns duplicate modules only once


def _get_module_flops(module: nn.Module):
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += _get_module_flops(child)
    return sum


def _get_module_macs(module: nn.Module):
    sum = module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum += _get_module_macs(child)
    return sum


def _get_module_duration(module: nn.Module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for m in module.children():
            duration += m.__duration__
    return duration

def _get_module_profile_table(module: nn.Module):
    sum_table = module.__profile_table__

    # TODO: tmp fix for duplicated counting
    if hasattr(module, '__cnt__'):
        module.__cnt__ += 1
    else:
        module.__cnt__ = 1
    if module.__cnt__ > 1:
        return sum_table

    # iterate over immediate children modules
    for child in module.children():
        table = _get_module_profile_table(child)
        for name, entry in table.items():
            sum_table[name].flops += entry.flops
            sum_table[name].macs += entry.macs
            sum_table[name].duration += entry.duration
    return sum_table


def get_model_profile(
    model: nn.Module,
    input_shape: tuple|None =None,
    args= [],
    kwargs={},
    print_profile: bool = True,
    detailed: bool = True,
    module_depth: int = -1,
    top_modules: int = 10,
    warm_up: int = 3,
    as_string: bool = False,
    output_file: str|None = None,
    ignore_modules: List[nn.Module]|None = None,
    func_name: str = 'forward',
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), 'model must be a PyTorch module'
    prof = FlopsProfiler(model)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, 'input_shape must be a tuple'
        assert len(input_shape) >= 1, 'input_shape must have at least one element'
        try:
            input = torch.ones(()).new_empty(
                (
                    *input_shape,
                ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape,))

        args = [input]
    assert (len(args) > 0) or (
        len(kwargs) >
        0
    ), 'args and/or kwargs must be specified if input_shape is None'

    for _ in range(warm_up):
        func = getattr(model, func_name)
        if kwargs:
            _ = func(*args, **kwargs)
        else:
            _ = func(*args)
    prof.start_profile(ignore_list=ignore_modules)

    func = getattr(model, func_name)
    if kwargs:
        _ = func(*args, **kwargs)
    else:
        _ = func(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(
            profile_step=warm_up,
            module_depth=module_depth,
            top_modules=top_modules,
            detailed=detailed,
            output_file=output_file,
        )

    prof.end_profile()
    if as_string:
        return _number_to_string(flops), _macs_to_string(macs), _params_to_string(params)

    return flops, macs, params

# not convinced we need to train this - see metaportrait Super Resolution model
# https://github.com/Meta-Portrait/MetaPortrait/tree/main/sr_model
import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
import decord
from omegaconf import OmegaConf
from torchvision import models
from model import MPGazeLoss,Encoder
from rome_losses import Vgg19 # use vgg19 for perceptualloss 
import cv2
import mediapipe as mp
from memory_profiler import profile
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image



# Create a directory to save the images (if it doesn't already exist)
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)


face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.autograd.set_detect_anomaly(True)# this slows thing down - only for debug



'''
We load the pre-trained DeepLabV3 model using models.segmentation.deeplabv3_resnet101(pretrained=True). This model is based on the ResNet-101 backbone and is pre-trained on the COCO dataset.
We define the necessary image transformations using transforms.Compose. The transformations include converting the image to a tensor and normalizing it using the mean and standard deviation values specific to the model.
We apply the transformations to the input image using transform(image) and add an extra dimension to represent the batch size using unsqueeze(0).
We move the input tensor to the same device as the model to ensure compatibility.
We perform the segmentation by passing the input tensor through the model using model(input_tensor). The output is a dictionary containing the segmentation map.
We obtain the predicted segmentation mask by taking the argmax of the output along the channel dimension using torch.max(output['out'], dim=1).
We convert the segmentation mask to a binary foreground mask by comparing the predicted class labels with the class index representing the person class (assuming it is 15 in this example). The resulting mask will have values of 1 for foreground pixels and 0 for background pixels.
Finally, we return the foreground mask.
'''

def get_foreground_mask(image):
    # Load the pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the input image
    input_tensor = transform(image).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted segmentation mask
    _, mask = torch.max(output['out'], dim=1)

    # Convert the segmentation mask to a binary foreground mask
    foreground_mask = (mask == 15).float()  # Assuming class 15 represents the person class

    return foreground_mask


'''
Perceptual Loss:

The PerceptualLoss class combines losses from VGG19, VGG Face, and a specialized gaze loss.
It computes the perceptual losses by passing the output and target frames through the respective models and calculating the MSE loss between the features.
The total perceptual loss is a weighted sum of the individual losses.


Adversarial Loss:

The adversarial_loss function computes the adversarial loss for the generator.
It passes the generated output frame through the discriminator and calculates the MSE loss between the predicted values and a tensor of ones (indicating real samples).


Cycle Consistency Loss:

The cycle_consistency_loss function computes the cycle consistency loss.
It passes the output frame and the source frame through the generator to reconstruct the source frame.
The L1 loss is calculated between the reconstructed source frame and the original source frame.


Contrastive Loss:

The contrastive_loss function computes the contrastive loss using cosine similarity.
It calculates the cosine similarity between positive pairs (output-source, output-driving) and negative pairs (output-random, source-random).
The loss is computed as the negative log likelihood of the positive pairs over the sum of positive and negative pair similarities.
The neg_pair_loss function calculates the loss for negative pairs using a margin.


Discriminator Loss:

The discriminator_loss function computes the loss for the discriminator.
It calculates the MSE loss between the predicted values for real samples and a tensor of ones, and the MSE loss between the predicted values for fake samples and a tensor of zeros.
The total discriminator loss is the sum of the real and fake losses.
'''

# @profile
def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss.requires_grad_()

# @profile
def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss.requires_grad_()


def contrastive_loss(output_frame, source_frame, driving_frame, encoder, margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out, requires_grad=True)

    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]

    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for pos_pair in pos_pairs:
        loss = loss + torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) /
                                (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) +
                                 neg_pair_loss(pos_pair, neg_pairs, margin)))

    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for neg_pair in neg_pairs:
        loss = loss + torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss
# @profile
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss).requires_grad_()


# @profile
def gaze_loss_fn(predicted_gaze, target_gaze, face_image):
    # Ensure face_image has shape (C, H, W)
    if face_image.dim() == 4 and face_image.shape[0] == 1:
        face_image = face_image.squeeze(0)
    if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
        raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
    
    # Convert face image from tensor to numpy array
    face_image = face_image.detach().cpu().numpy()
    if face_image.shape[0] == 3:  # if channels are first
        face_image = face_image.transpose(1, 2, 0)
    face_image = (face_image * 255).astype(np.uint8)

    # Extract eye landmarks using MediaPipe
    results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return torch.tensor(0.0, requires_grad=True).to(device)

    eye_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
        eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

    # Compute loss for each eye
    loss = 0.0
    h, w = face_image.shape[:2]
    for left_eye, right_eye in eye_landmarks:
        # Convert landmarks to pixel coordinates
        left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
        right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

        # Create eye mask
        left_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        right_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
        cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

        # Compute gaze loss for each eye
        left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
        right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
        loss += left_gaze_loss + right_gaze_loss

    return loss / len(eye_landmarks)


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device)
    encoder = Encoder(input_nc=3, output_nc=256).to(device)

    for epoch in range(cfg.training.base_epochs):
        print("epoch:", epoch)
        for batch in dataloader:
            source_frames = batch['source_frames'] #.to(device)
            driving_frames = batch['driving_frames'] #.to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx].to(device)
                driving_frame = driving_frames[idx].to(device)

                # Train generator
                optimizer_G.zero_grad()
                output_frame = Gbase(source_frame, driving_frame)

                # Resize output_frame to 256x256 to match the driving_frame size
                output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)


                # 💀 Compute losses -  "losses are calculated using ONLY foreground regions" 
                # Obtain the foreground mask for the target image
                foreground_mask = get_foreground_mask(source_frame)

                # Multiply the predicted and target images with the foreground mask
                masked_predicted_image = output_frame * foreground_mask
                masked_target_image = source_frame * foreground_mask


                output_vgg_features = vgg19(masked_predicted_image)
                driving_vgg_features = vgg19(masked_target_image)  
                total_loss = 0

                for output_feat, driving_feat in zip(output_vgg_features, driving_vgg_features):
                    total_loss = total_loss + perceptual_loss_fn(output_feat, driving_feat.detach())

                loss_adversarial = adversarial_loss(masked_predicted_image, Dbase)

                loss_gaze = gaze_loss_fn(output_frame, driving_frame, source_frame) # 🤷 fix this
                # Combine the losses and perform backpropagation and optimization
                total_loss = total_loss + loss_adversarial + loss_gaze

                
                # Accumulate gradients
                loss_gaze.backward()
                total_loss.backward(retain_graph=True)
                loss_adversarial.backward()

                # Update generator
                optimizer_G.step()

                # Train discriminator
                optimizer_D.zero_grad()
                real_pred = Dbase(driving_frame)
                fake_pred = Dbase(output_frame.detach())
                loss_D = discriminator_loss(real_pred, fake_pred)

                # Backpropagate and update discriminator
                loss_D.backward()
                optimizer_D.step()


        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_gaze.item():.4f}, Loss_D: {loss_D.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def train_hr(cfg, GHR, Genh, dataloader_hr):
    GHR.train()
    Genh.train()

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device=device)

    optimizer_G = torch.optim.AdamW(Genh.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.hr_epochs, eta_min=1e-6)

    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            source_frames = batch['source_frames'].to(device)
            driving_frames = batch['driving_frames'].to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx]
                driving_frame = driving_frames[idx]

                # Generate output frame using pre-trained base model
                with torch.no_grad():
                    xhat_base = GHR.Gbase(source_frame, driving_frame)

                # Train high-resolution model
                optimizer_G.zero_grad()
                xhat_hr = Genh(xhat_base)


                # Compute losses - option 1
                # loss_supervised = Genh.supervised_loss(xhat_hr, driving_frame)
                # loss_unsupervised = Genh.unsupervised_loss(xhat_base, xhat_hr)
                # loss_perceptual = perceptual_loss_fn(xhat_hr, driving_frame)

                # option2 ? 🤷 use vgg19 as per metaportrait?
                # - Compute losses
                xhat_hr_vgg_features = vgg19(xhat_hr)
                driving_vgg_features = vgg19(driving_frame)
                loss_perceptual = 0
                for xhat_hr_feat, driving_feat in zip(xhat_hr_vgg_features, driving_vgg_features):
                    loss_perceptual += perceptual_loss_fn(xhat_hr_feat, driving_feat.detach())

                loss_supervised = perceptual_loss_fn(xhat_hr, driving_frame)
                loss_unsupervised = perceptual_loss_fn(xhat_hr, xhat_base)
                loss_gaze = gaze_loss_fn(xhat_hr, driving_frame)
                loss_G = (
                    cfg.training.lambda_supervised * loss_supervised
                    + cfg.training.lambda_unsupervised * loss_unsupervised
                    + cfg.training.lambda_perceptual * loss_perceptual
                    + cfg.training.lambda_gaze * loss_gaze
                )

                # Backpropagate and update high-resolution model
                loss_G.backward()
                optimizer_G.step()

        # Update learning rate
        scheduler_G.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.hr_epochs}], "
                  f"Loss_G: {loss_G.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Genh.state_dict(), f"Genh_epoch{epoch+1}.pth")


def train_student(cfg, Student, GHR, dataloader_avatars):
    Student.train()
    
    optimizer_S = torch.optim.AdamW(Student.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_S = CosineAnnealingLR(optimizer_S, T_max=cfg.training.student_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            avatar_indices = batch['avatar_indices'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            
            # Generate high-resolution output frames using pre-trained HR model
            with torch.no_grad():
                xhat_hr = GHR(driving_frames)
            
            # Train student model
            optimizer_S.zero_grad()
            
            # Generate output frames using student model
            xhat_student = Student(driving_frames, avatar_indices)
            
            # Compute loss
            loss_S = F.mse_loss(xhat_student, xhat_hr)
            
            # Backpropagate and update student model
            loss_S.backward()
            optimizer_S.step()
        
        # Update learning rate
        scheduler_S.step()
        
        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.student_epochs}], "
                  f"Loss_S: {loss_S.item():.4f}")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Student.state_dict(), f"Student_epoch{epoch+1}.pth")

def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter() # as augmentation for both source and target images, we use color jitter and random flip
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    Gbase = model.Gbase()
    Dbase = model.Discriminator(input_nc=3).to(device) # 🤷
    
    train_base(cfg, Gbase, Dbase, dataloader)
    
    GHR = model.GHR()
    GHR.Gbase.load_state_dict(Gbase.state_dict())
    Dhr = model.Discriminator(input_nc=3).to(device) # 🤷
    train_hr(cfg, GHR, Dhr, dataloader)
    
    Student = model.Student(num_avatars=100) # this should equal the number of celebs in dataset
    train_student(cfg, Student, GHR, dataloader)
    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(GHR.state_dict(), 'GHR.pth')
    torch.save(Student.state_dict(), 'Student.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,Bottleneck, resnet18
import torchvision.models as models
import math
import colored_traceback.auto
from torchsummary import summary
from resnet50 import ResNet50
from memory_profiler import profile
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,Bottleneck, resnet18
import torchvision.models as models
import math
import colored_traceback.auto
from torchsummary import summary
from resnet50 import ResNet50
from memory_profiler import profile
import logging
import cv2
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
from lpips import LPIPS


from mysixdrepnet import SixDRepNet_Detector
# Set this flag to True for DEBUG mode, False for INFO mode
debug_mode = False

# Configure logging
if debug_mode:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# keep the code in one mega class for copying and pasting into Claude.ai
FEATURE_SIZE_AVG_POOL = 2 # use 2 - not 4. https://github.com/johndpope/MegaPortrait-hack/issues/23
FEATURE_SIZE = (2, 2) 
COMPRESS_DIM = 512 # 🤷 TODO 1: maybe 256 or 512, 512 may be more reasonable for Emtn/app compression

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
        super().__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv2d_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
    
#    @profile
    def forward(self, x):
        logging.debug(f"ResBlock_Custom > x.shape:  %s",x.shape)
        # logging.debug(f"x:",x)
        
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        # Assertions for shape and values
        assert output.shape[1] == self.out_channels, f"Expected {self.out_channels} channels, got {output.shape[1]}"
        assert output.shape[2] == x.shape[2] and output.shape[3] == x.shape[3], \
            f"Expected spatial dimensions {(x.shape[2], x.shape[3])}, got {(output.shape[2], output.shape[3])}"

        return output



# we need custom resnet blocks - so use the ResNet50  es.shape: torch.Size([1, 512, 1, 1])
# n.b. emoportraits reduced this from 512 -> 128 dim - these are feature maps / identity fingerprint of image 
class CustomResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        resnet = models.resnet50(*args, **kwargs)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
      #  self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # Remove the last residual block (layer4)
        # self.layer4 = resnet.layer4
        
        # Add an adaptive average pooling layer
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE_AVG_POOL)
        
        # Add a 1x1 convolutional layer to reduce the number of channels to 512
        self.conv_reduce = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Remove the forward pass through layer4
        # x = self.layer4(x)
        
        # Apply adaptive average pooling
        x = self.adaptive_avg_pool(x)
        
        # Apply the 1x1 convolutional layer to reduce the number of channels
        x = self.conv_reduce(x)
        
        return x

'''
Eapp Class:

The Eapp class represents the appearance encoder (Eapp) in the diagram.
It consists of two parts: producing volumetric features (vs) and producing a global descriptor (es).

Producing Volumetric Features (vs):

The conv layer corresponds to the 7x7-Conv-64 block in the diagram.
The resblock_128, resblock_256, resblock_512 layers correspond to the ResBlock2D-128, ResBlock2D-256, ResBlock2D-512 blocks respectively, with average pooling (self.avgpool) in between.
The conv_1 layer corresponds to the GN, ReLU, 1x1-Conv2D-1536 block in the diagram.
The output of conv_1 is reshaped to (batch_size, 96, 16, height, width) and passed through resblock3D_96 and resblock3D_96_2, which correspond to the two ResBlock3D-96 blocks in the diagram.
The final output of this part is the volumetric features (vs).

Producing Global Descriptor (es):

The resnet50 layer corresponds to the ResNet50 block in the diagram.
It takes the input image (x) and produces the global descriptor (es).

Forward Pass:

During the forward pass, the input image (x) is passed through both parts of the Eapp network.
The first part produces the volumetric features (vs) by passing the input through the convolutional layers, residual blocks, and reshaping operations.
The second part produces the global descriptor (es) by passing the input through the ResNet50 network.
The Eapp network returns both vs and es as output.

In summary, the Eapp class in the code aligns well with the appearance encoder (Eapp) shown in the diagram. The network architecture follows the same structure, with the corresponding layers and blocks mapped accurately. The conv, resblock_128, resblock_256, resblock_512, conv_1, resblock3D_96, and resblock3D_96_2 layers in the code correspond to the respective blocks in the diagram for producing volumetric features. The resnet50 layer in the code corresponds to the ResNet50 block in the diagram for producing the global descriptor.
'''


class Eapp(nn.Module):
    def __init__(self):
        super().__init__()
        
          # First part: producing volumetric features vs
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3).to(device)
        self.resblock_128 = ResBlock_Custom(dimension=2, in_channels=64, out_channels=128).to(device)
        self.resblock_256 = ResBlock_Custom(dimension=2, in_channels=128, out_channels=256).to(device)
        self.resblock_512 = ResBlock_Custom(dimension=2, in_channels=256, out_channels=512).to(device)

        # round 0
        self.resblock3D_96 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 1
        self.resblock3D_96_1 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_1_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 2
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0).to(device)

        # Adjusted AvgPool to reduce spatial dimensions effectively
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0).to(device)

        # Second part: producing global descriptor es
        self.custom_resnet50 = CustomResNet50().to(device)
        '''
        ### TODO 2: Change vs/es here for vector size
        According to the description of the paper (Page11: predict the head pose and expression vector), 
        zs should be a global descriptor, which is a vector. Otherwise, the existence of Emtn and Eapp is of little significance. 
        The output feature is a matrix, which means it is basically not compressed. This encoder can be completely replaced by a VAE.
        '''        
        filters = [64, 256, 512, 1024, 2048]
        outputs=COMPRESS_DIM
        self.fc = torch.nn.Linear(filters[4], outputs)       
       
    def forward(self, x):
        # First part
        logging.debug(f"image x: {x.shape}") # [1, 3, 256, 256]
        out = self.conv(x)
        logging.debug(f"After conv: {out.shape}")  # [1, 3, 256, 256]
        out = self.resblock_128(out)
        logging.debug(f"After resblock_128: {out.shape}") # [1, 128, 256, 256]
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_256(out)
        logging.debug(f"After resblock_256: {out.shape}")
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_512(out)
        logging.debug(f"After resblock_512: {out.shape}") # [1, 512, 64, 64]
        out = self.avgpool(out) # at 512x512 image training - we need this  🤷 i rip this out so we can keep things 64x64 - it doesnt align to diagram though
        # logging.debug(f"After avgpool: {out.shape}") # [1, 256, 64, 64]
   
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)
        logging.debug(f"After conv_1: {out.shape}") # [1, 1536, 32, 32]
        
     # reshape 1546 -> C96 x D16
        vs = out.view(out.size(0), 96, 16, *out.shape[2:]) # 🤷 this maybe inaccurate
        logging.debug(f"reshape 1546 -> C96 x D16 : {vs.shape}") 
        
        
        # 1
        vs = self.resblock3D_96(vs)
        logging.debug(f"After resblock3D_96: {vs.shape}") 
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]

        # 2
        vs = self.resblock3D_96_1(vs)
        logging.debug(f"After resblock3D_96_1: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_1_2(vs)
        logging.debug(f"After resblock3D_96_1_2: {vs.shape}")

        # 3
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_2_2(vs)
        logging.debug(f"After resblock3D_96_2_2: {vs.shape}")

        # Second part
        es_resnet = self.custom_resnet50(x)
        ### TODO 2
        # print(f"🍌 es:{es_resnet.shape}") # [1, 512, 2, 2]
        es_flatten = torch.flatten(es_resnet, start_dim=1)
        es = self.fc(es_flatten) # torch.Size([bs, 2048]) -> torch.Size([bs, COMPRESS_DIM])        
       
        return vs, es




class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super(AdaptiveGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        normalized = self.group_norm(x)
        return normalized * self.weight + self.bias


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
     

class ResBlock2D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='bilinear', align_corners=False)
        
        return out

class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

#    @profile
    def forward(self, x):
        residual = x
        logging.debug(f"   🍒 ResBlock3D x.shape:{x.shape}")
        out = self.conv1(x)
        logging.debug(f"   conv1 > out.shape:{out.shape}")
        out = self.norm1(out)
        logging.debug(f"   norm1 > out.shape:{out.shape}")
        out = F.relu(out)
        logging.debug(f"   F.relu(out) > out.shape:{out.shape}")
        out = self.conv2(out)
        logging.debug(f"   conv2 > out.shape:{out.shape}")
        out = self.norm2(out)
        logging.debug(f"   norm2 > out.shape:{out.shape}")
        
        residual = self.residual_conv(residual)
        logging.debug(f"   residual > residual.shape:{residual.shape}",)
        
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out






class FlowField(nn.Module):
    def __init__(self):
        super(FlowField, self).__init__()
        
        self.conv1x1 = nn.Conv2d(512, 2048, kernel_size=1).to(device)

        # reshape the tensor from [batch_size, 2048, height, width] to [batch_size, 512, 4, height, width], effectively splitting the channels into a channels dimension of size 512 and a depth dimension of size 4.
        self.reshape_layer = lambda x: x.view(-1, 512, 4, *x.shape[2:]).to(device)

        self.resblock1 = ResBlock3D_Adaptive(in_channels=512, out_channels=256).to(device)
        # self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.upsample1 = lambda x: F.interpolate(x, scale_factor=(2, 2, 2)).to(device)

        self.resblock2 = ResBlock3D_Adaptive( in_channels=256, out_channels=128).to(device)
        # self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.upsample2 = lambda x: F.interpolate(x, scale_factor=(2, 2, 2)).to(device)

        self.resblock3 =  ResBlock3D_Adaptive( in_channels=128, out_channels=64).to(device)
        # self.upsample3 = F.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.upsample3 = lambda x: F.interpolate(x, scale_factor=(1, 2, 2)).to(device)

        self.resblock4 = ResBlock3D_Adaptive( in_channels=64, out_channels=32).to(device)
        # self.upsample4 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.upsample4 = lambda x: F.interpolate(x, scale_factor=(1, 2, 2)).to(device)

        self.conv3x3x3 = nn.Conv3d(32, 3, kernel_size=3, padding=1).to(device)
        self.gn = nn.GroupNorm(1, 3).to(device)
        self.tanh = nn.Tanh().to(device)
    
#    @profile
    def forward(self, zs,adaptive_gamma, adaptive_beta): # 
       # zs = zs * adaptive_gamma.unsqueeze(-1).unsqueeze(-1) + adaptive_beta.unsqueeze(-1).unsqueeze(-1)
        



        logging.debug(f"FlowField > zs sum.shape:{zs.shape}") #torch.Size([1, 512, 1, 1])
        x = self.conv1x1(zs)
        logging.debug(f"      conv1x1 > x.shape:{x.shape}") #  -> [1, 2048, 1, 1]
        x = self.reshape_layer(x)
        logging.debug(f"      reshape_layer > x.shape:{x.shape}") # -> [1, 512, 4, 1, 1]
        x = self.resblock1(x)
        x = self.upsample1(x)
        logging.debug(f"      upsample1 > x.shape:{x.shape}") # [1, 512, 4, 1, 1]
        x = self.upsample2(self.resblock2(x))
        logging.debug(f"      upsample2 > x.shape:{x.shape}") #[512, 256, 8, 16, 16]
        x = self.upsample3(self.resblock3(x))
        logging.debug(f"      upsample3 > x.shape:{x.shape}")# [512, 128, 16, 32, 32]
        x = self.upsample4(self.resblock4(x))
        logging.debug(f"      upsample4 > x.shape:{x.shape}")
        x = self.conv3x3x3(x)
        logging.debug(f"      conv3x3x3 > x.shape:{x.shape}")
        x = self.gn(x)
        logging.debug(f"      gn > x.shape:{x.shape}")
        x = F.relu(x)
        logging.debug(f"      F.relu > x.shape:{x.shape}")

        x = self.tanh(x)
        logging.debug(f"      tanh > x.shape:{x.shape}")

        # Assertions for shape and values
        assert x.shape[1] == 3, f"Expected 3 channels after conv3x3x3, got {x.shape[1]}"

        return x
 # produce a 3D warping field w𝑠→
    
    

'''
The ResBlock3D class represents a 3D residual block. It consists of two 3D convolutional layers (conv1 and conv2) with group normalization (norm1 and norm2) and ReLU activation. The residual connection is implemented using a shortcut connection.
Let's break down the code:

The init method initializes the layers of the residual block.

conv1 and conv2 are 3D convolutional layers with the specified input and output channels, kernel size of 3, and padding of 1.
norm1 and norm2 are group normalization layers with 32 groups and the corresponding number of channels.
If the input and output channels are different, a shortcut connection is created using a 1x1 convolutional layer and group normalization to match the dimensions.


The forward method defines the forward pass of the residual block.

The input x is stored as the residual.
The input is passed through the first convolutional layer (conv1), followed by group normalization (norm1) and ReLU activation.
The output is then passed through the second convolutional layer (conv2) and group normalization (norm2).
If a shortcut connection exists (i.e., input and output channels are different), the residual is passed through the shortcut connection.
The residual is added to the output of the second convolutional layer.
Finally, ReLU activation is applied to the sum.



The ResBlock3D class can be used as a building block in a larger 3D convolutional neural network architecture. It allows for the efficient training of deep networks by enabling the gradients to flow directly through the shortcut connection, mitigating the vanishing gradient problem.
You can create an instance of the ResBlock3D class by specifying the input and output channels:'''
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super(ResBlock3D, self).__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out
    
    
'''
G3d Class:
- The G3d class represents the 3D convolutional network (G3D) in the diagram.
- It consists of a downsampling path and an upsampling path.

Downsampling Path:
- The downsampling block in the code corresponds to the downsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D average pooling (nn.AvgPool3d) operations.
- The architecture of the downsampling path follows the structure shown in the diagram:
  - ResBlock3D(in_channels, 96) corresponds to the ResBlock3D-96 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-96.
  - ResBlock3D(96, 192) corresponds to the ResBlock3D-192 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 384) corresponds to the ResBlock3D-384 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 768) corresponds to the ResBlock3D-768 block.

Upsampling Path:
- The upsampling block in the code corresponds to the upsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D upsampling (nn.Upsample) operations.
- The architecture of the upsampling path follows the structure shown in the diagram:
  - ResBlock3D(768, 384) corresponds to the ResBlock3D-384 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 192) corresponds to the ResBlock3D-192 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 96) corresponds to the ResBlock3D-96 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-96.

Final Convolution:
- The final_conv layer in the code corresponds to the GN, ReLU, 3x3x3-Conv3D-96 block in the diagram.
- It takes the output of the upsampling path and applies a 3D convolution with a kernel size of 3 and padding of 1 to produce the final output.

Forward Pass:
- During the forward pass, the input tensor x is passed through the downsampling path, then through the upsampling path, and finally through the final convolution layer.
- The output of the G3d network is a tensor of the same spatial dimensions as the input, but with 96 channels.

In summary, the G3d class in the code aligns well with the 3D convolutional network (G3D) shown in the diagram. The downsampling path, upsampling path, and final convolution layer in the code correspond to the respective blocks in the diagram. The ResBlock3D and pooling/upsampling operations are consistent with the diagram, and the forward pass follows the expected flow of data through the network.
'''


class G3d(nn.Module):
    def __init__(self, in_channels):
        super(G3d, self).__init__()
        self.downsampling = nn.Sequential(
            ResBlock3D(in_channels, 96),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(96, 192),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(192, 384),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(384, 768),
        ).to(device)
        self.upsampling = nn.Sequential(
            ResBlock3D(768, 384),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(384, 192),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(192, 96),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        ).to(device)
        self.final_conv = nn.Conv3d(96, 96, kernel_size=3, padding=1).to(device)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        return x


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2D, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
'''
The G2d class consists of the following components:

The input has 96 channels (C96)
The input has a depth dimension of 16 (D16)
The output should have 1536 channels (C1536)

The depth dimension (D16) is present because the input to G2d is a 3D tensor 
(volumetric features) with shape (batch_size, 96, 16, height/4, width/4).
The reshape operation is meant to collapse the depth dimension and increase the number of channels.


The ResBlock2D layers have 512 channels, not 1536 channels as I previously stated. 
The diagram clearly shows 8 ResBlock2D-512 layers before the upsampling blocks that reduce the number of channels.
To summarize, the G2D network takes the orthographically projected 2D feature map from the 3D volumetric features as input.
It first reshapes the number of channels to 512 using a 1x1 convolution layer. 
Then it passes the features through 8 residual blocks (ResBlock2D) that maintain 512 channels. 
This is followed by upsampling blocks that progressively halve the number of channels while doubling the spatial resolution, 
going from 512 to 256 to 128 to 64 channels.
Finally, a 3x3 convolution outputs the synthesized image with 3 color channels.

'''

class G2d(nn.Module):
    def __init__(self, in_channels):
        super(G2d, self).__init__()
        self.reshape = nn.Conv2d(96, 1536, kernel_size=1).to(device)  # Reshape C96xD16 → C1536
        self.conv1x1 = nn.Conv2d(1536, 512, kernel_size=1).to(device)  # 1x1 convolution to reduce channels to 512

        self.res_blocks = nn.Sequential(
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        ).to(device)

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256)
        ).to(device)

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128)
        ).to(device)

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64)
        ).to(device)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        logging.debug(f"G2d > x:{x.shape}")
        x = self.reshape(x)
        x = self.conv1x1(x)  # Added 1x1 convolution to reduce channels to 512
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)
        return x


'''
In this expanded version of compute_rt_warp, we first compute the rotation matrix from the rotation parameters using the compute_rotation_matrix function. The rotation parameters are assumed to be a tensor of shape (batch_size, 3), representing rotation angles in degrees around the x, y, and z axes.
Inside compute_rotation_matrix, we convert the rotation angles from degrees to radians and compute the individual rotation matrices for each axis using the rotation angles. We then combine the rotation matrices using matrix multiplication to obtain the final rotation matrix.
Next, we create a 4x4 affine transformation matrix and set the top-left 3x3 submatrix to the computed rotation matrix. We also set the first three elements of the last column to the translation parameters.
Finally, we create a grid of normalized coordinates using F.affine_grid based on the affine transformation matrix. 
The grid size is assumed to be 64x64x64, but you can adjust it according to your specific requirements.
The resulting grid represents the warping transformations based on the given rotation and translation parameters, which can be used to warp the volumetric features or other tensors.
https://github.com/Kevinfringe/MegaPortrait/issues/4

'''
def compute_rt_warp(rotation, translation, invert=False, grid_size=64):
    """
    Computes the rotation/translation warpings (w_rt).
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        translation (torch.Tensor): The translation vector of shape (batch_size, 3).
        invert (bool): If True, invert the transformation matrix.
        
    Returns:
        torch.Tensor: The resulting transformation grid.
    """
    # Compute the rotation matrix from the rotation parameters
    rotation_matrix = compute_rotation_matrix(rotation)

    # Create a 4x4 affine transformation matrix
    affine_matrix = torch.eye(4, device=rotation.device).repeat(rotation.shape[0], 1, 1)

    # Set the top-left 3x3 submatrix to the rotation matrix
    affine_matrix[:, :3, :3] = rotation_matrix

    # Set the first three elements of the last column to the translation parameters
    affine_matrix[:, :3, 3] = translation

    # Invert the transformation matrix if needed
    if invert:
        affine_matrix = torch.inverse(affine_matrix)

    # # Create a grid of normalized coordinates 
    grid = F.affine_grid(affine_matrix[:, :3], (rotation.shape[0], 1, grid_size, grid_size, grid_size), align_corners=False)
    # # Transpose the dimensions of the grid to match the expected shape
    grid = grid.permute(0, 4, 1, 2, 3)
    return grid

def compute_rotation_matrix(rotation):
    """
    Computes the rotation matrix from rotation angles.
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        
    Returns:
        torch.Tensor: The rotation matrix of shape (batch_size, 3, 3).
    """
    # Assumes rotation is a tensor of shape (batch_size, 3), representing rotation angles in degrees
    rotation_rad = rotation * (torch.pi / 180.0)  # Convert degrees to radians

    cos_alpha = torch.cos(rotation_rad[:, 0])
    sin_alpha = torch.sin(rotation_rad[:, 0])
    cos_beta = torch.cos(rotation_rad[:, 1])
    sin_beta = torch.sin(rotation_rad[:, 1])
    cos_gamma = torch.cos(rotation_rad[:, 2])
    sin_gamma = torch.sin(rotation_rad[:, 2])

    # Compute the rotation matrix using the rotation angles
    zero = torch.zeros_like(cos_alpha)
    one = torch.ones_like(cos_alpha)

    R_alpha = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_alpha, -sin_alpha], dim=1),
        torch.stack([zero, sin_alpha, cos_alpha], dim=1)
    ], dim=1)

    R_beta = torch.stack([
        torch.stack([cos_beta, zero, sin_beta], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_beta, zero, cos_beta], dim=1)
    ], dim=1)

    R_gamma = torch.stack([
        torch.stack([cos_gamma, -sin_gamma, zero], dim=1),
        torch.stack([sin_gamma, cos_gamma, zero], dim=1),
        torch.stack([zero, zero, one], dim=1)
    ], dim=1)

    # Combine the rotation matrices
    rotation_matrix = torch.matmul(R_alpha, torch.matmul(R_beta, R_gamma))

    return rotation_matrix



'''
In the updated Emtn class, we use two separate networks (head_pose_net and expression_net) to predict the head pose and expression parameters, respectively.

The head_pose_net is a ResNet-18 model pretrained on ImageNet, with the last fully connected layer replaced to output 6 values (3 for rotation and 3 for translation).
The expression_net is another ResNet-18 model with the last fully connected layer adjusted to output the desired dimensions of the expression vector (e.g., 50).

In the forward method, we pass the input x through both networks to obtain the head pose and expression predictions. We then split the head pose output into rotation and translation parameters.
The Emtn module now returns the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
Note: Make sure to adjust the dimensions of the rotation, translation, and expression parameters according to your specific requirements and the details provided in the MegaPortraits paper.'''
class Emtn(nn.Module):
    def __init__(self):
        super().__init__()
        # https://github.com/johndpope/MegaPortrait-hack/issues/19
        # replace this with off the shelf SixDRepNet
        self.head_pose_net = resnet18(pretrained=True).to(device)
        self.head_pose_net.fc = nn.Linear(self.head_pose_net.fc.in_features, 6).to(device)  # 6 corresponds to rotation and translation parameters
        self.rotation_net =  SixDRepNet_Detector()

        model = resnet18(pretrained=False,num_classes=512).to(device)  # 512 feature_maps = resnet18(input_image) ->   Should print: torch.Size([1, 512, 7, 7])
        # Remove the fully connected layer and the adaptive average pooling layer
        self.expression_net = nn.Sequential(*list(model.children())[:-1])
        self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE)  # https://github.com/neeek2303/MegaPortraits/issues/3
        # self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) #OPTIONAL 🤷 - 16x16 is better?

        ## TODO 2
        outputs=COMPRESS_DIM ## 512,,方便后面的WarpS2C操作 512 -> 2048 channel
        self.fc = torch.nn.Linear(2048, outputs)

    def forward(self, x):
        # Forward pass through head pose network
        rotations,_ = self.rotation_net.predict(x)
        logging.debug(f"📐 rotation :{rotations}")
       

        head_pose = self.head_pose_net(x)

        # Split head pose into rotation and translation parameters
        # rotation = head_pose[:, :3]  - this is shit
        translation = head_pose[:, 3:]


        # Forward pass image through expression network
        expression_resnet = self.expression_net(x)
        ### TODO 2
        expression_flatten = torch.flatten(expression_resnet, start_dim=1)
        expression = self.fc(expression_flatten)  # (bs, 2048) ->>> (bs, COMPRESS_DIM)

        return rotations, translation, expression
    #This encoder outputs head rotations R𝑠/𝑑 ,translations t𝑠/𝑑 , and latent expression descriptors z𝑠/𝑑



'''
Rotation and Translation Warping (𝑤𝑟𝑡_wrt_):

For 𝑤𝑟𝑡→𝑑_wrt_→_d_​: This warping applies a transformation matrix (rotation and translation) to an identity grid.
For 𝑤𝑟𝑡𝑠→_wrts_→​: This warping applies an inverse transformation matrix to an identity grid.


Expression Warping (𝑤𝑒𝑚_wem_​):

Separate warping generators are used for source to canonical (𝑤𝑒𝑚𝑠→_wems_→​) and canonical to driver (𝑤𝑒𝑚→𝑑_wem_→_d_​).
Both warping generators share the same architecture, which includes several 3D residual blocks with Adaptive GroupNorms.
Inputs to these generators are the sums of the expression and appearance descriptors (𝑧𝑠+𝑒𝑠_zs_​+_es_​ for source and 𝑧𝑑+𝑒𝑠_zd_​+_es_​ for driver).
Adaptive parameters are generated by multiplying these sums with learned matrices.

'''
class WarpGeneratorS2C(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorS2C, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        # Adaptive parameters are generated by multiplying these sums with learned matrices.
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rs, ts, zs, es):
        # Assert shapes of input tensors
        assert Rs.shape == (zs.shape[0], 3), f"Expected Rs shape (batch_size, 3), got {Rs.shape}"
        assert ts.shape == (zs.shape[0], 3), f"Expected ts shape (batch_size, 3), got {ts.shape}"
        assert zs.shape == es.shape, f"Expected zs and es to have the same shape, got {zs.shape} and {es.shape}"

        # Sum es with zs
        zs_sum = zs + es

        # Generate adaptive parameters
        # adaptive_gamma = torch.matmul(zs_sum, self.adaptive_matrix_gamma.T)
        # adaptive_beta = torch.matmul(zs_sum, self.adaptive_matrix_beta.T)
        '''
        ### TODO 3: add adaptive_matrix_gamma
        According to the description of the paper (Page11: To generate adaptive parameters, 
        we multiply the foregoing sums and additionally learned matrices for each pair of parameters.), 
        adaptive_matrix_gamma should be retained. It is not used to change the shape, but can generate learning parameters, 
        which is more reasonable than just using sum.
        '''
        zs_sum = torch.matmul(zs_sum, self.adaptive_matrix_gamma) 
        zs_sum = zs_sum.unsqueeze(-1).unsqueeze(-1) ### TODO 3: add unsqueeze(-1).unsqueeze(-1) to match the shape of w_em_s2c

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_s2c = self.flowfield(zs_sum,adaptive_gamma,adaptive_beta) ### TODO 3: flowfield do not need them (adaptive_gamma,adaptive_beta)
        logging.debug(f"w_em_s2c:  :{w_em_s2c.shape}") # 🤷 this is [1, 3, 16, 16, 16] but should it be 16x16 or 64x64?  
        # Compute rotation/translation warping
        w_rt_s2c = compute_rt_warp(Rs, ts, invert=True, grid_size=64)
        logging.debug(f"w_rt_s2c: :{w_rt_s2c.shape}") 
        

        # 🤷 its the wrong dimensions - idk - 
        # Resize w_em_s2c to match w_rt_s2c
        w_em_s2c_resized = F.interpolate(w_em_s2c, size=w_rt_s2c.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_s2c_resized: {w_em_s2c_resized.shape}")
        w_s2c = w_rt_s2c + w_em_s2c_resized

        return w_s2c


class WarpGeneratorC2D(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorC2D, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        # Adaptive parameters are generated by multiplying these sums with learned matrices.
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rd, td, zd, es):
        # Assert shapes of input tensors
        assert Rd.shape == (zd.shape[0], 3), f"Expected Rd shape (batch_size, 3), got {Rd.shape}"
        assert td.shape == (zd.shape[0], 3), f"Expected td shape (batch_size, 3), got {td.shape}"
        assert zd.shape == es.shape, f"Expected zd and es to have the same shape, got {zd.shape} and {es.shape}"

        # Sum es with zd
        zd_sum = zd + es
        
        # Generate adaptive parameters
        # adaptive_gamma = torch.matmul(zd_sum, self.adaptive_matrix_gamma)
        # adaptive_beta = torch.matmul(zd_sum, self.adaptive_matrix_beta)
        '''
        ### TODO 3: add adaptive_matrix_gamma
        According to the description of the paper (Page11: To generate adaptive parameters, 
        we multiply the foregoing sums and additionally learned matrices for each pair of parameters.), 
        adaptive_matrix_gamma should be retained. It is not used to change the shape, but can generate learning parameters, 
        which is more reasonable than just using sum.
        '''
        zd_sum = torch.matmul(zd_sum, self.adaptive_matrix_gamma) 
        zd_sum = zd_sum.unsqueeze(-1).unsqueeze(-1) ### TODO 3 add unsqueeze(-1).unsqueeze(-1) to match the shape of w_em_c2d

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_c2d = self.flowfield(zd_sum,adaptive_gamma,adaptive_beta)

        # Compute rotation/translation warping
        w_rt_c2d = compute_rt_warp(Rd, td, invert=False, grid_size=64)

         # Resize w_em_c2d to match w_rt_c2d
        w_em_c2d_resized = F.interpolate(w_em_c2d, size=w_rt_c2d.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_c2d_resized:{w_em_c2d_resized.shape}" )

        w_c2d = w_rt_c2d + w_em_c2d_resized

        return w_c2d


# Function to apply the 3D warping field
def apply_warping_field(v, warp_field):
    B, C, D, H, W = v.size()
    logging.debug(f"🍏 apply_warping_field v:{v.shape}", )
    logging.debug(f"warp_field:{warp_field.shape}" )

    device = v.device

    # Resize warp_field to match the dimensions of v
    warp_field = F.interpolate(warp_field, size=(D, H, W), mode='trilinear', align_corners=True)
    logging.debug(f"Resized warp_field:{warp_field.shape}" )

    # Create a meshgrid for the canonical coordinates
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # Shape: [D, H, W, 3]
    logging.debug(f"Canonical grid:{grid.shape}" )

    # Add batch dimension and repeat the grid for each item in the batch
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Batch grid:{grid.shape}" )

    # Apply the warping field to the grid
    warped_grid = grid + warp_field.permute(0, 2, 3, 4, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Warped grid:{warped_grid.shape}" )

    # Normalize the grid to the range [-1, 1]
    normalization_factors = torch.tensor([W-1, H-1, D-1], device=device)
    logging.debug(f"Normalization factors:{normalization_factors}" )
    warped_grid = 2.0 * warped_grid / normalization_factors - 1.0
    logging.debug(f"Normalized warped grid:{warped_grid.shape}" )

    # Apply grid sampling
    v_canonical = F.grid_sample(v, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
    logging.debug(f"v_canonical:{v_canonical.shape}" )

    return v_canonical




'''
The main changes made to align the code with the training stages are:

Introduced Gbase class that combines the components of the base model.
Introduced Genh class for the high-resolution model.
Introduced GHR class that combines the base model Gbase and the high-resolution model Genh.
Introduced Student class for the student model, which includes an encoder, decoder, and SPADE blocks for avatar conditioning.
Added separate training functions for each stage: train_base, train_hr, and train_student.
Demonstrated the usage of the training functions and saving the trained models.

Note: The code assumes the presence of appropriate dataloaders (dataloader, dataloader_hr, dataloader_avatars) and the implementation of the SPADEResBlock class for the student model. Additionally, the specific training loop details and loss functions need to be implemented based on the paper's description.

The main changes made to align the code with the paper are:

The Emtn (motion encoder) is now a single module that outputs the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
The warping generators (Ws2c and Wc2d) now take the rotation, translation, expression, and appearance features as separate inputs, as described in the paper.
The warping process is updated to match the paper's description. First, the volumetric features (vs) are warped using ws2c to obtain the canonical volume (vc). Then, vc is processed by G3d to obtain vc2d. Finally, vc2d is warped using wc2d to impose the driving motion.
The orthographic projection (denoted as P in the paper) is implemented as an average pooling operation followed by a squeeze operation to reduce the spatial dimensions.
The projected features (vc2d_projected) are passed through G2d to obtain the final output image (xhat).

These changes align the code more closely with the architecture and processing steps described in the MegaPortraits paper.


The volumetric features (vs) obtained from the appearance encoder (Eapp) are warped using the warping field ws2c generated by Ws2c. This warping transforms the volumetric features into the canonical coordinate space, resulting in the canonical volume (vc).
The canonical volume (vc) is then processed by the 3D convolutional network G3d to obtain vc2d.
The vc2d features are warped using the warping field wc2d generated by Wc2d. This warping imposes the driving motion onto the features.
The warped vc2d features are then orthographically projected using average pooling along the depth dimension (denoted as P in the paper). This step reduces the spatial dimensions of the features.
Finally, the projected features (vc2d_projected) are passed through the 2D convolutional network G2d to obtain the final output image (xhat).



'''

import matplotlib.pyplot as plt


class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.appearanceEncoder = Eapp()
        self.motionEncoder = Emtn()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512) # source-to-canonical
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512) # canonical-to-driving 
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

#    @profile
    def forward(self, xs, xd):
        vs, es = self.appearanceEncoder(xs)
   
        # The motionEncoder outputs head rotations R𝑠/𝑑 ,translations t𝑠/𝑑 , and latent expression descriptors z𝑠/𝑑
        Rs, ts, zs = self.motionEncoder(xs)
        Rd, td, zd = self.motionEncoder(xd)

        logging.debug(f"es shape:{es.shape}")
        logging.debug(f"zs shape:{zs.shape}")


        w_s2c = self.warp_generator_s2c(Rs, ts, zs, es)


        logging.debug(f"vs shape:{vs.shape}") 
        # Warp vs using w_s2c to obtain canonical volume vc
        vc = apply_warping_field(vs, w_s2c)
        assert vc.shape[1:] == (96, 16, 64, 64), f"Expected vc shape (_, 96, 16, 64, 64), got {vc.shape}"

        # Process canonical volume (vc) using G3d to obtain vc2d
        vc2d = self.G3d(vc)

        # Generate warping field w_c2d
        w_c2d = self.warp_generator_c2d(Rd, td, zd, es)
        logging.debug(f"w_c2d shape:{w_c2d.shape}") 

        # Warp vc2d using w_c2d to impose driving motion
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        assert vc2d_warped.shape[1:] == (96, 16, 64, 64), f"Expected vc2d_warped shape (_, 96, 16, 64, 64), got {vc2d_warped.shape}"

        # Perform orthographic projection (P)
        vc2d_projected = torch.sum(vc2d_warped, dim=2)

        # Pass projected features through G2d to obtain the final output image (xhat)
        xhat = self.G2d(vc2d_projected)

        #self.visualize_warp_fields(xs, xd, w_s2c, w_c2d, Rs, ts, Rd, td)
        return xhat

    def visualize_warp_fields(self, xs, xd, w_s2c, w_c2d, Rs, ts, Rd, td):
        """
        Visualize images, warp fields, and rotations for source and driving data.

        Parameters:
        - xs (torch.Tensor): Source image tensor.
        - xd (torch.Tensor): Driving image tensor.
        - w_s2c (torch.Tensor): Warp field from source to canonical.
        - w_c2d (torch.Tensor): Warp field from canonical to driving.
        - Rs (torch.Tensor): Rotation matrix for source.
        - ts (torch.Tensor): Translation vectors for source.
        - Rd (torch.Tensor): Rotation matrix for driving.
        - td (torch.Tensor): Translation vectors for driving.
        """

        # Extract pitch, yaw, and roll from rotation vectors
        pitch_s, yaw_s, roll_s = Rs[:, 0], Rs[:, 1], Rs[:, 2]
        pitch_d, yaw_d, roll_d = Rd[:, 0], Rd[:, 1], Rd[:, 2]

        logging.debug(f"Source Image Pitch: {pitch_s}, Yaw: {yaw_s}, Roll: {roll_s}")
        logging.debug(f"Driving Image Pitch: {pitch_d}, Yaw: {yaw_d}, Roll: {roll_d}")

        fig = plt.figure(figsize=(15, 10))

        # Convert tensors to numpy images
        source_image = xs[0].permute(1, 2, 0).cpu().detach().numpy()
        driving_image = xd[0].permute(1, 2, 0).cpu().detach().numpy()



        # Draw rotation axes on images
        # source_image = self.draw_axis(source_image, Rs[0,1], Rs[0,0], Rs[0,2])
        # driving_image = self.draw_axis(driving_image, Rd[0,1], Rd[0,0], Rd[0,2])

        # Plot images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(source_image)
        axs[0].set_title('Source Image with Axes')
        axs[0].axis('off')

        axs[1].imshow(driving_image)
        axs[1].set_title('Driving Image with Axes')
        axs[1].axis('off')


        # Plot w_s2c warp field
        ax_w_s2c = fig.add_subplot(2, 3, 4, projection='3d')
        self.plot_warp_field(ax_w_s2c, w_s2c, 'w_s2c Warp Field')

        # Plot w_c2d warp field
        ax_w_c2d = fig.add_subplot(2, 3, 3, projection='3d')
        self.plot_warp_field(ax_w_c2d, w_c2d, 'w_c2d Warp Field')


        # pitch = Rs[0,1].cpu().detach().numpy() * np.pi / 180
        # yaw = -(Rs[0,0].cpu().detach().numpy() * np.pi / 180)
        # roll = Rs[0,2].cpu().detach().numpy() * np.pi / 180

        # # # Plot canonical head rotations
        # ax_rotations_s = fig.add_subplot(2, 3, 5, projection='3d')
        # self.plot_rotations(ax_rotations_s, pitch,yaw,roll, 'Canonical Head Rotations')


        # pitch = Rd[0,1].cpu().detach().numpy() * np.pi / 180
        # yaw = -(Rd[0,0].cpu().detach().numpy() * np.pi / 180)
        # roll = Rd[0,2].cpu().detach().numpy() * np.pi / 180

        # # # Plot driving head rotations and translations
        # ax_rotations_d = fig.add_subplot(2, 3, 6, projection='3d')
        # self.plot_rotations(ax_rotations_d, pitch,yaw,roll, 'Driving Head Rotations') 

        plt.tight_layout()
        plt.show()

    def plot_rotations(ax,pitch,yaw, roll,title,bla):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Set the aspect ratio to 'auto' to prevent scaling distortion
        ax.set_aspect('auto')

        # Center of the plot (origin)
        tdx, tdy, tdz = 0, 0, 0

        # Convert angles to radians
        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180
        roll = roll * np.pi / 180

        # Calculate axis vectors
        x_axis = np.array([np.cos(yaw) * np.cos(roll),
                        np.cos(pitch) * np.sin(roll) + np.sin(pitch) * np.sin(yaw) * np.cos(roll),
                        np.sin(yaw)])
        y_axis = np.array([-np.cos(yaw) * np.sin(roll),
                        np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll),
                        -np.cos(yaw) * np.sin(pitch)])
        z_axis = np.array([np.sin(yaw),
                        -np.cos(yaw) * np.sin(pitch),
                        np.cos(pitch)])

        # Length of the axes
        axis_length = 1

        # Plot each axis
        ax.quiver(tdx, tdy, tdz, x_axis[0], x_axis[1], x_axis[2], color='r', length=axis_length, label='X-axis')
        ax.quiver(tdx, tdy, tdz, y_axis[0], y_axis[1], y_axis[2], color='g', length=axis_length, label='Y-axis')
        ax.quiver(tdx, tdy, tdz, z_axis[0], z_axis[1], z_axis[2], color='b', length=axis_length, label='Z-axis')

        # Setting labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(title)



    def plot_warp_field(self, ax, warp_field, title, sample_rate=3):
        # Convert the warp field to numpy array
        warp_field_np = warp_field.detach().cpu().numpy()[0]  # Assuming batch size of 1

        # Get the spatial dimensions of the warp field
        depth, height, width = warp_field_np.shape[1:]

        # Create meshgrids for the spatial dimensions with sample_rate
        x = np.arange(0, width, sample_rate)
        y = np.arange(0, height, sample_rate)
        z = np.arange(0, depth, sample_rate)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Extract the x, y, and z components of the warp field with sample_rate
        U = warp_field_np[0, ::sample_rate, ::sample_rate, ::sample_rate]
        V = warp_field_np[1, ::sample_rate, ::sample_rate, ::sample_rate]
        W = warp_field_np[2, ::sample_rate, ::sample_rate, ::sample_rate]

        # Create a mask for positive and negative values
        mask_pos = (U > 0) | (V > 0) | (W > 0)
        mask_neg = (U < 0) | (V < 0) | (W < 0)

        # Set colors for positive and negative values
        color_pos = 'red'
        color_neg = 'blue'

        # Plot the quiver3D for positive values
        ax.quiver3D(X[mask_pos], Y[mask_pos], Z[mask_pos], U[mask_pos], V[mask_pos], W[mask_pos],
                    color=color_pos, length=0.3, normalize=True)

        # Plot the quiver3D for negative values
        ax.quiver3D(X[mask_neg], Y[mask_neg], Z[mask_neg], U[mask_neg], V[mask_neg], W[mask_neg],
                    color=color_neg, length=0.3, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)





'''
The high-resolution model (Genh) 
The encoder consists of a series of convolutional layers, residual blocks, and average pooling operations to downsample the input image.
The res_blocks section contains multiple residual blocks operating at the same resolution.
The decoder consists of upsampling operations, residual blocks, and a final convolutional layer to generate the high-resolution output.
'''
class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
        )
        self.res_blocks = nn.Sequential(
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

    def unsupervised_loss(self, x, x_hat):
        # Cycle consistency loss
        x_cycle = self.forward(x_hat)
        cycle_loss = F.l1_loss(x_cycle, x)

        # Other unsupervised losses can be added here

        return cycle_loss

    def supervised_loss(self, x_hat, y):
        # Supervised losses
        l1_loss = F.l1_loss(x_hat, y)
        perceptual_loss = self.perceptual_loss(x_hat, y)

        return l1_loss + perceptual_loss


# We load a pre-trained VGG19 network using models.vgg19(pretrained=True) and extract its feature layers using .features. We set the VGG network to evaluation mode and move it to the same device as the input images.
# We define the normalization parameters for the VGG network. The mean and standard deviation values are based on the ImageNet dataset, which was used to train the VGG network. We create tensors for the mean and standard deviation values and move them to the same device as the input images.
# We normalize the input images x and y using the defined mean and standard deviation values. This normalization is necessary to match the expected input format of the VGG network.
# We define the layers of the VGG network to be used for computing the perceptual loss. In this example, we use layers 1, 6, 11, 20, and 29, which correspond to different levels of feature extraction in the VGG network.
# We initialize a variable perceptual_loss to accumulate the perceptual loss values.
# We iterate over the layers of the VGG network using enumerate(vgg). For each layer, we pass the input images x and y through the layer and update their values.
# If the current layer index is in the perceptual_layers list, we compute the perceptual loss for that layer using the L1 loss between the features of x and y. We accumulate the perceptual loss values by adding them to the perceptual_loss variable.
# Finally, we return the computed perceptual loss.

    def perceptual_loss(self, x, y):
        # Load pre-trained VGG network
        vgg = models.vgg19(pretrained=True).features.eval().to(x.device)
        
        # Define VGG normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize input images
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Define perceptual loss layers
        perceptual_layers = [1, 6, 11, 20, 29]
        
        # Initialize perceptual loss
        perceptual_loss = 0.0
        
        # Extract features from VGG network
        for i, layer in enumerate(vgg):
            x = layer(x)
            y = layer(y)
            
            if i in perceptual_layers:
                # Compute perceptual loss for current layer
                perceptual_loss += nn.functional.l1_loss(x, y)
        
        return perceptual_loss

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base = self.Gbase(xs, xd)
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr



'''
In this expanded code, we have the SPADEResBlock class which represents a residual block with SPADE (Spatially-Adaptive Normalization) layers. The block consists of two convolutional layers (conv_0 and conv_1) with normalization layers (norm_0 and norm_1) and a learnable shortcut connection (conv_s and norm_s) if the input and output channels differ.
The SPADE class implements the SPADE layer, which learns to modulate the normalized activations based on the avatar embedding. It consists of a shared convolutional layer (conv_shared) followed by separate convolutional layers for gamma and beta (conv_gamma and conv_beta). The avatar embeddings (avatar_shared_emb, avatar_gamma_emb, and avatar_beta_emb) are learned for each avatar index and are added to the corresponding activations.
During the forward pass of SPADEResBlock, the input x is passed through the shortcut connection and the main branch. The main branch applies the SPADE normalization followed by the convolutional layers. The output of the block is the sum of the shortcut and the main branch activations.
The SPADE layer first normalizes the input x using instance normalization. It then applies the shared convolutional layer to obtain the shared embedding. The gamma and beta values are computed by adding the avatar embeddings to the shared embedding and passing them through the respective convolutional layers. Finally, the normalized activations are modulated using the computed gamma and beta values.
Note that this implementation assumes the presence of the avatar index tensor avatar_index during the forward pass, which is used to retrieve the corresponding avatar embeddings.
'''
class SPADEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_avatars):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_channels, num_avatars)
        self.norm_1 = SPADE(middle_channels, num_avatars)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, num_avatars)

    def forward(self, x, avatar_index):
        x_s = self.shortcut(x, avatar_index)

        dx = self.conv_0(self.actvn(self.norm_0(x, avatar_index)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, avatar_index)))

        out = x_s + dx

        return out

    def shortcut(self, x, avatar_index):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, avatar_index))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, num_avatars):
        super().__init__()
        self.num_avatars = num_avatars
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

        self.avatar_shared_emb = nn.Embedding(num_avatars, 128)
        self.avatar_gamma_emb = nn.Embedding(num_avatars, norm_nc)
        self.avatar_beta_emb = nn.Embedding(num_avatars, norm_nc)

    def forward(self, x, avatar_index):
        avatar_shared = self.avatar_shared_emb(avatar_index)
        avatar_gamma = self.avatar_gamma_emb(avatar_index)
        avatar_beta = self.avatar_beta_emb(avatar_index)

        x = self.norm(x)
        shared_emb = self.conv_shared(x)
        gamma = self.conv_gamma(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        beta = self.conv_beta(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        gamma = gamma + avatar_gamma.view(-1, self.norm_nc, 1, 1)
        beta = beta + avatar_beta.view(-1, self.norm_nc, 1, 1)

        out = x * (1 + gamma) + beta
        return out



'''
The encoder consists of convolutional layers, custom residual blocks, and average pooling operations to downsample the input image.
The SPADE (Spatially-Adaptive Normalization) blocks are applied after the encoder, conditioned on the avatar index.
The decoder consists of custom residual blocks, upsampling operations, and a final convolutional layer to generate the output image.

GDT
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.avg_pool(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)
        return input
    

class Student(nn.Module):
    def __init__(self, num_avatars):
        super(Student, self).__init__()
        self.encoder = nn.Sequential(
            ResNet18(),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 96),
            ResBlock(96, 48),
            ResBlock(48, 24),
        )
        self.decoder = nn.Sequential(
            SPADEResBlock(24, 48, num_avatars),
            SPADEResBlock(48, 96, num_avatars),
            SPADEResBlock(96, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
        )
        self.final_layer = nn.Sequential(
            nn.InstanceNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 3, kernel_size=1),
        )

    def forward(self, xd, avatar_index):
        features = self.encoder(xd)
        features = self.decoder(features, avatar_index)
        output = self.final_layer(features)
        return output


'''

Gaze Loss:

The GazeLoss class computes the gaze loss between the output and target frames.
It uses a pre-trained GazeModel to predict the gaze directions for both frames.
The MSE loss is calculated between the predicted gaze directions.


Gaze Model:

The GazeModel class is based on the VGG16 architecture.
It uses the pre-trained VGG16 model as the base and modifies the classifier to predict gaze directions.
The classifier is modified to remove the last layer and add a fully connected layer that outputs a 2-dimensional gaze vector.


Encoder Model:

The Encoder class is a convolutional neural network that encodes the input frames into a lower-dimensional representation.
It consists of a series of convolutional layers with increasing number of filters, followed by batch normalization and ReLU activation.
The encoder performs downsampling at each stage to reduce the spatial dimensions.
The final output is obtained by applying adaptive average pooling and a 1x1 convolutional layer to produce the desired output dimensionality.



The GazeLoss can be used in the PerceptualLoss class to compute the gaze loss component. The Encoder model can be used in the contrastive_loss function to encode the frames into lower-dimensional representations for computing the cosine similarity.
'''
# Gaze Lossimport torch
# from rt_gene.estimate_gaze_pytorch import GazeEstimator
# import os
# class GazeLoss(nn.Module):
#     def __init__(self, model_path='./models/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'):
#         super(GazeLoss, self).__init__()
#         self.gaze_model = GazeEstimator("cuda:0", [os.path.expanduser(model_path)])
#         # self.gaze_model.eval()
#         self.loss_fn = nn.MSELoss()

#     def forward(self, output_frame, target_frame):
#         output_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(output_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(output_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         target_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(target_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(target_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         loss = self.loss_fn(output_gaze, target_gaze)
#         return loss

# Encoder Model
class PatchGanEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(PatchGanEncoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
'''
In this updated code:

We define a GazeBlinkLoss module that combines the gaze and blink prediction tasks.
The module consists of a backbone network (VGG-16), a keypoint network, a gaze prediction head, and a blink prediction head.
The backbone network is used to extract features from the left and right eye images separately. The features are then summed to obtain a combined eye representation.
The keypoint network takes the 2D keypoints as input and produces a latent vector of size 64.
The gaze prediction head takes the concatenated eye features and keypoint features as input and predicts the gaze direction.
The blink prediction head takes only the eye features as input and predicts the blink probability.
The gaze loss is computed using both MAE and MSE losses, weighted by w_mae and w_mse, respectively.
The blink loss is computed using binary cross-entropy loss.
The total loss is the sum of the gaze loss and blink loss.

To train this model, you can follow the training procedure you described:

Use Adam optimizer with the specified hyperparameters.
Train for 60 epochs with a batch size of 64.
Use the one-cycle learning rate schedule.
Treat the predictions from RT-GENE and RT-BENE as ground truth.

Note that you'll need to preprocess your data to provide the left eye image, right eye image, 2D keypoints, target gaze, and target blink for each sample during training.
This code provides a starting point for aligning the MediaPipe-based gaze and blink loss with the approach you described. You may need to make further adjustments based on your specific dataset and requirements.

'''
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class GazeBlinkLoss(nn.Module):
    def __init__(self, device, w_mae=15, w_mse=10):
        super(GazeBlinkLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.w_mae = w_mae
        self.w_mse = w_mse
        
        self.backbone = self._create_backbone()
        self.keypoint_net = self._create_keypoint_net()
        self.gaze_head = self._create_gaze_head()
        self.blink_head = self._create_blink_head()
        
    def _create_backbone(self):
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:1])
        return model
    
    def _create_keypoint_net(self):
        return nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def _create_gaze_head(self):
        return nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def _create_blink_head(self):
        return nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, left_eye, right_eye, keypoints, target_gaze, target_blink):
        # Extract eye features using the backbone
        left_features = self.backbone(left_eye)
        right_features = self.backbone(right_eye)
        eye_features = left_features + right_features
        
        # Extract keypoint features
        keypoint_features = self.keypoint_net(keypoints)
        
        # Predict gaze
        gaze_input = torch.cat((eye_features, keypoint_features), dim=1)
        predicted_gaze = self.gaze_head(gaze_input)
        
        # Predict blink
        predicted_blink = self.blink_head(eye_features)
        
        # Compute gaze loss
        gaze_mae_loss = nn.L1Loss()(predicted_gaze, target_gaze)
        gaze_mse_loss = nn.MSELoss()(predicted_gaze, target_gaze)
        gaze_loss = self.w_mae * gaze_mae_loss + self.w_mse * gaze_mse_loss
        
        # Compute blink loss
        blink_loss = nn.BCEWithLogitsLoss()(predicted_blink, target_blink)
        
        # Total loss
        total_loss = gaze_loss + blink_loss
        
        return total_loss, predicted_gaze, predicted_blink
    


# vanilla gazeloss using mediapipe
class MPGazeLoss(nn.Module):
    def __init__(self, device):
        super(MPGazeLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_gaze, target_gaze, face_image):
        # Ensure face_image has shape (C, H, W)
        if face_image.dim() == 4 and face_image.shape[0] == 1:
            face_image = face_image.squeeze(0)
        if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
            raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
        
        # Convert face image from tensor to numpy array
        face_image = face_image.detach().cpu().numpy()
        if face_image.shape[0] == 3:  # if channels are first
            face_image = face_image.transpose(1, 2, 0)
        face_image = (face_image * 255).astype(np.uint8)

        # Extract eye landmarks using MediaPipe
        results = self.face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return torch.tensor(0.0).to(self.device)

        eye_landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
            eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

        # Compute loss for each eye
        loss = 0.0
        h, w = face_image.shape[:2]
        for left_eye, right_eye in eye_landmarks:
            # Convert landmarks to pixel coordinates
            left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
            right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

            # Create eye mask
            left_mask = torch.zeros((1, h, w)).to(self.device)
            right_mask = torch.zeros((1, h, w)).to(self.device)
            cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
            cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

            # Compute gaze loss for each eye
            left_gaze_loss = self.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
            right_gaze_loss = self.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
            loss += left_gaze_loss + right_gaze_loss

        return loss / len(eye_landmarks)
        
# class Discriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3):
#         super(Discriminator, self).__init__()
        
#         layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
#                   nn.LeakyReLU(0.2, True)]
        
#         for i in range(1, n_layers):
#             layers += [nn.Conv2d(ndf * 2**(i-1), ndf * 2**i, kernel_size=4, stride=2, padding=1),
#                        nn.InstanceNorm2d(ndf * 2**i),
#                        nn.LeakyReLU(0.2, True)]
        
#         layers += [nn.Conv2d(ndf * 2**(n_layers-1), 1, kernel_size=4, stride=1, padding=1)]
        
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class PerceptualLoss(nn.Module):
    def __init__(self, device, weights={'vgg19': 20.0, 'vggface': 5.0, 'gaze': 4.0, 'lpips': 10.0}):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.weights = weights

        # VGG19 network
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg19 = nn.Sequential(*[vgg19[i] for i in range(30)]).to(device).eval()
        self.vgg19_layers = [1, 6, 11, 20, 29]

        # VGGFace network
        self.vggface = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        self.vggface_layers = [4, 5, 6, 7]

        # LPIPS
        self.lpips = LPIPS(net='vgg').to(device).eval()

        # Gaze loss
        self.gaze_loss = MPGazeLoss(device)

    def forward(self, predicted, target, use_fm_loss=False):
        # Normalize input images
        predicted = self.normalize_input(predicted)
        target = self.normalize_input(target)

        # Compute VGG19 perceptual loss
        vgg19_loss = self.compute_vgg19_loss(predicted, target)

        # Compute VGGFace perceptual loss
        vggface_loss = self.compute_vggface_loss(predicted, target)

        # Compute LPIPS loss
        lpips_loss = self.lpips(predicted, target).mean()

        # Compute gaze loss
        # gaze_loss = self.gaze_loss(predicted, target) - broken

        # Compute total perceptual loss
        total_loss = (
            self.weights['vgg19'] * vgg19_loss +
            self.weights['vggface'] * vggface_loss +
            self.weights['lpips'] * lpips_loss +
            self.weights['gaze'] * 1 #gaze_loss
        )

        if use_fm_loss:
            # Compute feature matching loss
            fm_loss = self.compute_feature_matching_loss(predicted, target)
            total_loss += fm_loss

        return total_loss

    def compute_vgg19_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target)

    def compute_vggface_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vggface, self.vggface_layers, predicted, target)

    def compute_feature_matching_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target, detach=True)

    def compute_perceptual_loss(self, model, layers, predicted, target, detach=False):
        loss = 0.0
        predicted_features = predicted
        target_features = target

        for i, layer in enumerate(model.children()):
            if isinstance(layer, nn.Conv2d):
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            elif isinstance(layer, nn.Linear):
                predicted_features = predicted_features.view(predicted_features.size(0), -1)
                target_features = target_features.view(target_features.size(0), -1)
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            else:
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)

            if i in layers:
                if detach:
                    loss += torch.mean(torch.abs(predicted_features - target_features.detach()))
                else:
                    loss += torch.mean(torch.abs(predicted_features - target_features))

        return loss

    def normalize_input(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std


'''
As for driving image, before sending it to Emtn we do a center crop around the face of
a person. Next, we augment it using a random warping based on
thin-plate-splines, which severely degrades the shape of the facial
features, yet keeps the expression intact (ex., it cannot close or open
eyes or change the eyes’ direction). Finally, we apply a severe color
jitter.
'''

from rembg import remove
import io
import os

def remove_background_and_convert_to_rgb(image_tensor):
    """
    Remove the background from an image tensor and convert the image to RGB format.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor.
        
    Returns:
        PIL.Image.Image: Image with background removed and converted to RGB.
    """
    # Convert the tensor to a PIL Image
    image = to_pil_image(image_tensor)
    
    # Remove the background from the image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    bg_removed_bytes = remove(img_byte_arr)
    bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
    
    # Convert the image to RGB format
    bg_removed_image_rgb = bg_removed_image.convert("RGB")
    
    return bg_removed_image_rgb
def crop_and_warp_face(image_tensor, video_name, frame_idx, output_dir="output_images", pad_to_original=False, apply_warping=True,warp_strength=0.05):
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

        # Convert the face image to a numpy array
        face_array = np.array(face_image)

        # Generate random control points for thin-plate-spline warping
        rows, cols = face_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * 0.1)

        # Create a PiecewiseAffineTransform object
        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)

        # Apply the thin-plate-spline warping to the face image
        warped_face_array = warp(face_array, tps, output_shape=(rows, cols))

        # Convert the warped face array back to a PIL image
        warped_face_image = Image.fromarray((warped_face_array * 255).astype(np.uint8))

        if pad_to_original:
            # Create a new blank image with the same size as the original image
            padded_image = Image.new('RGBA', bg_removed_image.size)

            # Paste the warped face image onto the padded image at the original location
            padded_image.paste(warped_face_image, (left, top))

            # Convert the padded PIL image back to a tensor
            return to_tensor(padded_image)
        else:
            # Convert the warped PIL image back to a tensor
            return to_tensor(warped_face_image)
    else:
        return None

'''
We load the pre-trained DeepLabV3 model using models.segmentation.deeplabv3_resnet101(pretrained=True). This model is based on the ResNet-101 backbone and is pre-trained on the COCO dataset.
We define the necessary image transformations using transforms.Compose. The transformations include converting the image to a tensor and normalizing it using the mean and standard deviation values specific to the model.
We apply the transformations to the input image using transform(image) and add an extra dimension to represent the batch size using unsqueeze(0).
We move the input tensor to the same device as the model to ensure compatibility.
We perform the segmentation by passing the input tensor through the model using model(input_tensor). The output is a dictionary containing the segmentation map.
We obtain the predicted segmentation mask by taking the argmax of the output along the channel dimension using torch.max(output['out'], dim=1).
We convert the segmentation mask to a binary foreground mask by comparing the predicted class labels with the class index representing the person class (assuming it is 15 in this example). The resulting mask will have values of 1 for foreground pixels and 0 for background pixels.
Finally, we return the foreground mask.
'''
def get_foreground_mask(image):
    # Load the pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check if the input is a PyTorch tensor
    if isinstance(image, torch.Tensor):
        # Assume the tensor is already in the range [0, 1]
        if image.dim() == 4 and image.shape[0] == 1:
            # Remove the extra dimension if present
            image = image.squeeze(0)
        input_tensor = transform(image).unsqueeze(0)
    else:
        # Convert PIL Image or NumPy array to tensor
        input_tensor = transforms.ToTensor()(image)
        input_tensor = transform(input_tensor).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted segmentation mask
    _, mask = torch.max(output['out'], dim=1)

    # Convert the segmentation mask to a binary foreground mask
    foreground_mask = (mask == 15).float()  # Assuming class 15 represents the person class

    return foreground_mask.to(device)
import torch
print(torch.backends.cudnn.is_available())
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# build resnet for cifar10, debug use only
# from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

import os
import requests
from tqdm import tqdm
import zipfile
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torch
import torch.nn as nn

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]
weights_downloaded = False



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    global weights_downloaded
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if not weights_downloaded:
            download_weights()
            weights_downloaded = True
        
        script_dir = os.path.dirname(__file__)
        state_dict_path = os.path.join(script_dir, "cifar10_models/state_dicts", arch + ".pt")
        if os.path.isfile(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No such file or directory: '{state_dict_path}'")
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs)


def resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs)


def resnet50(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs)


def download_weights():

    script_dir = os.path.dirname(__file__)
    state_dicts_dir = os.path.join(script_dir, "cifar10_models")

    if os.path.isdir(state_dicts_dir) and len(os.listdir(state_dicts_dir)) > 0:
        print("Weights already downloaded. Skipping download.")
        return

    url = "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open("state_dicts.zip", "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
    directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
    
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print("Unzip file successful!")


        # original resblock
class ResBlock2D(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        layer_s = list()
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # convolution
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU(inplace=True)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return self.final_activation(x + out)


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        # state_dict = torch.load('/apdcephfs/share_1290939/kevinyxpang/STIT/resnet18-5c106cde.pth')
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params




import torch
import model
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def inference_base(source_image_path, driving_image_path, Gbase):
    print("fyi - using normalize.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load source and driving images
    source_image = load_image(source_image_path, transform)
    driving_image = load_image(driving_image_path, transform)

    # Move images to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_image = source_image.to(device)
    driving_image = driving_image.to(device)

    # Set Gbase to evaluation mode
    Gbase.eval()

    with torch.no_grad():
        # Generate output frame
        output_frame = Gbase(source_image, driving_image)

        # Convert output frame to numpy array
        output_frame = output_frame.squeeze(0).cpu().numpy()
        output_frame = np.transpose(output_frame, (1, 2, 0))
        output_frame = (output_frame + 1) / 2
        output_frame = (output_frame * 255).astype(np.uint8)

        # Convert BGR to RGB
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    return output_frame

def main():
      # Load pretrained base model
    Gbase = model.Gbase()
    # Load pretrained base model
    checkpoint = torch.load("Gbase_epoch1.pth")
    Gbase.load_state_dict(checkpoint, strict=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Gbase.to(device)

    # Specify paths to source and driving images
    source_image_path = "./output_images/source_frame_0.png"
    driving_image_path = "./output_images/driving_frame_0.png"

    # Perform inference
    output_frame = inference_base(source_image_path, driving_image_path, Gbase)

    # Save output frame
    cv2.imwrite("output_base.jpg", output_frame)

if __name__ == "__main__":
    main()
# Todo 
import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
import decord
from omegaconf import OmegaConf
from torchvision import models
from model import MPGazeLoss,Encoder
from rome_losses import Vgg19 # use vgg19 for perceptualloss 
import cv2
import mediapipe as mp
from memory_profiler import profile
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image



# Create a directory to save the images (if it doesn't already exist)
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)


face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.autograd.set_detect_anomaly(True)# this slows thing down - only for debug



'''
We load the pre-trained DeepLabV3 model using models.segmentation.deeplabv3_resnet101(pretrained=True). This model is based on the ResNet-101 backbone and is pre-trained on the COCO dataset.
We define the necessary image transformations using transforms.Compose. The transformations include converting the image to a tensor and normalizing it using the mean and standard deviation values specific to the model.
We apply the transformations to the input image using transform(image) and add an extra dimension to represent the batch size using unsqueeze(0).
We move the input tensor to the same device as the model to ensure compatibility.
We perform the segmentation by passing the input tensor through the model using model(input_tensor). The output is a dictionary containing the segmentation map.
We obtain the predicted segmentation mask by taking the argmax of the output along the channel dimension using torch.max(output['out'], dim=1).
We convert the segmentation mask to a binary foreground mask by comparing the predicted class labels with the class index representing the person class (assuming it is 15 in this example). The resulting mask will have values of 1 for foreground pixels and 0 for background pixels.
Finally, we return the foreground mask.
'''

def get_foreground_mask(image):
    # Load the pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the input image
    input_tensor = transform(image).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted segmentation mask
    _, mask = torch.max(output['out'], dim=1)

    # Convert the segmentation mask to a binary foreground mask
    foreground_mask = (mask == 15).float()  # Assuming class 15 represents the person class

    return foreground_mask


'''
Perceptual Loss:

The PerceptualLoss class combines losses from VGG19, VGG Face, and a specialized gaze loss.
It computes the perceptual losses by passing the output and target frames through the respective models and calculating the MSE loss between the features.
The total perceptual loss is a weighted sum of the individual losses.


Adversarial Loss:

The adversarial_loss function computes the adversarial loss for the generator.
It passes the generated output frame through the discriminator and calculates the MSE loss between the predicted values and a tensor of ones (indicating real samples).


Cycle Consistency Loss:

The cycle_consistency_loss function computes the cycle consistency loss.
It passes the output frame and the source frame through the generator to reconstruct the source frame.
The L1 loss is calculated between the reconstructed source frame and the original source frame.


Contrastive Loss:

The contrastive_loss function computes the contrastive loss using cosine similarity.
It calculates the cosine similarity between positive pairs (output-source, output-driving) and negative pairs (output-random, source-random).
The loss is computed as the negative log likelihood of the positive pairs over the sum of positive and negative pair similarities.
The neg_pair_loss function calculates the loss for negative pairs using a margin.


Discriminator Loss:

The discriminator_loss function computes the loss for the discriminator.
It calculates the MSE loss between the predicted values for real samples and a tensor of ones, and the MSE loss between the predicted values for fake samples and a tensor of zeros.
The total discriminator loss is the sum of the real and fake losses.
'''

# @profile
def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss.requires_grad_()

# @profile
def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss.requires_grad_()


def contrastive_loss(output_frame, source_frame, driving_frame, encoder, margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out, requires_grad=True)

    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]

    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for pos_pair in pos_pairs:
        loss = loss + torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) /
                                (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) +
                                 neg_pair_loss(pos_pair, neg_pairs, margin)))

    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for neg_pair in neg_pairs:
        loss = loss + torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss
# @profile
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss).requires_grad_()


# @profile
def gaze_loss_fn(predicted_gaze, target_gaze, face_image):
    # Ensure face_image has shape (C, H, W)
    if face_image.dim() == 4 and face_image.shape[0] == 1:
        face_image = face_image.squeeze(0)
    if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
        raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
    
    # Convert face image from tensor to numpy array
    face_image = face_image.detach().cpu().numpy()
    if face_image.shape[0] == 3:  # if channels are first
        face_image = face_image.transpose(1, 2, 0)
    face_image = (face_image * 255).astype(np.uint8)

    # Extract eye landmarks using MediaPipe
    results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return torch.tensor(0.0, requires_grad=True).to(device)

    eye_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
        eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

    # Compute loss for each eye
    loss = 0.0
    h, w = face_image.shape[:2]
    for left_eye, right_eye in eye_landmarks:
        # Convert landmarks to pixel coordinates
        left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
        right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

        # Create eye mask
        left_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        right_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
        cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

        # Compute gaze loss for each eye
        left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
        right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
        loss += left_gaze_loss + right_gaze_loss

    return loss / len(eye_landmarks)


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device)
    encoder = Encoder(input_nc=3, output_nc=256).to(device)

    for epoch in range(cfg.training.base_epochs):
        print("epoch:", epoch)
        for batch in dataloader:
            source_frames = batch['source_frames'] #.to(device)
            driving_frames = batch['driving_frames'] #.to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx].to(device)
                driving_frame = driving_frames[idx].to(device)

                # Train generator
                optimizer_G.zero_grad()
                output_frame = Gbase(source_frame, driving_frame)

                # Resize output_frame to 256x256 to match the driving_frame size
                output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)


                # 💀 Compute losses -  "losses are calculated using ONLY foreground regions" 
                # Obtain the foreground mask for the target image
                foreground_mask = get_foreground_mask(source_frame)

                # Multiply the predicted and target images with the foreground mask
                masked_predicted_image = output_frame * foreground_mask
                masked_target_image = source_frame * foreground_mask


                output_vgg_features = vgg19(masked_predicted_image)
                driving_vgg_features = vgg19(masked_target_image)  
                total_loss = 0

                for output_feat, driving_feat in zip(output_vgg_features, driving_vgg_features):
                    total_loss = total_loss + perceptual_loss_fn(output_feat, driving_feat.detach())

                loss_adversarial = adversarial_loss(masked_predicted_image, Dbase)

                loss_gaze = gaze_loss_fn(output_frame, driving_frame, source_frame) # 🤷 fix this
                # Combine the losses and perform backpropagation and optimization
                total_loss = total_loss + loss_adversarial + loss_gaze

                
                # Accumulate gradients
                loss_gaze.backward()
                total_loss.backward(retain_graph=True)
                loss_adversarial.backward()

                # Update generator
                optimizer_G.step()

                # Train discriminator
                optimizer_D.zero_grad()
                real_pred = Dbase(driving_frame)
                fake_pred = Dbase(output_frame.detach())
                loss_D = discriminator_loss(real_pred, fake_pred)

                # Backpropagate and update discriminator
                loss_D.backward()
                optimizer_D.step()


        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_gaze.item():.4f}, Loss_D: {loss_D.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def train_hr(cfg, GHR, Genh, dataloader_hr):
    GHR.train()
    Genh.train()

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device=device)

    optimizer_G = torch.optim.AdamW(Genh.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.hr_epochs, eta_min=1e-6)

    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            source_frames = batch['source_frames'].to(device)
            driving_frames = batch['driving_frames'].to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx]
                driving_frame = driving_frames[idx]

                # Generate output frame using pre-trained base model
                with torch.no_grad():
                    xhat_base = GHR.Gbase(source_frame, driving_frame)

                # Train high-resolution model
                optimizer_G.zero_grad()
                xhat_hr = Genh(xhat_base)


                # Compute losses - option 1
                # loss_supervised = Genh.supervised_loss(xhat_hr, driving_frame)
                # loss_unsupervised = Genh.unsupervised_loss(xhat_base, xhat_hr)
                # loss_perceptual = perceptual_loss_fn(xhat_hr, driving_frame)

                # option2 ? 🤷 use vgg19 as per metaportrait?
                # - Compute losses
                xhat_hr_vgg_features = vgg19(xhat_hr)
                driving_vgg_features = vgg19(driving_frame)
                loss_perceptual = 0
                for xhat_hr_feat, driving_feat in zip(xhat_hr_vgg_features, driving_vgg_features):
                    loss_perceptual += perceptual_loss_fn(xhat_hr_feat, driving_feat.detach())

                loss_supervised = perceptual_loss_fn(xhat_hr, driving_frame)
                loss_unsupervised = perceptual_loss_fn(xhat_hr, xhat_base)
                loss_gaze = gaze_loss_fn(xhat_hr, driving_frame)
                loss_G = (
                    cfg.training.lambda_supervised * loss_supervised
                    + cfg.training.lambda_unsupervised * loss_unsupervised
                    + cfg.training.lambda_perceptual * loss_perceptual
                    + cfg.training.lambda_gaze * loss_gaze
                )

                # Backpropagate and update high-resolution model
                loss_G.backward()
                optimizer_G.step()

        # Update learning rate
        scheduler_G.step()

        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.hr_epochs}], "
                  f"Loss_G: {loss_G.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Genh.state_dict(), f"Genh_epoch{epoch+1}.pth")


def train_student(cfg, Student, GHR, dataloader_avatars):
    Student.train()
    
    optimizer_S = torch.optim.AdamW(Student.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_S = CosineAnnealingLR(optimizer_S, T_max=cfg.training.student_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            avatar_indices = batch['avatar_indices'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            
            # Generate high-resolution output frames using pre-trained HR model
            with torch.no_grad():
                xhat_hr = GHR(driving_frames)
            
            # Train student model
            optimizer_S.zero_grad()
            
            # Generate output frames using student model
            xhat_student = Student(driving_frames, avatar_indices)
            
            # Compute loss
            loss_S = F.mse_loss(xhat_student, xhat_hr)
            
            # Backpropagate and update student model
            loss_S.backward()
            optimizer_S.step()
        
        # Update learning rate
        scheduler_S.step()
        
        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.student_epochs}], "
                  f"Loss_S: {loss_S.item():.4f}")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Student.state_dict(), f"Student_epoch{epoch+1}.pth")

def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter() # as augmentation for both source and target images, we use color jitter and random flip
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    Gbase = model.Gbase()
    Dbase = model.Discriminator(input_nc=3).to(device) # 🤷
    
    train_base(cfg, Gbase, Dbase, dataloader)
    
    GHR = model.GHR()
    GHR.Gbase.load_state_dict(Gbase.state_dict())
    Dhr = model.Discriminator(input_nc=3).to(device) # 🤷
    train_hr(cfg, GHR, Dhr, dataloader)
    
    Student = model.Student(num_avatars=100) # this should equal the number of celebs in dataset
    train_student(cfg, Student, GHR, dataloader)
    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(GHR.state_dict(), 'GHR.pth')
    torch.save(Student.state_dict(), 'Student.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Create tensors for x, y, and z coordinates
# x = torch.linspace(0, 10, 50)
# y = torch.linspace(0, 10, 50)
# X, Y = torch.meshgrid(x, y)
# Z1 = torch.sin(X) + torch.randn(X.shape) * 0.2
# Z2 = torch.sin(X + 1.5) + torch.randn(X.shape) * 0.2
# Z3 = Z1 + Z2

# # Create a figure and 3D axis
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the dots with quiver for direction/flow
# q1 = ax.quiver(X, Y, Z1, Z1, Z1, Z1, length=0.1, normalize=True, cmap='viridis', label='x_e,k')
# q2 = ax.quiver(X, Y, Z2, Z2, Z2, Z2, length=0.1, normalize=True, cmap='plasma', label='R_d+c,k')
# q3 = ax.quiver(X, Y, Z3, Z3, Z3, Z3, length=0.1, normalize=True, cmap='inferno', label='R_d+c,k + t_d')

# # Set labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('PyTorch Tensor Plot (3D)')

# # Add a legend
# ax.legend()

# # Display the plot
# plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import torch
import numpy as np
import torch.nn.functional as F


k = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                 dtype=torch.float32)
base = F.affine_grid(k.unsqueeze(0), [1, 1, 2, 3, 4], align_corners=True)

k = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
                 dtype=torch.float32)  # rotate
grid = F.affine_grid(k.unsqueeze(0), [1, 1, 2, 3, 4], align_corners=True)
grid = grid - base
grid = grid[0]

D, H, W, _ = grid.shape

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

k, j, i = np.meshgrid(
    np.arange(0, D, 1),
    np.arange(0, H, 1),
    np.arange(0, W, 1),
    indexing="ij",
)

u = grid[..., 0].numpy()
v = grid[..., 1].numpy()
w = grid[..., 2].numpy()

ax.quiver(k, j, i, w, v, u, length=0.3)
plt.show()
import math
import torch


'''
This function converts the head pose predictions to degrees.
It takes the predicted head pose tensor (pred) as input.
It creates an index tensor (idx_tensor) with the same length as the head pose tensor.
It performs a weighted sum of the head pose predictions multiplied by the index tensor.
The result is then scaled and shifted to obtain the head pose in degrees.
'''
def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx, _ in enumerate(pred)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = pred.squeeze()
    pred = torch.sum(pred * idx_tensor) * 3 - 99
    return pred


'''
This function computes the rotation matrix based on the yaw, pitch, and roll angles.
It takes the yaw, pitch, and roll angles (in degrees) as input.
It converts the angles from degrees to radians using torch.deg2rad.
It creates separate rotation matrices for roll, pitch, and yaw using the corresponding angles.
It combines the rotation matrices using Einstein summation (torch.einsum) to obtain the final rotation matrix.
'''
def get_rotation_matrix(yaw, pitch, roll):
    yaw = torch.deg2rad(yaw)
    pitch = torch.deg2rad(pitch)
    roll = torch.deg2rad(roll)

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.zeros(roll.shape[0], 3, 3).to(roll.device)
    roll_mat[:, 0, 0] = torch.cos(roll)
    roll_mat[:, 0, 1] = -torch.sin(roll)
    roll_mat[:, 1, 0] = torch.sin(roll)
    roll_mat[:, 1, 1] = torch.cos(roll)
    roll_mat[:, 2, 2] = 1

    pitch_mat = torch.zeros(pitch.shape[0], 3, 3).to(pitch.device)
    pitch_mat[:, 0, 0] = torch.cos(pitch)
    pitch_mat[:, 0, 2] = torch.sin(pitch)
    pitch_mat[:, 1, 1] = 1
    pitch_mat[:, 2, 0] = -torch.sin(pitch)
    pitch_mat[:, 2, 2] = torch.cos(pitch)

    yaw_mat = torch.zeros(yaw.shape[0], 3, 3).to(yaw.device)
    yaw_mat[:, 0, 0] = torch.cos(yaw)
    yaw_mat[:, 0, 2] = -torch.sin(yaw)
    yaw_mat[:, 1, 1] = 1
    yaw_mat[:, 2, 0] = torch.sin(yaw)
    yaw_mat[:, 2, 2] = torch.cos(yaw)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', yaw_mat, pitch_mat, roll_mat)
    return rot_mat



'''
This function creates a coordinate grid based on the given spatial size.
It takes the spatial size (spatial_size) and data type (type) as input.
It creates 1D tensors (x, y, z) representing the coordinates along each dimension.
It normalizes the coordinate values to the range [-1, 1].
It meshes the coordinate tensors using broadcasting to create a 3D coordinate grid.
The resulting coordinate grid has shape (height, width, depth, 3), where the last dimension represents the (x, y, z) coordinates.
'''
def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = torch.arange(w).to(type)
    y = torch.arange(h).to(type)
    z = torch.arange(d).to(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    yy = y.view(-1, 1, 1).repeat(1, w, d)
    xx = x.view(1, -1, 1).repeat(h, 1, d)
    zz = z.view(1, 1, -1).repeat(h, w, 1)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    return meshed

def compute_rt_warp2(rt, v_s, inverse=False):
    bs, _, d, h, w = v_s.shape
    yaw, pitch, roll = rt['yaw'], rt['pitch'], rt['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    # Invert the transformation matrix if needed
    if inverse:
        rot_mat = torch.inverse(rot_mat)

    rot_mat = rot_mat.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
    rot_mat = rot_mat.repeat(1, d, h, w, 1, 1)

    identity_grid = make_coordinate_grid((d, h, w), type=v_s.type())
    identity_grid = identity_grid.view(1, d, h, w, 3)
    identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

    t = t.view(t.shape[0], 1, 1, 1, 3)

    # Rotate
    warp_field = torch.bmm(identity_grid.reshape(-1, 1, 3), rot_mat.reshape(-1, 3, 3))
    warp_field = warp_field.reshape(identity_grid.shape)
    warp_field = warp_field - t

    return warp_field
import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import models
from model import PerceptualLoss, crop_and_warp_face, get_foreground_mask,remove_background_and_convert_to_rgb,apply_warping_field
import mediapipe as mp
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils
import time
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
# from flops_profiler.profiler import get_model_profile
from profiler import FlopsProfiler




output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




# align to cyclegan
def discriminator_loss(real_pred, fake_pred, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = torch.mean((real_pred - 1)**2)
        fake_loss = torch.mean(fake_pred**2)
    elif loss_type == 'vanilla':
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    
    return ((real_loss + fake_loss) * 0.5).requires_grad_()


# cosine distance formula
# s · (⟨zi, zj⟩ − m)
def cosine_loss(pos_pairs, neg_pairs, s=5.0, m=0.2):
    assert isinstance(pos_pairs, list) and isinstance(neg_pairs, list), "pos_pairs and neg_pairs should be lists"
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    assert len(neg_pairs) > 0, "neg_pairs should not be empty"
    assert s > 0, "s should be greater than 0"
    assert 0 <= m <= 1, "m should be between 0 and 1"
    
    loss = torch.tensor(0.0, requires_grad=True).to(device)

    for pos_pair in pos_pairs:
        assert isinstance(pos_pair, tuple) and len(pos_pair) == 2, "Each pos_pair should be a tuple of length 2"
        pos_sim = F.cosine_similarity(pos_pair[0], pos_pair[1], dim=0)
        pos_dist = s * (pos_sim - m)
        
        neg_term = torch.tensor(0.0, requires_grad=True).to(device)
        for neg_pair in neg_pairs:
            assert isinstance(neg_pair, tuple) and len(neg_pair) == 2, "Each neg_pair should be a tuple of length 2"
            neg_sim = F.cosine_similarity(pos_pair[0], neg_pair[1], dim=0)
            neg_term = neg_term + torch.exp(s * (neg_sim - m))
        
        assert pos_dist.shape == neg_term.shape, f"Shape mismatch: pos_dist {pos_dist.shape}, neg_term {neg_term.shape}"
        loss = loss + torch.log(torch.exp(pos_dist) / (torch.exp(pos_dist) + neg_term))
        
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    return torch.mean(-loss / len(pos_pairs)).requires_grad_()

def train_base(cfg, Gbase, Dbase, dataloader):
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})

    scaler = GradScaler()

    # Initialize generator profiler
    prof_G = FlopsProfiler(Gbase)
    prof_D = FlopsProfiler(Dbase)
    

    profile_step = 5  # Change this to the step you want to start profiling



    for epoch in range(cfg.training.base_epochs):
        print("Epoch:", epoch)
        

        for batch in dataloader:




                source_frames = batch['source_frames']
                driving_frames = batch['driving_frames']
                video_id = batch['video_id'][0]

                # Access videos from dataloader2 for cycle consistency
                source_frames2 = batch['source_frames_star']
                driving_frames2 = batch['driving_frames_star']
                video_id2 = batch['video_id_star'][0]


                num_frames = len(driving_frames)
                len_source_frames = len(source_frames)
                len_driving_frames = len(driving_frames)
                len_source_frames2 = len(source_frames2)
                len_driving_frames2 = len(driving_frames2)

                for idx in range(num_frames):

                    # Start profiling at the specified step
                    if idx == profile_step:
                        prof_G.start_profile()
                        prof_D.start_profile()

                    # loop around if idx exceeds video length
                    source_frame = source_frames[idx % len_source_frames].to(device)
                    driving_frame = driving_frames[idx % len_driving_frames].to(device)

                    source_frame_star = source_frames2[idx % len_source_frames2].to(device)
                    driving_frame_star = driving_frames2[idx % len_driving_frames2].to(device)


                    with autocast():

                        # We use multiple loss functions for training, which can be split  into two groups.
                        # The first group consists of the standard training objectives for image synthesis. 
                        # These include perceptual [14] and GAN [ 33 ] losses that match 
                        # the predicted image ˆx𝑠→𝑑 to the  ground-truth x𝑑 . 
                        pred_frame = Gbase(source_frame, driving_frame)

                        # Obtain the foreground mask for the driving image
                        # foreground_mask = get_foreground_mask(source_frame)

                        # # Move the foreground mask to the same device as output_frame
                        # foreground_mask = foreground_mask.to(pred_frame.device)

                        # # Multiply the predicted and driving images with the foreground mask
                        # # masked_predicted_image = pred_frame * foreground_mask
                        # masked_target_image = driving_frame * foreground_mask

                        save_images = True
                        # Save the images
                        if save_images:
                            # vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
                            # vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
                            vutils.save_image(pred_frame, f"{output_dir}/pred_frame_{idx}.png")
                            # vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
                            # vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
                            # vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
                            # vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

                        # Calculate perceptual losses
                        loss_G_per = perceptual_loss_fn(pred_frame, source_frame)
                      
                        # Adversarial ground truths - from Kevin Fringe
                        valid = Variable(torch.Tensor(np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)
                        fake = Variable(torch.Tensor(-1 * np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)

                        # real loss
                        real_pred = Dbase(driving_frame, source_frame)
                        loss_real = hinge_loss(real_pred, valid)

                        # fake loss
                        fake_pred = Dbase(pred_frame.detach(), source_frame)
                        loss_fake = hinge_loss(fake_pred, fake)

                        # Train discriminator
                        optimizer_D.zero_grad()
                        
                        # Calculate adversarial losses
                        real_pred = Dbase(driving_frame, source_frame)
                        fake_pred = Dbase(pred_frame.detach(), source_frame)
                        loss_D = discriminator_loss(real_pred, fake_pred, loss_type='lsgan')

                        # End discriminator profiling and print the output
                        if idx == profile_step:
                            # generator
                            prof_D.stop_profile()
                            flops = prof_D.get_total_flops(as_string=True)
                            macs = prof_D.get_total_macs(as_string=True)
                            params = prof_D.get_total_params(as_string=True)
                            prof_D.print_model_profile(profile_step=profile_step)
                            prof_D.end_profile()
                            print(f"Step {idx}: FLOPS - {flops}, MACs - {macs}, Params - {params}")


                        scaler.scale(loss_D).backward()
                        scaler.step(optimizer_D)
                        scaler.update()

                        # Calculate adversarial losses
                        loss_G_adv = 0.5 * (loss_real + loss_fake)

                         # Feature matching loss
                        loss_fm = feature_matching_loss(pred_frame, driving_frame)
                    
                        # The other objective CycleGAN regularizes the training and introduces disentanglement between the motion and canonical space
                        # In order to calculate this loss, we use an additional source-driving  pair x𝑠∗ and x𝑑∗ , 
                        # which is sampled from a different video! and therefore has different appearance from the current x𝑠 , x𝑑 pair.

                        # produce the following cross-reenacted image: ˆx𝑠∗→𝑑 = Gbase (x𝑠∗ , x𝑑 )
                        cross_reenacted_image = Gbase(source_frame_star, driving_frame)
                        if save_images:
                            vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

                        # Store the motion descriptors z𝑠→𝑑(predicted) and z𝑠∗→𝑑 (star predicted) from the 
                        # respective forward passes of the base network.
                        _, _, z_pred = Gbase.motionEncoder(pred_frame) 
                        _, _, zd = Gbase.motionEncoder(driving_frame) 
                        
                        _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                        _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

              
                        # Calculate cycle consistency loss 
                        # We then arrange the motion descriptors into positive pairs P that
                        # should align with each other: P = (z𝑠→𝑑 , z𝑑 ), (z𝑠∗→𝑑 , z𝑑 ) , and
                        # the negative pairs: N = (z𝑠→𝑑 , z𝑑∗ ), (z𝑠∗→𝑑 , z𝑑∗ ) . These pairs are
                        # used to calculate the following cosine distance:

                        P = [(z_pred, zd)     ,(z_star__pred, zd)]
                        N = [(z_pred, zd_star),(z_star__pred, zd_star)]
                        loss_G_cos = cosine_loss(P, N)

                       
                        
                        # Backpropagate and update generator
                        optimizer_G.zero_grad()
                        total_loss = cfg.training.w_per * loss_G_per + \
                            cfg.training.w_adv * loss_G_adv + \
                            cfg.training.w_fm * loss_fm + \
                            cfg.training.w_cos * loss_G_cos
                        
                        # End profiling and print the output
                        if idx == profile_step:
                            # generator
                            prof_G.stop_profile()
                            flops = prof_G.get_total_flops(as_string=True)
                            macs = prof_G.get_total_macs(as_string=True)
                            params = prof_G.get_total_params(as_string=True)
                            prof_G.print_model_profile(profile_step=profile_step)
                            prof_G.end_profile()
                            print(f"Step {idx}: FLOPS - {flops}, MACs - {macs}, Params - {params}")


                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()

                      

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G_cos.item():.4f}, Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def unnormalize(tensor):
    """
    Unnormalize a tensor using the specified mean and std.
    
    Args:
    tensor (torch.Tensor): The normalized tensor.
    mean (list): The mean used for normalization.
    std (list): The std used for normalization.
    
    Returns:
    torch.Tensor: The unnormalized tensor.
    """
    # Check if the tensor is on a GPU and if so, move it to the CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Ensure tensor is a float and detach it from the computation graph
    tensor = tensor.float().detach()
    
    # Unnormalize
    # Define mean and std used for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return tensor


def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        remove_background=True,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform,
        apply_crop_warping=True
    )


    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator().to(device)
    
    train_base(cfg, Gbase, Dbase, dataloader)    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(Dbase.state_dict(), 'Dbase.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)
import torch
from torch import nn
from torch.nn import functional as F

from typing import List



class AdversarialLoss(nn.Module):
    def __init__(self, loss_type = 'hinge'):
        super(AdversarialLoss, self).__init__()
        # TODO: different adversarial loss types
        self.loss_type = loss_type

    def forward(self, 
                fake_scores: List[List[torch.Tensor]], 
                real_scores: List[List[torch.Tensor]] = None, 
                mode: str = 'gen') -> torch.Tensor:
        """
        scores: a list of lists of scores (the second layer corresponds to a
                separate input to each of these discriminators)
        """
        loss = 0

        if mode == 'dis':
            for real_scores_net, fake_scores_net in zip(real_scores, fake_scores):
                # *_scores_net corresponds to outputs of a separate discriminator
                loss_real = 0
                
                for real_scores_net_i in real_scores_net:
                    if self.loss_type == 'hinge':
                        loss_real += torch.relu(1.0 - real_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_real /= len(real_scores_net)

                loss_fake = 0
                
                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        loss_fake += torch.relu(1.0 + fake_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_fake /= len(fake_scores_net)

                loss_net = loss_real + loss_fake
                loss += loss_net

        elif mode == 'gen':
            for fake_scores_net in fake_scores:
                assert isinstance(fake_scores_net, list), 'Expect a list of fake scores per discriminator'

                loss_net = 0

                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        # *_scores_net_i corresponds to outputs for separate inputs
                        loss_net -= fake_scores_net_i.mean()

                    else:
                        raise # not implemented

                loss_net /= len(fake_scores_net) # normalize by the number of inputs
                loss += loss_net
        
        loss /= len(fake_scores) # normalize by the nubmer of discriminators

        return loss
import torch



class PSNR(object):
    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        mse = ((y_pred - y_true) ** 2).mean()
        return 10 * torch.log10(1 / mse)
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
# try:
#     from pytorch3d.loss.mesh_laplacian_smoothing import cot_laplacian
# except:
#     from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot as cot_laplacian


def make_grid(h, w, device, dtype):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h * w, 2)

    return grid


class Transform(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(Transform, self).__init__()
        self.sigma_affine = sigma_affine
        self.sigma_tps = sigma_tps
        self.points_tps = points_tps

    def transform_img(self, img):
        b, _, h, w = img.shape
        device = img.device
        dtype = img.dtype

        if not hasattr(self, 'identity_grid'):
            identity_grid = make_grid(h, w, device, dtype)
            self.register_buffer('identity_grid', identity_grid, persistent=False)

        if not hasattr(self, 'control_grid'):
            control_grid = make_grid(self.points_tps, self.points_tps, device, dtype)
            self.register_buffer('control_grid', control_grid, persistent=False)

        # Sample transform
        noise = torch.normal(
            mean=0,
            std=self.sigma_affine,
            size=(b, 2, 3),
            device=device,
            dtype=dtype)

        self.theta = (noise + torch.eye(2, 3, device=device, dtype=dtype)[None])[:, None]  # b x 1 x 2 x 3

        self.control_params = torch.normal(
            mean=0,
            std=self.sigma_tps,
            size=(b, 1, self.points_tps ** 2),
            device=device,
            dtype=dtype)

        grid = self.warp_pts(self.identity_grid).view(-1, h, w, 2)

        return F.grid_sample(img, grid, padding_mode="reflection")

    def warp_pts(self, pts):
        b = self.theta.shape[0]
        n = pts.shape[1]
 
        pts_transformed = torch.matmul(self.theta[:, :, :, :2], pts[..., None]) + self.theta[:, :, :, 2:]
        pts_transformed = pts_transformed[..., 0]

        pdists = pts[:, :, None] - self.control_grid[:, None]
        pdists = (pdists).abs().sum(dim=3)

        result = pdists**2 * torch.log(pdists + 1e-5) * self.control_params
        result = result.sum(dim=2).view(b, n, 1)

        pts_transformed = pts_transformed + result

        return pts_transformed

    def jacobian(self, pts):
        new_pts = self.warp_pts(pts)
        grad_x = grad(new_pts[..., 0].sum(), pts, create_graph=True)
        grad_y = grad(new_pts[..., 1].sum(), pts, create_graph=True)
        jac = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        
        return jac


class EquivarianceLoss(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(EquivarianceLoss, self).__init__()
        self.transform = Transform(sigma_affine, sigma_tps, points_tps)

    def forward(self, img, kp, jac, kp_detector):
        img_transformed = self.transform.transform_img(img)
        kp_transformed, jac_transformed = kp_detector(img_transformed)
        kp_recon = self.transform.warp_pts(kp_transformed)

        loss_kp = (kp - kp_recon).abs().mean()

        jac_recon = torch.matmul(self.transform.jacobian(kp_transformed), jac_transformed)
        inv_jac = torch.linalg.inv(jac)

        loss_jac = (torch.matmul(inv_jac, jac_recon) - torch.eye(2)[None, None].type(inv_jac.type())).abs().mean()

        return loss_kp, loss_jac, img_transformed, kp_transformed, kp_recon


class LaplaceMeshLoss(nn.Module):
    def __init__(self, type='uniform', use_vector_constant=False):
        super(LaplaceMeshLoss, self).__init__()
        self.method = type
        self.precomputed_laplacian = None
        self.use_vector_constant = use_vector_constant

    def _compute_loss(self, L, verts_packed, inv_areas=None):
        if self.method == "uniform":
            loss = L.mm(verts_packed)
        
        elif self.method == "cot":
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]
            loss = L.mm(verts_packed) * norm_w - verts_packed
        
        elif self.method == "cotcurv":
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            norm_w = 0.25 * inv_areas
            loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
        
        return loss.norm(dim=1)

    def forward(self, meshes, coefs=None):
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )

        N = len(meshes)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
        verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
        weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
        weights = 1.0 / weights.float()
        norm_w, inv_areas = None, None
        with torch.no_grad():
            if self.method == "uniform":
                if self.precomputed_laplacian is None or self.precomputed_laplacian.shape[0] != verts_packed.shape[0]:
                    L = meshes.laplacian_packed()
                    self.precomputed_laplacian = L
                else:
                    L = self.precomputed_laplacian
            elif self.method in ["cot", "cotcurv"]:
                L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            else:
                raise ValueError("Method should be one of {uniform, cot, cotcurv}")

        loss = self._compute_loss(L, verts_packed,
                                  inv_areas=inv_areas)
        loss = loss * weights
        if coefs is not None:
            loss = loss * coefs.view(-1)

        return loss.sum() / N
import torch
from torch import nn
import torch.nn.functional as F

from typing import List



class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l1', ):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                real_features: List[List[List[torch.Tensor]]], 
                fake_features: List[List[List[torch.Tensor]]]
        ) -> torch.Tensor:
        """
        features: a list of features of different inputs (the third layer corresponds to
                  features of a separate input to each of these discriminators)
        """
        loss = 0

        for real_feats_net, fake_feats_net in zip(real_features, fake_features):
            # *_feats_net corresponds to outputs of a separate discriminator
            loss_net = 0

            for real_feats_layer, fake_feats_layer in zip(real_feats_net, fake_feats_net):
                assert len(real_feats_layer) == 1 or len(real_feats_layer) == len(fake_feats_layer), 'Wrong number of real inputs'
                if len(real_feats_layer) == 1:
                    real_feats_layer = [real_feats_layer[0]] * len(fake_feats_layer)

                for real_feats_layer_i, fake_feats_layer_i in zip(real_feats_layer, fake_feats_layer):
                    if self.loss_type == 'l1':
                        loss_net += F.l1_loss(fake_feats_layer_i, real_feats_layer_i)
                    elif self.loss_type == 'l2':
                        loss_net += F.mse_loss(fake_feats_layer_i, real_feats_layer_i)

            loss_net /= len(fake_feats_layer) # normalize by the number of inputs
            loss_net /= len(fake_feats_net) # normalize by the number of layers
            loss += loss_net

        loss /= len(real_features) # normalize by the number of networks

        return loss
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class KeypointsMatchingLoss(nn.Module):
    def __init__(self):
        super(KeypointsMatchingLoss, self).__init__()
        self.register_buffer('weights', torch.ones(68), persistent=False)
        self.weights[5:7] = 2.0
        self.weights[10:12] = 2.0
        self.weights[27:36] = 1.5
        self.weights[30] = 3.0
        self.weights[31] = 3.0
        self.weights[35] = 3.0
        self.weights[60:68] = 1.5
        self.weights[48:60] = 1.5
        self.weights[48] = 3
        self.weights[54] = 3

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff = pred_keypoints - keypoints

        loss = (diff.abs().mean(-1) * self.weights[None] / self.weights.sum()).sum(-1).mean()

        return loss
import torch
from torch import nn
import lpips



class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net='alex')

        for m in self.metric.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

            names = [name for name, _ in m.named_buffers()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    @torch.no_grad()
    def __call__(self, inputs, targets):
        return self.metric(inputs, targets, normalize=True).mean()

    def train(self, mode: bool = True):
        return self
# from .adversarial import AdversarialLoss
# from .feature_matching import FeatureMatchingLoss
# from .keypoints_matching import KeypointsMatchingLoss
# from .eye_closure import EyeClosureLoss
# from .lip_closure import LipClosureLoss
# from .head_pose_matching import HeadPoseMatchingLoss
# from .perceptual import PerceptualLoss

# from .segmentation import SegmentationLoss, MultiScaleSilhouetteLoss
# from .chamfer_silhouette import ChamferSilhouetteLoss
# from .equivariance import EquivarianceLoss, LaplaceMeshLoss
# from .vgg2face import VGGFace2Loss
# from .gaze import GazeLoss

# from .psnr import PSNR
# from .lpips import LPIPS
from pytorch_msssim import SSIM, MS_SSIM
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from typing import Union

# from src.utils import misc

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.
    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [0, 1].
    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input using the ImageNet mean and std
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output


class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.
    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the inputsut images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
        use_fp16 (bool) : If ``True``, use cast networks and inputs to FP16
    """

    def __init__(
            self, 
            network='vgg19', 
            layers=('relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'), 
            weights=(0.03125, 0.0625, 0.125, 0.25, 1.0),
            criterion='l1', 
            resize=False, 
            resize_mode='bilinear',
            instance_normalized=False,
            replace_maxpool_with_avgpool=False,
            num_scales=1,
            use_fp16=False
        ) -> None:
        super(PerceptualLoss, self).__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        if replace_maxpool_with_avgpool:
	        for k, v in self.model.network._modules.items():
	        	if isinstance(v, nn.MaxPool2d):
	        		self.model.network._modules[k] = nn.AvgPool2d(2)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSEloss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.fp16 = use_fp16
        if self.fp16:
            self.model.half()

    @torch.cuda.amp.autocast(True)
    def forward(self, 
                inputs: Union[torch.Tensor, list], 
                target: torch.Tensor) -> Union[torch.Tensor, list]:
        r"""Perceptual loss forward.
        Args:
           inputs (4D tensor or list of 4D tensors) : inputsut tensor.
           target (4D tensor) : Ground truth tensor, same shape as the inputsut.
        Returns:
           (scalar tensor or list of tensors) : The perceptual loss.
        """
        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            inputs = torch.cat(inputs)
        else:
            input_is_a_list = False

        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inputs, target = \
            apply_imagenet_normalization(inputs), \
            apply_imagenet_normalization(target)
        if self.resize:
            inputs = F.interpolate(
                inputs, mode=self.resize_mode, size=(224, 224),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0

        for scale in range(self.num_scales):
            if self.fp16:
                input_features = self.model(inputs.half())
                with torch.no_grad():
                    target_features = self.model(target.half())
            else:
                input_features = self.model(inputs)
                with torch.no_grad():
                    target_features = self.model(target)

            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if input_is_a_list:
                    target_feature = torch.cat([target_feature] * num_chunks)

                loss += weight * self.criterion(input_feature,
                                                target_feature)

            # Downsample the inputsut and target.
            if scale != self.num_scales - 1:
                inputs = F.interpolate(
                    inputs, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        loss /= self.num_scales

        return loss

    def train(self, mode: bool = True):
        return self


class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers

        for m in self.network.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output


def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    r"""Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          18: 'relu_4_1',
                          20: 'relu_4_2',
                          22: 'relu_4_3',
                          25: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _alexnet(layers):
    r"""Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {0: 'conv_1',
                          1: 'relu_1',
                          3: 'conv_2',
                          4: 'relu_2',
                          6: 'conv_3',
                          7: 'relu_3',
                          8: 'conv_4',
                          9: 'relu_4',
                          10: 'conv_5',
                          11: 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    r"""Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3,
                            inception.Conv2d_2a_3x3,
                            inception.Conv2d_2b_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Conv2d_3b_1x1,
                            inception.Conv2d_4a_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Mixed_5b,
                            inception.Mixed_5c,
                            inception.Mixed_5d,
                            inception.Mixed_6a,
                            inception.Mixed_6b,
                            inception.Mixed_6c,
                            inception.Mixed_6d,
                            inception.Mixed_6e,
                            inception.Mixed_7a,
                            inception.Mixed_7b,
                            inception.Mixed_7c,
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {3: 'pool_1',
                          6: 'pool_2',
                          14: 'mixed_6e',
                          18: 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    r"""Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    r"""Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url(
        'http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    r"""Get vgg face layers"""
    network = torchvision.models.vgg16(num_classes=2622).features
    state_dict = torch.utils.model_zoo.load_url(
        'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/'
        'vgg_face_dag.pth')
    layer_name_mapping = {
        0: 'conv1_1',
        2: 'conv1_2',
        5: 'conv2_1',
        7: 'conv2_2',
        10: 'conv3_1',
        12: 'conv3_2',
        14: 'conv3_3',
        17: 'conv4_1',
        19: 'conv4_2',
        21: 'conv4_3',
        24: 'conv5_1',
        26: 'conv5_2',
        28: 'conv5_3'}
    new_state_dict = {}
    for k, v in layer_name_mapping.items():
        new_state_dict[str(k) + '.weight'] =\
            state_dict[v + '.weight']
        new_state_dict[str(k) + '.bias'] = \
            state_dict[v + '.bias']

    return _PerceptualNetwork(network, layer_name_mapping, layers)
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import math


import torch
import torch.nn as nn


class Resnet50_scratch_dag(nn.Module):

    def __init__(self):
        super(Resnet50_scratch_dag, self).__init__()
        self.meta = {'mean': [131.0912, 103.8827, 91.4953],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7_s2 = nn.ReLU()
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv2_1_1x1_reduce = nn.Conv2d(64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_1x1_reduce_relu = nn.ReLU()
        self.conv2_1_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_1_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_3x3_relu = nn.ReLU()
        self.conv2_1_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_1x1_proj = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_relu = nn.ReLU()
        self.conv2_2_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_1x1_reduce_relu = nn.ReLU()
        self.conv2_2_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_2_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_3x3_relu = nn.ReLU()
        self.conv2_2_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_relu = nn.ReLU()
        self.conv2_3_1x1_reduce = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_1x1_reduce_relu = nn.ReLU()
        self.conv2_3_3x3 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_3_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_3x3_relu = nn.ReLU()
        self.conv2_3_1x1_increase = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_relu = nn.ReLU()
        self.conv3_1_1x1_reduce = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_1x1_reduce_relu = nn.ReLU()
        self.conv3_1_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_1_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_3x3_relu = nn.ReLU()
        self.conv3_1_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_1x1_proj = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_relu = nn.ReLU()
        self.conv3_2_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_1x1_reduce_relu = nn.ReLU()
        self.conv3_2_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_2_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_3x3_relu = nn.ReLU()
        self.conv3_2_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_relu = nn.ReLU()
        self.conv3_3_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_1x1_reduce_relu = nn.ReLU()
        self.conv3_3_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_3_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_3x3_relu = nn.ReLU()
        self.conv3_3_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_relu = nn.ReLU()
        self.conv3_4_1x1_reduce = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_1x1_reduce_relu = nn.ReLU()
        self.conv3_4_3x3 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_4_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_3x3_relu = nn.ReLU()
        self.conv3_4_1x1_increase = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_relu = nn.ReLU()
        self.conv4_1_1x1_reduce = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_1x1_reduce_relu = nn.ReLU()
        self.conv4_1_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_1_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_3x3_relu = nn.ReLU()
        self.conv4_1_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_1x1_proj = nn.Conv2d(512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_relu = nn.ReLU()
        self.conv4_2_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_1x1_reduce_relu = nn.ReLU()
        self.conv4_2_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_2_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_3x3_relu = nn.ReLU()
        self.conv4_2_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_relu = nn.ReLU()
        self.conv4_3_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_1x1_reduce_relu = nn.ReLU()
        self.conv4_3_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_3_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_3x3_relu = nn.ReLU()
        self.conv4_3_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_relu = nn.ReLU()
        self.conv4_4_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_1x1_reduce_relu = nn.ReLU()
        self.conv4_4_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_4_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_3x3_relu = nn.ReLU()
        self.conv4_4_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_relu = nn.ReLU()
        self.conv4_5_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_1x1_reduce_relu = nn.ReLU()
        self.conv4_5_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_5_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_3x3_relu = nn.ReLU()
        self.conv4_5_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_relu = nn.ReLU()
        self.conv4_6_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_1x1_reduce_relu = nn.ReLU()
        self.conv4_6_3x3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_6_3x3_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_3x3_relu = nn.ReLU()
        self.conv4_6_1x1_increase = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_relu = nn.ReLU()
        self.conv5_1_1x1_reduce = nn.Conv2d(1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_1x1_reduce_relu = nn.ReLU()
        self.conv5_1_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_1_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_3x3_relu = nn.ReLU()
        self.conv5_1_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_1x1_proj = nn.Conv2d(1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_relu = nn.ReLU()
        self.conv5_2_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_1x1_reduce_relu = nn.ReLU()
        self.conv5_2_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_2_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_3x3_relu = nn.ReLU()
        self.conv5_2_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_relu = nn.ReLU()
        self.conv5_3_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_1x1_reduce_relu = nn.ReLU()
        self.conv5_3_3x3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_3_3x3_bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_3x3_relu = nn.ReLU()
        self.conv5_3_1x1_increase = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_relu = nn.ReLU()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.classifier = nn.Conv2d(2048, 8631, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, data):
        conv1_7x7_s2 = self.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)
        classifier_preflatten = self.classifier(pool5_7x7_s1)
        classifier = classifier_preflatten.view(classifier_preflatten.size(0), -1)
        return classifier, pool5_7x7_s1

def resnet50_scratch_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Resnet50_scratch_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class VGGFace2Loss(object):
    def __init__(self, pretrained_model, pretrained_data='vggface2', device='cuda'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50_scratch_dag(pretrained_model).eval().cuda()
        # self.reg_model.load_state_dict(torch.load(pretrained_model), strict=False)
        # self.reg_model = self.reg_model.eval().cuda()
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()
        self.mean_rgb = torch.tensor((131.0912, 103.8827, 91.4953)).cuda()

    def reg_features(self, x):
        # out = []
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)[1]
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_rgb
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss

import torch
import torch.nn.functional as F
from torch import nn
from typing import Union

import torch
import torch.nn.functional as F
# from pytorch3d.ops.knn import knn_gather, knn_points
# from pytorch3d.structures.pointclouds import Pointclouds

from typing import Union



# class ChamferSilhouetteLoss(nn.Module):
#     def __init__(
#         self, 
#         num_neighbours=1, 
#         use_same_number_of_points=False, 
#         sample_outside_of_silhouette=False,
#         use_visibility=True
#     ):
#         super(ChamferSilhouetteLoss, self).__init__()
#         self.num_neighbours = num_neighbours
#         self.use_same_number_of_points = use_same_number_of_points
#         self.sample_outside_of_silhouette = sample_outside_of_silhouette
#         self.use_visibility = use_visibility

#     def forward(self, 
#                 pred_points: torch.Tensor,
#                 points_visibility: torch.Tensor,
#                 target_silhouette: torch.Tensor,
#                 target_segs: torch.Tensor) -> torch.Tensor:        
#         target_points, target_lengths, weight = self.get_pointcloud(target_segs, target_silhouette)

#         if self.use_visibility:
#             pred_points, pred_lengths = self.get_visible_points(pred_points, points_visibility)
                
#         if self.use_same_number_of_points:
#             target_points = target_points[:, :pred_points.shape[1]]    

#             target_lengths = pred_lengths = torch.minimum(target_lengths, pred_lengths)
            
#             if self.sample_outside_of_silhouette:
#                 target_lengths = (target_lengths.clone() * weight).long()

#             for i in range(target_points.shape[0]):
#                 target_points[i, target_lengths[i]:] = -100.0

#             for i in range(pred_points.shape[0]):
#                 pred_points[i, pred_lengths[i]:] = -100.0

#         visible_batch = target_lengths > 0
#         if self.use_visibility:
#             visible_batch *= pred_lengths > 0

#         if self.use_visibility:
#             loss = chamfer_distance(
#                 pred_points[visible_batch], 
#                 target_points[visible_batch], 
#                 x_lengths=pred_lengths[visible_batch], 
#                 y_lengths=target_lengths[visible_batch],
#                 num_neighbours=self.num_neighbours
#             )        
#         else:
#             loss = chamfer_distance(
#                 pred_points[visible_batch], 
#                 target_points[visible_batch], 
#                 y_lengths=target_lengths[visible_batch],
#                 num_neighbours=self.num_neighbours
#             )

#         if isinstance(loss, tuple):
#             loss = loss[0]
        
#         return loss, pred_points, target_points
    
#     @torch.no_grad()
#     def get_pointcloud(self, seg, silhouette):
#         if self.sample_outside_of_silhouette:
#             silhouette = (silhouette > 0.0).type(seg.type())

#             old_area = seg.view(seg.shape[0], -1).sum(1)
#             seg = seg * (1 - silhouette)
#             new_area = seg.view(seg.shape[0], -1).sum(1)

#             weight = new_area / (old_area + 1e-7)
        
#         else:
#             weight = torch.ones(seg.shape[0], dtype=seg.dtype, device=seg.device)

#         batch, coords = torch.nonzero(seg[:, 0] > 0.5).split([1, 2], dim=1)
#         batch = batch[:, 0]
#         coords = coords.float()
#         coords[:, 0] = (coords[:, 0] / seg.shape[2] - 0.5) * 2
#         coords[:, 1] = (coords[:, 1] / seg.shape[3] - 0.5) * 2

#         pointcloud = -100.0 * torch.ones(seg.shape[0], seg.shape[2]*seg.shape[3], 2).to(seg.device)
#         length = torch.zeros(seg.shape[0]).to(seg.device).long()
#         for i in range(seg.shape[0]):
#             pt = coords[batch == i]
#             pt = pt[torch.randperm(pt.shape[0])] # randomly permute the points
#             pointcloud[i][:pt.shape[0]] = torch.cat([pt[:, 1:], pt[:, :1]], dim=1)
#             length[i] = pt.shape[0]
        
#         return pointcloud, length, weight
    
#     @staticmethod
#     def get_visible_points(points, visibility):
#         batch, indices = torch.nonzero(visibility > 0.0).split([1, 1], dim=1)
#         batch = batch[:, 0]
#         indices = indices[:, 0]

#         length = torch.zeros(points.shape[0]).to(points.device).long()
#         for i in range(points.shape[0]):
#             batch_i = batch == i
#             indices_i = indices[batch_i]
#             points[i][:indices_i.shape[0]] = points[i][indices_i]
#             points[i][indices_i.shape[0]:] = -100.0
#             length[i] = indices_i.shape[0]

#         return points, length


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# def _validate_chamfer_reduction_inputs(
#     batch_reduction: Union[str, None], point_reduction: str
# ):
#     """Check the requested reductions are valid.
#     Args:
#         batch_reduction: Reduction operation to apply for the loss across the
#             batch, can be one of ["mean", "sum"] or None.
#         point_reduction: Reduction operation to apply for the loss across the
#             points, can be one of ["mean", "sum"].
#     """
#     if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
#         raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
#     if point_reduction not in ["mean", "sum"]:
#         raise ValueError('point_reduction must be one of ["mean", "sum"]')


# def _handle_pointcloud_input(
#     points: Union[torch.Tensor, Pointclouds],
#     lengths: Union[torch.Tensor, None],
#     normals: Union[torch.Tensor, None],
# ):
#     """
#     If points is an instance of Pointclouds, retrieve the padded points tensor
#     along with the number of points per batch and the padded normals.
#     Otherwise, return the input points (and normals) with the number of points per cloud
#     set to the size of the second dimension of `points`.
#     """
#     if isinstance(points, Pointclouds):
#         X = points.points_padded()
#         lengths = points.num_points_per_cloud()
#         normals = points.normals_padded()  # either a tensor or None
#     elif torch.is_tensor(points):
#         if points.ndim != 3:
#             raise ValueError("Expected points to be of shape (N, P, D)")
#         X = points
#         if lengths is not None and (
#             lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
#         ):
#             raise ValueError("Expected lengths to be of shape (N,)")
#         if lengths is None:
#             lengths = torch.full(
#                 (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
#             )
#         if normals is not None and normals.ndim != 3:
#             raise ValueError("Expected normals to be of shape (N, P, 3")
#     else:
#         raise ValueError(
#             "The input pointclouds should be either "
#             + "Pointclouds objects or torch.Tensor of shape "
#             + "(minibatch, num_points, 3)."
#         )
#     return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    num_neighbours=1,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    Returns:
        2-element tuple containing
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=num_neighbours)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=num_neighbours)

    cham_x = x_nn.dists.mean(-1)  # (N, P1)
    cham_y = y_nn.dists.mean(-1)  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals
import torch
import torch.nn.functional as F
from torch import nn

from typing import Union



class SegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'bce_with_logits'):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, 
                pred_seg_logits: Union[torch.Tensor, list], 
                target_segs: Union[torch.Tensor, list]) -> torch.Tensor:
        if isinstance(pred_seg_logits, list):
            # Concat alongside the batch axis
            pred_seg_logits = torch.cat(pred_seg_logits)
            target_segs = torch.cat(target_segs)

        if target_segs.shape[2] != pred_seg_logits.shape[2]:
            target_segs = F.interpolate(target_segs, size=pred_seg_logits.shape[2:], mode='bilinear')

        if self.loss_type == 'bce_with_logits':
            loss = self.criterion(pred_seg_logits, target_segs)
        
        elif self.loss_type == 'dice':
            pred_segs = torch.sigmoid(pred_seg_logits)

            intersection = (pred_segs * target_segs).view(pred_segs.shape[0], -1)
            cardinality = (pred_segs**2 + target_segs**2).view(pred_segs.shape[0], -1)
            loss = 1 - ((2. * intersection.mean(1)) / (cardinality.mean(1) + 1e-7)).mean(0)

        return loss


class MultiScaleSilhouetteLoss(nn.Module):
    def __init__(self, num_scales: int = 1, loss_type: str = 'bce'):
        super().__init__()
        self.num_scales = num_scales
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            self.loss = nn.BCELoss()
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        original_size = targets.size()[-1]
        loss = 0.0
        for i in range(self.num_scales):
            if i > 0:
                x = F.interpolate(inputs, size=original_size // (2 ** i))
                gt = F.interpolate(targets, size=original_size // (2 ** i))
            else:
                x = inputs
                gt = targets
            
            if self.loss_type == 'iou':
                intersection = (x * gt).view(x.shape[0], -1)
                union = (x + gt).view(x.shape[0], -1)
                loss += 1 - (intersection.mean(1) / (union - intersection).mean(1)).mean(0)
            
            elif self.loss_type == 'mse':
                loss += ((x - gt)**2).mean() * 0.5

            elif self.loss_type == 'bce':
                loss += self.loss(x, gt.float())
            elif self.loss_type == 'mse':
                loss += self.loss(x, gt.float())
        return loss / self.num_scales
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class LipClosureLoss(nn.Module):
    def __init__(self):
        super(LipClosureLoss, self).__init__()
        self.register_buffer('upper_lips', torch.LongTensor([61, 62, 63]), persistent=False)
        self.register_buffer('lower_lips', torch.LongTensor([67, 66, 65]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lips] - pred_keypoints[:, self.lower_lips]
        diff = keypoints[:, self.upper_lips] - keypoints[:, self.lower_lips]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class EyeClosureLoss(nn.Module):
    def __init__(self):
        super(EyeClosureLoss, self).__init__()
        self.register_buffer('upper_lids', torch.LongTensor([37, 38, 43, 44]), persistent=False)
        self.register_buffer('lower_lids', torch.LongTensor([41, 40, 47, 46]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lids] - pred_keypoints[:, self.lower_lids]
        diff = keypoints[:, self.upper_lids] - keypoints[:, self.lower_lids]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from typing import Union



class HeadPoseMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l2'):
        super(HeadPoseMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                pred_thetas: Union[torch.Tensor, list], 
                target_thetas: Union[torch.Tensor, list]) -> torch.Tensor:
        loss = 0

        if isinstance(pred_thetas, torch.Tensor):
            pred_thetas = [pred_thetas]
            target_thetas = [target_thetas]

        for pred_theta, target_theta in zip(pred_thetas, target_thetas):
            if self.loss_type == 'l1':
                loss += (pred_theta - target_theta).abs().mean()
            elif self.loss_type == 'l2':
                loss += ((pred_theta - target_theta)**2).mean()

        return loss
import torch
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
import cv2
import numpy as np

# from typing import Union
# from typing import Tuple, List
# from rt_gene.estimate_gaze_pytorch import GazeEstimator
# from rt_gene import FaceBox



# class GazeLoss(object):
#     def __init__(self,
#                  device: str,
#                  gaze_model_types: Union[List[str], str] = ['vgg16',],
#                  criterion: str = 'l1',
#                  interpolate: bool = False,
#                  layer_indices: tuple = (1, 6, 11, 18, 25),
# #                  layer_indices: tuple = (4, 5, 6, 7), # for resnet 
# #                  weights: tuple = (2.05625e-3, 2.78125e-4, 5.125e-5, 6.575e-8, 9.67e-10)
# #                  weights: tuple = (1.0, 1e-1, 4e-3, 2e-6, 1e-8),
# #                  weights: tuple = (0.0625, 0.125, 0.25, 1.0),
#                  weights: tuple = (0.03125, 0.0625, 0.125, 0.25, 1.0),
#                  ) -> None:
#         super(GazeLoss, self).__init__()
#         self.len_features = len(layer_indices)
#         # checkpoints_paths_dict = {'vgg16':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/vgg_16_2_forward_sum.pt', 'resnet18':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/resnet_18_2_forward_sum.pt'}
#         # if interpolate:
#         checkpoints_paths_dict = {'vgg16': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt',
#                                 'resnet18': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt'}
            
#         self.gaze_estimator = GazeEstimator(device=device,
#                                               model_nets_path=[checkpoints_paths_dict[m] for m in gaze_model_types],
#                                               gaze_model_types=gaze_model_types,
#                                               interpolate = interpolate,
#                                               align_face=True)

#         if criterion == 'l1':
#             self.criterion = nn.L1Loss()
#         elif criterion == 'l2':
#             self.criterion = nn.MSELoss()

#         self.layer_indices = layer_indices
#         self.weights = weights

#     @torch.cuda.amp.autocast(False)
#     def forward(self,
#                 inputs: Union[torch.Tensor, list],
#                 target: torch.Tensor,
#                 keypoints: torch.Tensor = None,
#                 interpolate=True) -> Union[torch.Tensor, list]:
#         if isinstance(inputs, list):
#             # Concat alongside the batch axis
#             input_is_a_list = True
#             num_chunks = len(inputs)
#             chunk_size = inputs[0].shape[0]
#             inputs = torch.cat(inputs)

#         else:
#             input_is_a_list = False
            
#         if interpolate:   
#             inputs = F.interpolate(inputs, (224, 224), mode='bicubic', align_corners=False)
#             target = F.interpolate(target, (224, 224), mode='bicubic', align_corners=False)
        
#         if keypoints is not None:
#             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
# #             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
# #             keypoints_np = [(kp[:, :2].cpu().numpy()/tgt.shape[2]*224).astype(np.int32) for kp, tgt in zip(keypoints, target)]
            
#             faceboxes = [FaceBox(left=kp[:, 0].min(),
#                                                top=kp[:, 1].min(),
#                                                right=kp[:, 0].max(),
#                                                bottom=kp[:, 1].max()) for kp in keypoints_np]
#         else:
#             faceboxes = None
#             keypoints_np = None

#         target = target.float()
#         inputs = inputs.float()

#         with torch.no_grad():
#             target_subjects = self.gaze_estimator.get_eye_embeddings(target,
#                                                                      self.layer_indices,
#                                                                      faceboxes,
#                                                                      keypoints_np)

#         # Filter subjects with visible eyes
#         visible_eyes = [subject is not None and subject.eye_embeddings is not None for subject in target_subjects]

#         if not any(visible_eyes):
#             return torch.zeros(1).to(target.device)

#         target_subjects = self.select_by_mask(target_subjects, visible_eyes)

#         faceboxes = [subject.box for subject in target_subjects]
#         keypoints_np = [subject.landmarks for subject in target_subjects]

#         target_features = [[] for i in range(self.len_features)]
#         for subject in target_subjects:
#             for k in range(self.len_features):
#                 target_features[k].append(subject.eye_embeddings[k])
#         target_features = [torch.cat(feats) for feats in target_features]

#         eye_masks = self.draw_eye_masks(keypoints_np, target.shape[2], target.device)

#         if input_is_a_list:
#             visible_eyes *= num_chunks
#             faceboxes *= num_chunks
#             keypoints_np *= num_chunks
#             eye_masks = torch.cat([eye_masks] * num_chunks)

#         # Grads are masked
#         inputs = inputs[visible_eyes]
# #         inputs.retain_grad() # turn it on while debugging
#         inputs_ = inputs * eye_masks + inputs.detach() * (1 - eye_masks)
# #         inputs_.retain_grad() # turn it on while debugging
        
#         # In order to apply eye masks for gradients, first calc the grads
# #         inputs_ = inputs.detach().clone().requires_grad_()
# #         inputs_ = inputs_ * eye_masks + inputs_.detach() * (1 - eye_masks)
#         input_subjects = self.gaze_estimator.get_eye_embeddings(inputs_,
#                                                                 self.layer_indices,
#                                                                 faceboxes,
#                                                                 keypoints_np)

#         input_features = [[] for i in range(self.len_features)]
#         for subject in input_subjects:
#             for k in range(self.len_features):
#                 input_features[k].append(subject.eye_embeddings[k])
#         input_features = [torch.cat(feats) for feats in input_features]

#         loss = 0

#         for input_feature, target_feature, weight in zip(input_features, target_features, self.weights):
#             if input_is_a_list:
#                 target_feature = torch.cat([target_feature.detach()] * num_chunks)

#             loss += weight * self.criterion(input_feature, target_feature)
        
#         return loss

#     @staticmethod
#     def select_by_mask(a, mask):
#         return [v for (is_true, v) in zip(mask, a) if is_true]

#     @staticmethod
#     def draw_eye_masks(keypoints_np, image_size, device):
#         ### Define drawing options ###
#         edges_parts = [list(range(36, 42)), list(range(42, 48))]

#         mask_kernel = np.ones((5, 5), np.uint8)

#         ### Start drawing ###
#         eye_masks = []

#         for xy in keypoints_np:
#             xy = xy[None, :, None].astype(np.int32)

#             eye_mask = np.zeros((image_size, image_size, 3), np.uint8)

#             for edges in edges_parts:
#                 eye_mask = cv2.fillConvexPoly(eye_mask, xy[0, edges], (255, 255, 255))

#             eye_mask = cv2.dilate(eye_mask, mask_kernel, iterations=1)
#             eye_mask = cv2.blur(eye_mask, mask_kernel.shape)
#             eye_mask = torch.FloatTensor(eye_mask[:, :, [0]].transpose(2, 0, 1)) / 255.
#             eye_masks.append(eye_mask)

#         eye_masks = torch.stack(eye_masks).to(device)

#         return eye_masks


# cherry pick from  metaportrait 
import torchvision.models as models
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        self.mean = self.mean.to(X.device)
        self.std = self.std.to(X.device)

        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

import os

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

import torch

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)

class MySixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(MySixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)  # Call the function to create an instance
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        rotation_6d = x[:, :6]
        translation = x[:, 6:]
        rotation_matrix = compute_rotation_matrix_from_ortho6d(rotation_6d)
        return rotation_matrix, translation


class SixDRepNet2(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      


        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = compute_rotation_matrix_from_ortho6d(x)

        return out

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

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

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R



def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

    
class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
     
        R = get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])


        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] # * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2] # * 180 / np.pi

        # Gray images

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        #mu, sigma = 0, 0.01 
        #noise = np.random.normal(mu, sigma, [3,3])
        #print(noise) 

        # Get target tensors
        R = get_R(pitch, yaw, roll)#+ noise
        
        #labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img = self.transform(img)

        return img,  torch.FloatTensor(R),[], self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

def getDataset(dataset, data_dir, filename_list, transformations, train_mode = True):
    if dataset == 'Pose_300W_LP':
            pose_dataset = Pose_300W_LP(
                data_dir, filename_list, transformations)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000(
            data_dir, filename_list, transformations)
    elif dataset == 'BIWI':
        pose_dataset = BIWI(
            data_dir, filename_list, transformations, train_mode= train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(
            data_dir, filename_list, transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(
            data_dir, filename_list, transformations)
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset

import time
import math
import re
import sys
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math

import torch
from torch import nn




import os
import math
from math import cos, sin

import numpy as np
import torch
#from torch.serialization import load_lua
import scipy.io as sio
import cv2


## Amir Shahroudy  
# https://github.com/shahroudy

import os
import sys
import argparse

import numpy as np




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Create filenames list txt file from datasets root dir.'
        ' For head pose analysis.')
    parser.add_argument('--root_dir = ', 
        dest='root_dir', 
        help='root directory of the datasets files', 
        default='./datasets/300W_LP', 
        type=str)
    parser.add_argument('--filename', 
        dest='filename', 
        help='Output filename.',
        default='files.txt', 
        type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    os.chdir(args.root_dir)

    file_counter = 0
    rej_counter = 0
    outfile = open(args.filename, 'w')

    for root, dirs, files in os.walk('.'): 
        for f in files: 
            if f[-4:] == '.jpg': 
                mat_path = os.path.join(root, f.replace('.jpg', '.mat'))
                # We get the pose in radians
                pose = get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi

                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    if file_counter > 0:
                        outfile.write('\n')
                    outfile.write(root + '/' + f[:-4])
                    file_counter += 1
                else:
                   rej_counter += 1

    outfile.close()
    print(f'{file_counter} files listed! {rej_counter} files had out-of-range'
        f' values and kept out of the list!')
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))


"""
6DRepNet.

Accurate and unconstrained head pose estimation.
"""

__version__ = "0.1.6"
__author__ = 'Thorsten Hempel'

from math import cos, sin

import torch
from torch.hub import load_state_dict_from_url
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np





class SixDRepNet_Detector():

    def __init__(self, gpu_id : int=0, dict_path: str=''):
        """
        Constructs the SixDRepNet instance with all necessary attributes.

        Parameters
        ----------
            gpu:id : int
                gpu identifier, for selecting cpu set -1
            dict_path : str
                Path for local weight file. Leaving it empty will automatically download a finetuned weight file.
        """

        self.gpu = gpu_id
        self.model = MySixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False)
        # Load snapshot
        if dict_path=='':
            saved_state_dict = load_state_dict_from_url("https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth")    
        else:
            saved_state_dict = torch.load(dict_path)

        self.model.eval()
        self.model.load_state_dict(saved_state_dict)
        
        if self.gpu != -1:
            self.model.cuda(self.gpu)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def predict(self, img):
        """
        Predicts the persons head pose and returning it in euler angles.

        Parameters
        ----------
        img : array 
            Face crop to be predicted

        Returns
        -------
        pitch, yaw, roll
        """


        if self.gpu != -1:
            img = img.cuda(self.gpu)
     
        rotations,translations  = self.model(img)
        
        euler = compute_euler_angles_from_rotation_matrices(rotations)*180/np.pi
        # p = euler[:, 0].cpu().detach().numpy()
        # y = euler[:, 1].cpu().detach().numpy()
        # r = euler[:, 2].cpu().detach().numpy()

        return euler,translations


    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        img : array
            Target image to be drawn on
        yaw : int
            yaw rotation
        pitch: int
            pitch rotation
        roll: int
            roll rotation
        tdx : int , optional
            shift on x axis
        tdy : int , optional
            shift on y axis
            
        Returns
        -------
        img : array
        """

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

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

        return img



import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
# import matplotlib
# from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')





def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = MySixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    train_loader = torch.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)

            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )
        
        if b_scheduler:
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.tar')
                  )


import copy

import torch.nn as nn
import numpy as np
import torch



def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)

def create_RepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
'RepVGG-D2se': create_RepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]



#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F

#   https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

