import torch
import torch.nn.functional as F


def rotation_matrix_x(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, z, s], 2),
            torch.cat([z, o, z], 2),
            torch.cat([-s, z, c], 2),
        ],
        1,
    )


def rotation_matrix_y(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([o, z, z], 2),
            torch.cat([z, c, -s], 2),
            torch.cat([z, s, c], 2),
        ],
        1,
    )


def rotation_matrix_z(theta):
    theta = theta.reshape(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, -s, z], 2),
            torch.cat([s, c, z], 2),
            torch.cat([z, z, o], 2),
        ],
        1,
    )


def transform_kp(canonical_kp, yaw, pitch, roll, t, delta):
    # [N,K,3] [N,] [N,] [N,] [N,3] [N,K,3]
    # y, x, z
    # w, h, d
    rot_mat = rotation_matrix_y(pitch) @ rotation_matrix_x(yaw) @ rotation_matrix_z(roll)
    transformed_kp = torch.matmul(rot_mat.unsqueeze(1), canonical_kp.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1) + delta
    return transformed_kp, rot_mat


def transform_kp_with_new_pose(canonical_kp, yaw, pitch, roll, t, delta, new_yaw, new_pitch, new_roll):
    # [N,K,3] [N,] [N,] [N,] [N,3] [N,K,3]
    # y, x, z
    # w, h, d
    old_rot_mat = rotation_matrix_y(pitch) @ rotation_matrix_x(yaw) @ rotation_matrix_z(roll)
    rot_mat = rotation_matrix_y(new_pitch) @ rotation_matrix_x(new_yaw) @ rotation_matrix_z(new_roll)
    R = torch.matmul(rot_mat, torch.inverse(old_rot_mat))
    transformed_kp = (
        torch.matmul(rot_mat.unsqueeze(1), canonical_kp.unsqueeze(-1)).squeeze(-1)
        + t.unsqueeze(1)
        + torch.matmul(R.unsqueeze(1), delta.unsqueeze(-1)).squeeze(-1)
    )
    zt = 0.33 - transformed_kp[:, :, 2].mean()
    transformed_kp = transformed_kp + torch.FloatTensor([0, 0, zt]).cuda()
    return transformed_kp, rot_mat


def make_coordinate_grid_2d(spatial_size):
    h, w = spatial_size
    x = torch.arange(h).cuda()
    y = torch.arange(w).cuda()
    x = 2 * (x / (h - 1)) - 1
    y = 2 * (y / (w - 1)) - 1
    xx = x.reshape(-1, 1).repeat(1, w)
    yy = y.reshape(1, -1).repeat(h, 1)
    meshed = torch.cat([yy.unsqueeze(2), xx.unsqueeze(2)], 2)
    return meshed


def make_coordinate_grid_3d(spatial_size):
    d, h, w = spatial_size
    z = torch.arange(d).cuda()
    x = torch.arange(h).cuda()
    y = torch.arange(w).cuda()
    z = 2 * (z / (d - 1)) - 1
    x = 2 * (x / (h - 1)) - 1
    y = 2 * (y / (w - 1)) - 1
    zz = z.reshape(-1, 1, 1).repeat(1, h, w)
    xx = x.reshape(1, -1, 1).repeat(d, 1, w)
    yy = y.reshape(1, 1, -1).repeat(d, h, 1)
    meshed = torch.cat([yy.unsqueeze(3), xx.unsqueeze(3), zz.unsqueeze(3)], 3)
    return meshed


def out2heatmap(out, temperature=0.1):
    final_shape = out.shape
    heatmap = out.reshape(final_shape[0], final_shape[1], -1)
    heatmap = F.softmax(heatmap / temperature, dim=2)
    heatmap = heatmap.reshape(*final_shape)
    return heatmap


def heatmap2kp(heatmap):
    shape = heatmap.shape
    grid = make_coordinate_grid_3d(shape[2:]).unsqueeze(0).unsqueeze(0)
    kp = (heatmap.unsqueeze(-1) * grid).sum(dim=(2, 3, 4))
    return kp


def kp2gaussian_2d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_2d(spatial_size).reshape(1, 1, *spatial_size, 2).repeat(N, K, 1, 1, 1)
    mean = kp.reshape(N, K, 1, 1, 2)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def kp2gaussian_3d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_3d(spatial_size).reshape(1, 1, *spatial_size, 3).repeat(N, K, 1, 1, 1, 1)
    mean = kp.reshape(N, K, 1, 1, 1, 3)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def create_heatmap_representations(fs, kp_s, kp_d):
    spatial_size = fs.shape[2:]
    heatmap_d = kp2gaussian_3d(kp_d, spatial_size)
    heatmap_s = kp2gaussian_3d(kp_s, spatial_size)
    heatmap = heatmap_d - heatmap_s
    zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size).cuda()
    # [N,21,16,64,64]
    heatmap = torch.cat([zeros, heatmap], dim=1)
    # [N,21,1,16,64,64]
    heatmap = heatmap.unsqueeze(2)
    return heatmap


def create_sparse_motions(fs, kp_s, kp_d, Rs, Rd):
    N, _, D, H, W = fs.shape
    K = kp_s.shape[1]
    identity_grid = make_coordinate_grid_3d((D, H, W)).reshape(1, 1, D, H, W, 3).repeat(N, 1, 1, 1, 1, 1)
    # [N,20,16,64,64,3]
    coordinate_grid = identity_grid.repeat(1, K, 1, 1, 1, 1) - kp_d.reshape(N, K, 1, 1, 1, 3)
    # [N,1,1,1,1,3,3]
    jacobian = torch.matmul(Rs, torch.inverse(Rd)).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
    coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1)).squeeze(-1)
    driving_to_source = coordinate_grid + kp_s.reshape(N, K, 1, 1, 1, 3)
    sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
    # sparse_motions = driving_to_source
    # [N,21,16,64,64,3]
    return sparse_motions

def create_deformed_source_image2d(fs, sparse_motions):
    N, _, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    # [N*21,4,16,64,64]
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1).reshape(N * (K + 1), -1, H, W)
    # [N*21,16,64,64,3]
    sparse_motions = sparse_motions.reshape((N * (K + 1), H, W, -1))
    # [N*21,4,16,64,64]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = sparse_deformed.reshape((N, K + 1, -1, H, W))
    # [N,21,4,16,64,64]
    return sparse_deformed

def create_deformed_source_image(fs, sparse_motions):
    N, _, D, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    # [N*21,4,16,64,64]
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1, 1).reshape(N * (K + 1), -1, D, H, W)
    # [N*21,16,64,64,3]
    sparse_motions = sparse_motions.reshape((N * (K + 1), D, H, W, -1))
    # [N*21,4,16,64,64]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = sparse_deformed.reshape((N, K + 1, -1, D, H, W))
    # [N,21,4,16,64,64]
    return sparse_deformed


def apply_imagenet_normalization(input):
    mean = input.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    output = (input - mean) / std
    return output


def apply_vggface_normalization(input):
    mean = input.new_tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).reshape(1, 3, 1, 1)
    std = input.new_tensor([1, 1, 1]).reshape(1, 3, 1, 1)
    output = (input * 255 - mean) / std
    return output
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

class _ConvBlock(nn.Module):
    def __init__(self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, dim, activation_type, nonlinearity_type):
        # the default weight norm is spectral norm
        # pattern: C for conv, N for activation norm(SyncBatchNorm), A for nonlinearity(ReLU)
        super().__init__()
        norm_channels = out_channels if pattern.find("C") < pattern.find("N") else in_channels
        weight_norm = spectral_norm if use_weight_norm else lambda x: x
        base_conv = nn.Conv2d if dim == 2 else nn.Conv3d

        def _get_activation():
            if activation_type == "batch":
                return nn.SyncBatchNorm(norm_channels)
            elif activation_type == "instance":
                return nn.InstanceNorm2d(norm_channels, affine=True) if dim == 2 else nn.InstanceNorm3d(norm_channels, affine=True)
            elif activation_type == "none":
                return nn.Identity()

        def _get_nonlinearity():
            if nonlinearity_type == "relu":
                return nn.ReLU(inplace=True)
            elif nonlinearity_type == "leakyrelu":
                return nn.LeakyReLU(0.2, inplace=True)

        mappings = {
            "C": weight_norm(base_conv(in_channels, out_channels, kernel_size, stride, padding)),
            "N": _get_activation(),
            "A": _get_nonlinearity(),
        }

        module_list = []
        for c in pattern:
            module_list.append(mappings[c])
        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        return self.layers(x)


class ConvBlock2D(_ConvBlock):
    def __init__(
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu",
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 2, activation_type, nonlinearity_type)


class ConvBlock3D(_ConvBlock):
    def __init__(
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu",
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 3, activation_type, nonlinearity_type)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv, base_pooling, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(base_conv("CNA", in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_weight_norm=use_weight_norm), base_pooling(kernel_size))

    def forward(self, x):
        return self.layers(x)


class DownBlock2D(_DownBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock2D, nn.AvgPool2d, (2, 2))


class DownBlock3D(_DownBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock3D, nn.AvgPool3d, (1, 2, 2))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv, scale_factor):
        super().__init__()
        self.layers = nn.Sequential(nn.Upsample(scale_factor=scale_factor), base_conv("CNA", in_channels, out_channels, 3, 1, 1, use_weight_norm))

    def forward(self, x):
        return self.layers(x)


class UpBlock2D(_UpBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock2D, (2, 2))


class UpBlock3D(_UpBlock):
    def __init__(self, in_channels, out_channels, use_weight_norm):
        super().__init__(in_channels, out_channels, use_weight_norm, ConvBlock3D, (1, 2, 2))


class _ResBlock(nn.Module):
    def __init__(self, in_channels, use_weight_norm, base_block):
        super().__init__()
        self.layers = nn.Sequential(
            base_block("NAC", in_channels, in_channels, 3, 1, 1, use_weight_norm),
            base_block("NAC", in_channels, in_channels, 3, 1, 1, use_weight_norm),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResBlock2D(_ResBlock):
    def __init__(self, in_channels, use_weight_norm):
        super().__init__(in_channels, use_weight_norm, ConvBlock2D)


class ResBlock3D(_ResBlock):
    def __init__(self, in_channels, use_weight_norm):
        super().__init__(in_channels, use_weight_norm, ConvBlock3D)


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_weight_norm):
        super().__init__()
        self.down_sample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.down_sample = ConvBlock2D("CN", in_channels, out_channels, 1, stride, 0, use_weight_norm)
        self.layers = nn.Sequential(
            ConvBlock2D("CNA", in_channels, out_channels // 4, 1, 1, 0, use_weight_norm),
            ConvBlock2D("CNA", out_channels // 4, out_channels // 4, 3, stride, 1, use_weight_norm),
            ConvBlock2D("CN", out_channels // 4, out_channels, 1, 1, 0, use_weight_norm),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.down_sample(x) + self.layers(x))
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import nn
from modules.real3d.facev2v_warp.func_utils import apply_imagenet_normalization, apply_vggface_normalization


@torch.jit.script
def fuse_math_min_mean_pos(x):
    r"""Fuse operation min mean for hinge loss computation of positive
    samples"""
    minval = torch.min(x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


@torch.jit.script
def fuse_math_min_mean_neg(x):
    r"""Fuse operation min mean for hinge loss computation of negative
    samples"""
    minval = torch.min(-x - 1, x * 0)
    loss = -torch.mean(minval)
    return loss


class _PerceptualNetwork(nn.Module):
    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        self.network = network.cuda()
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _vgg19(layers):
    network = torchvision.models.vgg19()
    state_dict = torch.utils.model_zoo.load_url(
        "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", map_location=torch.device("cpu"), progress=True
    )
    network.load_state_dict(state_dict)
    network = network.features
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        17: "relu_3_4",
        20: "relu_4_1",
        22: "relu_4_2",
        24: "relu_4_3",
        26: "relu_4_4",
        29: "relu_5_1",
    }
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face(layers):
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/" "vgg_face_dag.pth", map_location=torch.device("cpu"), progress=True
    )
    feature_layer_name_mapping = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        17: "conv4_1",
        19: "conv4_2",
        21: "conv4_3",
        24: "conv5_1",
        26: "conv5_2",
        28: "conv5_3",
    }
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict["features." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["features." + str(k) + ".bias"] = state_dict[v + ".bias"]
    classifier_layer_name_mapping = {0: "fc6", 3: "fc7", 6: "fc8"}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict["classifier." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["classifier." + str(k) + ".bias"] = state_dict[v + ".bias"]
    network.load_state_dict(new_state_dict)
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1",
    }
    return _PerceptualNetwork(network.features, layer_name_mapping, layers)


class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        layers_weight={"relu_1_1": 0.03125, "relu_2_1": 0.0625, "relu_3_1": 0.125, "relu_4_1": 0.25, "relu_5_1": 1.0}, 
        n_scale=3,
        vgg19_loss_weight=1.0,
        vggface_loss_weight=1.0,
    ):
        super().__init__()
        self.vgg19 = _vgg19(layers_weight.keys())
        self.vggface = _vgg_face(layers_weight.keys())
        self.mse_criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.layers_weight, self.n_scale = layers_weight, n_scale
        self.vgg19_loss_weight = vgg19_loss_weight
        self.vggface_loss_weight = vggface_loss_weight
        self.vgg19.eval()
        self.vggface.eval()

    def forward(self, input, target):
        """
        input: [B, 3, H, W] in 0.~1. scale
        """
        if input.shape[-1] != 512:
            assert input.ndim == 4
            input = F.interpolate(input, mode="bilinear", size=(512,512), antialias=True, align_corners=False)
            target = F.interpolate(target, mode="bilinear", size=(512,512), antialias=True, align_corners=False)

        self.vgg19.eval()
        self.vggface.eval()
        loss = 0
        features_vggface_input = self.vggface(apply_vggface_normalization(input))
        features_vggface_target = self.vggface(apply_vggface_normalization(target))
        input = apply_imagenet_normalization(input)
        target = apply_imagenet_normalization(target)
        features_vgg19_input = self.vgg19(input)
        features_vgg19_target = self.vgg19(target)
        for layer, weight in self.layers_weight.items():
            tmp = self.vggface_loss_weight * weight * self.criterion(features_vggface_input[layer], features_vggface_target[layer].detach()) / 255
            if not torch.any(torch.isnan(tmp)):
                loss += tmp
            else:
                loss += torch.zeros_like(tmp)
            tmp = self.vgg19_loss_weight * weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
            if not torch.any(torch.isnan(tmp)):
                loss += tmp
            else:
                loss += torch.zeros_like(tmp)
        for i in range(self.n_scale):
            input = F.interpolate(input, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            target = F.interpolate(target, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            features_vgg19_input = self.vgg19(input)
            features_vgg19_target = self.vgg19(target)
            tmp = weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
            if not torch.any(torch.isnan(tmp)):
                loss += tmp
            else:
                loss += torch.zeros_like(tmp)
        return loss


class GANLoss(nn.Module):
    # Update generator: gan_loss(fake_output, True, False) + other losses
    # Update discriminator: gan_loss(fake_output(detached), False, True) + gan_loss(real_output, True, True)
    def __init__(self):
        super().__init__()

    def forward(self, dis_output, t_real, dis_update=True):
        r"""GAN loss computation.
        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise
                uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the
                discriminator, otherwise the generator.
        Returns:
            loss (tensor): Loss value.
        """

        if dis_update:
            if t_real:
                loss = fuse_math_min_mean_pos(dis_output)
            else:
                loss = fuse_math_min_mean_neg(dis_output)
        else:
            loss = -torch.mean(dis_output)
        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_features, real_features):
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss


class EquivarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, kp_d, reverse_kp):
        loss = self.criterion(kp_d[:, :, :2], reverse_kp)
        return loss


class KeypointPriorLoss(nn.Module):
    def __init__(self, Dt=0.1, zt=0.33):
        super().__init__()
        self.Dt, self.zt = Dt, zt

    def forward(self, kp_d):
        # use distance matrix to avoid loop
        dist_mat = torch.cdist(kp_d, kp_d).square()
        loss = (
            torch.max(0 * dist_mat, self.Dt - dist_mat).sum((1, 2)).mean()
            + torch.abs(kp_d[:, :, 2].mean(1) - self.zt).mean()
            - kp_d.shape[1] * self.Dt
        )
        return loss


class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, yaw, pitch, roll, real_yaw, real_pitch, real_roll):
        loss = (self.criterion(yaw, real_yaw.detach()) + self.criterion(pitch, real_pitch.detach()) + self.criterion(roll, real_roll.detach())) / 3
        return loss / np.pi * 180


class DeformationPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_d):
        loss = delta_d.abs().mean()
        return loss


if __name__ == '__main__':
    loss_fn = PerceptualLoss()
    x1 = torch.randn([4, 3, 512, 512]).cuda()
    x2 = torch.randn([4, 3, 512, 512]).cuda()
    loss = loss_fn(x1, x2)import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import copy 

from modules.real3d.facev2v_warp.network2 import AppearanceFeatureExtractor, CanonicalKeypointDetector, PoseExpressionEstimator, MotionFieldEstimator, Generator
from modules.real3d.facev2v_warp.func_utils import transform_kp, make_coordinate_grid_2d, apply_imagenet_normalization
from modules.real3d.facev2v_warp.losses import PerceptualLoss, GANLoss, FeatureMatchingLoss, EquivarianceLoss, KeypointPriorLoss, HeadPoseLoss, DeformationPriorLoss
from utils.commons.image_utils import erode, dilate
from utils.commons.hparams import hparams


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll


class Transform:
    """
    Random tps transformation for equivariance constraints.
    reference: FOMM
    """

    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0, std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners=True, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.type(coordinates.type())
        control_params = self.control_params.type(coordinates.type())
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed


class WarpBasedTorsoModel(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.appearance_extractor = AppearanceFeatureExtractor(model_scale)
        self.canonical_kp_detector = CanonicalKeypointDetector(model_scale)
        self.pose_exp_estimator = PoseExpressionEstimator(model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale)
        self.deform_based_generator = Generator()

        self.pretrained_hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66).cuda()
        pretrained_path = "/home/tiger/nfs/myenv/cache/useful_ckpts/hopenet_robust_alpha1.pkl" # https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR
        self.pretrained_hopenet.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        self.pretrained_hopenet.requires_grad_(False)

        self.pose_loss_fn = HeadPoseLoss() # 20
        self.equivariance_loss_fn = EquivarianceLoss() # 20
        self.keypoint_prior_loss_fn = KeypointPriorLoss()# 10
        self.deform_prior_loss_fn = DeformationPriorLoss() # 5

    def forward(self, torso_src_img, src_img, drv_img, cal_loss=False):
        # predict cano keypoint
        cano_keypoint = self.canonical_kp_detector(src_img)
        # predict src_pose and drv_pose
        transform_fn = Transform(drv_img.shape[0])
        transformed_drv_img = transform_fn.transform_frame(drv_img)
        cat_imgs = torch.cat([src_img, drv_img, transformed_drv_img], dim=0)
        yaw, pitch, roll, t, delta = self.pose_exp_estimator(cat_imgs)
        [yaw_s, yaw_d, yaw_tran], [pitch_s, pitch_d, pitch_tran], [roll_s, roll_d, roll_tran] = (
            torch.chunk(yaw, 3, dim=0),
            torch.chunk(pitch, 3, dim=0),
            torch.chunk(roll, 3, dim=0),
        )
        [t_s, t_d, t_tran], [delta_s, delta_d, delta_tran] = (
            torch.chunk(t, 3, dim=0),
            torch.chunk(delta, 3, dim=0),
        )
        kp_s, Rs = transform_kp(cano_keypoint, yaw_s, pitch_s, roll_s, t_s, delta_s)
        kp_d, Rd = transform_kp(cano_keypoint, yaw_d, pitch_d, roll_d, t_d, delta_d)
        # deform the torso img
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        deformation, occlusion = self.motion_field_estimator(torso_appearance_feats, kp_s, kp_d, Rs, Rd)
        deformed_torso_img = self.deform_based_generator(torso_appearance_feats, deformation, occlusion)
        
        ret = {'kp_src': kp_s, 'kp_drv': kp_d}
        if cal_loss:
            losses = {}
            with torch.no_grad():
                self.pretrained_hopenet.eval()
                real_yaw, real_pitch, real_roll = self.pretrained_hopenet(F.interpolate(apply_imagenet_normalization(cat_imgs), size=(224, 224)))
            pose_loss = self.pose_loss_fn(yaw, pitch, roll, real_yaw, real_pitch, real_roll)
            losses['facev2v/pose_pred_loss'] = pose_loss

            kp_tran, _ = transform_kp(cano_keypoint, yaw_tran, pitch_tran, roll_tran, t_tran, delta_tran)
            reverse_kp = transform_fn.warp_coordinates(kp_tran[:, :, :2])
            equivariance_loss = self.equivariance_loss_fn(kp_d, reverse_kp)
            losses['facev2v/equivariance_loss'] = equivariance_loss

            keypoint_prior_loss = self.keypoint_prior_loss_fn(kp_d)
            losses['facev2v/keypoint_prior_loss'] = keypoint_prior_loss

            deform_prior_loss = self.deform_prior_loss_fn(delta_d)
            losses['facev2v/deform_prior_loss'] = deform_prior_loss
            ret['losses'] = losses

        return deformed_torso_img, ret


class WarpBasedTorsoModelMediaPipe(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.hparams = copy.deepcopy(hparams)
        if hparams.get("torso_inp_mode", "rgb") == 'rgb_alpha':
            torso_in_dim = 5
        else:
            torso_in_dim = 3
        self.appearance_extractor = AppearanceFeatureExtractor(in_dim=torso_in_dim, model_scale=model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=self.hparams['torso_kp_num']) # 32 channel appearance channel, and 3 channel for segmap
        # self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=9) # 32 channel appearance channel, and 3 channel for segmap
        self.deform_based_generator = Generator()

        self.occlusion_2_predictor = nn.Sequential(*[
            nn.Conv2d(64+1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        ])

    #  V2, 先warp， 再mean
    def forward(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, tgt_head_weights, cal_loss=False, target_torso_mask=None):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        if hparams.get("torso_inp_mode", "rgb") == 'rgb_alpha':
            torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(torso_src_img.shape[-2],torso_src_img.shape[-1]), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
            torso_src_img = torch.cat([torso_src_img, torso_segmap], dim=1)

        torso_appearance_feats = self.appearance_extractor(torso_src_img) # [B, C, D, H, W]
        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=self.hparams.get("torso_mask_dilate_ksize", 7))
        if self.hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)

        if self.hparams['torso_kp_num'] == 4:
            kp_s = kp_s[:,[0,8,16,27],:]
            kp_d = kp_d[:,[0,8,16,27],:]
        elif self.hparams['torso_kp_num'] == 9:
            kp_s = kp_s[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
            kp_d = kp_d[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
        else:
            raise NotImplementedError()

        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd, tgt_head_img, tgt_head_weights)
        motion_estimator_grad_scale_factor = 0.1
        # motion_estimator_grad_scale_factor = 1.0
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion, a 0~1 mask that predict the segment map of warped torso, used in oclcusion-aware decoder
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion_2, a 0~1 mask that predict the segment map of warped torso, but is used in alpha-blending
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}

        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        occlusion_2 = self.occlusion_2_predictor(torch.cat([deformed_torso_hid, F.interpolate(occlusion_2, size=(256,256), mode='bilinear')], dim=1))
        ret['occlusion_2'] = occlusion_2
        alphas = occlusion_2.clamp(1e-5, 1 - 1e-5) 

        if target_torso_mask is None:
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': occlusion.mean(),
                'facev2v/occlusion_2_reg_l1': occlusion_2.mean(),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        else:
            non_target_torso_mask_1 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion.shape[-2:])
            non_target_torso_mask_2 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion_2.shape[-2:])
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': self.masked_l1_reg_loss(occlusion, non_target_torso_mask_1.bool(), masked_weight=1, unmasked_weight=self.hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_reg_l1': self.masked_l1_reg_loss(occlusion_2, non_target_torso_mask_2.bool(), masked_weight=1, unmasked_weight=self.hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        # if self.hparams.get("fuse_with_deform_source"):
        #     B, _, H, W = deformed_torso_img.shape
        #     deformation_256 = F.interpolate(deformation.mean(dim=1).permute(0,3,1,2), size=256, mode='bilinear',antialias=True).permute(0,2,3,1)[...,:2]
        #     deformed_source_torso_img = F.grid_sample(torso_src_img, deformation_256, align_corners=True).view(B, -1, H, W)
        #     occlusion_256 = F.interpolate(occlusion, size=256, antialias=True, mode='bilinear').reshape([B,1,H,W])
        #     # deformed_torso_img = deformed_torso_img * (1 - occlusion_256[:,0]) + deformed_source_torso_img[:,0] * occlusion_256[:,0]
        #     deformed_torso_img = deformed_torso_img * (1 - occlusion_256) + deformed_source_torso_img * occlusion_256
        return deformed_torso_img, ret

    def masked_l1_reg_loss(self, img_pred, mask, masked_weight=0.01, unmasked_weight=0.001, mode='l1'):
        # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred).abs().sum(dim=1) * weight_mask
        else:
            error = (img_pred).pow(2).sum(dim=1) * weight_mask
        loss = error.mean()
        return loss

    @torch.no_grad()
    def infer_forward_stage1(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, cal_loss=False):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        kp_s = kp_s[:,[0,8,16,27],:]
        kp_d = kp_d[:,[0,8,16,27],:]

        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=self.hparams.get("torso_mask_dilate_ksize", 7))
        if self.hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)
        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd)
        motion_estimator_grad_scale_factor = 0.1
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}
        ret['torso_appearance_feats'] = torso_appearance_feats
        ret['deformation'] = deformation
        ret['occlusion'] = occlusion
        return ret
    
    @torch.no_grad()
    def infer_forward_stage2(self, ret):
        torso_appearance_feats = ret['torso_appearance_feats']
        deformation = ret['deformation']
        occlusion = ret['occlusion']
        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        return deformed_torso_img
    
if __name__ == '__main__':
    from utils.nn.model_utils import num_params
    import tqdm
    model = WarpBasedTorsoModel('small')
    model.cuda()
    num_params(model)
    for n, m in model.named_children():
        num_params(m, model_name=n)
    torso_ref_img = torch.randn([2, 3, 256, 256]).cuda()
    ref_img = torch.randn([2, 3, 256, 256]).cuda()
    mv_img = torch.randn([2, 3, 256, 256]).cuda()
    out = model(torso_ref_img, ref_img, mv_img)
    for i in tqdm.trange(100):
        out_img, losses = model(torso_ref_img, ref_img, mv_img, cal_loss=True)
    print(" ")import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np

from modules.real3d.facev2v_warp.network import AppearanceFeatureExtractor, CanonicalKeypointDetector, PoseExpressionEstimator, MotionFieldEstimator, Generator
from modules.real3d.facev2v_warp.func_utils import transform_kp, make_coordinate_grid_2d, apply_imagenet_normalization
from modules.real3d.facev2v_warp.losses import PerceptualLoss, GANLoss, FeatureMatchingLoss, EquivarianceLoss, KeypointPriorLoss, HeadPoseLoss, DeformationPriorLoss
from utils.commons.image_utils import erode, dilate
from utils.commons.hparams import hparams


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll


class Transform:
    """
    Random tps transformation for equivariance constraints.
    reference: FOMM
    """

    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0, std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners=True, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.type(coordinates.type())
        control_params = self.control_params.type(coordinates.type())
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed


class WarpBasedTorsoModel(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.appearance_extractor = AppearanceFeatureExtractor(model_scale)
        self.canonical_kp_detector = CanonicalKeypointDetector(model_scale)
        self.pose_exp_estimator = PoseExpressionEstimator(model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale)
        self.deform_based_generator = Generator()

        self.pretrained_hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66).cuda()
        pretrained_path = "/home/tiger/nfs/myenv/cache/useful_ckpts/hopenet_robust_alpha1.pkl" # https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR
        self.pretrained_hopenet.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        self.pretrained_hopenet.requires_grad_(False)

        self.pose_loss_fn = HeadPoseLoss() # 20
        self.equivariance_loss_fn = EquivarianceLoss() # 20
        self.keypoint_prior_loss_fn = KeypointPriorLoss()# 10
        self.deform_prior_loss_fn = DeformationPriorLoss() # 5

    def forward(self, torso_src_img, src_img, drv_img, cal_loss=False):
        # predict cano keypoint
        cano_keypoint = self.canonical_kp_detector(src_img)
        # predict src_pose and drv_pose
        transform_fn = Transform(drv_img.shape[0])
        transformed_drv_img = transform_fn.transform_frame(drv_img)
        cat_imgs = torch.cat([src_img, drv_img, transformed_drv_img], dim=0)
        yaw, pitch, roll, t, delta = self.pose_exp_estimator(cat_imgs)
        [yaw_s, yaw_d, yaw_tran], [pitch_s, pitch_d, pitch_tran], [roll_s, roll_d, roll_tran] = (
            torch.chunk(yaw, 3, dim=0),
            torch.chunk(pitch, 3, dim=0),
            torch.chunk(roll, 3, dim=0),
        )
        [t_s, t_d, t_tran], [delta_s, delta_d, delta_tran] = (
            torch.chunk(t, 3, dim=0),
            torch.chunk(delta, 3, dim=0),
        )
        kp_s, Rs = transform_kp(cano_keypoint, yaw_s, pitch_s, roll_s, t_s, delta_s)
        kp_d, Rd = transform_kp(cano_keypoint, yaw_d, pitch_d, roll_d, t_d, delta_d)
        # deform the torso img
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        deformation, occlusion = self.motion_field_estimator(torso_appearance_feats, kp_s, kp_d, Rs, Rd)
        deformed_torso_img = self.deform_based_generator(torso_appearance_feats, deformation, occlusion)
        
        ret = {'kp_src': kp_s, 'kp_drv': kp_d}
        if cal_loss:
            losses = {}
            with torch.no_grad():
                self.pretrained_hopenet.eval()
                real_yaw, real_pitch, real_roll = self.pretrained_hopenet(F.interpolate(apply_imagenet_normalization(cat_imgs), size=(224, 224)))
            pose_loss = self.pose_loss_fn(yaw, pitch, roll, real_yaw, real_pitch, real_roll)
            losses['facev2v/pose_pred_loss'] = pose_loss

            kp_tran, _ = transform_kp(cano_keypoint, yaw_tran, pitch_tran, roll_tran, t_tran, delta_tran)
            reverse_kp = transform_fn.warp_coordinates(kp_tran[:, :, :2])
            equivariance_loss = self.equivariance_loss_fn(kp_d, reverse_kp)
            losses['facev2v/equivariance_loss'] = equivariance_loss

            keypoint_prior_loss = self.keypoint_prior_loss_fn(kp_d)
            losses['facev2v/keypoint_prior_loss'] = keypoint_prior_loss

            deform_prior_loss = self.deform_prior_loss_fn(delta_d)
            losses['facev2v/deform_prior_loss'] = deform_prior_loss
            ret['losses'] = losses

        return deformed_torso_img, ret


class WarpBasedTorsoModelMediaPipe(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.appearance_extractor = AppearanceFeatureExtractor(model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=hparams['torso_kp_num']) # 32 channel appearance channel, and 3 channel for segmap
        # self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=9) # 32 channel appearance channel, and 3 channel for segmap
        self.deform_based_generator = Generator()

        self.occlusion_2_predictor = nn.Sequential(*[
            nn.Conv2d(64+1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        ])

    #  V2, 先warp， 再mean
    def forward(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, cal_loss=False, target_torso_mask=None):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        torso_appearance_feats = self.appearance_extractor(torso_src_img) # [B, C, D, H, W]
        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=hparams.get("torso_mask_dilate_ksize", 7))
        if hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)

        if hparams['torso_kp_num'] == 4:
            kp_s = kp_s[:,[0,8,16,27],:]
            kp_d = kp_d[:,[0,8,16,27],:]
        elif hparams['torso_kp_num'] == 9:
            kp_s = kp_s[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
            kp_d = kp_d[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
        else:
            raise NotImplementedError()

        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd)
        motion_estimator_grad_scale_factor = 0.1
        # motion_estimator_grad_scale_factor = 1.0
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion, a 0~1 mask that predict the segment map of warped torso, used in oclcusion-aware decoder
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion_2, a 0~1 mask that predict the segment map of warped torso, but is used in alpha-blending
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}

        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        occlusion_2 = self.occlusion_2_predictor(torch.cat([deformed_torso_hid, F.interpolate(occlusion_2, size=(256,256), mode='bilinear')], dim=1))
        ret['occlusion_2'] = occlusion_2
        alphas = occlusion_2.clamp(1e-5, 1 - 1e-5) 

        if target_torso_mask is None:
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': occlusion.mean(),
                'facev2v/occlusion_2_reg_l1': occlusion_2.mean(),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        else:
            non_target_torso_mask_1 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion.shape[-2:])
            non_target_torso_mask_2 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion_2.shape[-2:])
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': self.masked_l1_reg_loss(occlusion, non_target_torso_mask_1.bool(), masked_weight=1, unmasked_weight=hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_reg_l1': self.masked_l1_reg_loss(occlusion_2, non_target_torso_mask_2.bool(), masked_weight=1, unmasked_weight=hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        # if hparams.get("fuse_with_deform_source"):
        #     B, _, H, W = deformed_torso_img.shape
        #     deformation_256 = F.interpolate(deformation.mean(dim=1).permute(0,3,1,2), size=256, mode='bilinear',antialias=True).permute(0,2,3,1)[...,:2]
        #     deformed_source_torso_img = F.grid_sample(torso_src_img, deformation_256, align_corners=True).view(B, -1, H, W)
        #     occlusion_256 = F.interpolate(occlusion, size=256, antialias=True, mode='bilinear').reshape([B,1,H,W])
        #     # deformed_torso_img = deformed_torso_img * (1 - occlusion_256[:,0]) + deformed_source_torso_img[:,0] * occlusion_256[:,0]
        #     deformed_torso_img = deformed_torso_img * (1 - occlusion_256) + deformed_source_torso_img * occlusion_256
        return deformed_torso_img, ret

    def masked_l1_reg_loss(self, img_pred, mask, masked_weight=0.01, unmasked_weight=0.001, mode='l1'):
        # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred).abs().sum(dim=1) * weight_mask
        else:
            error = (img_pred).pow(2).sum(dim=1) * weight_mask
        loss = error.mean()
        return loss

    @torch.no_grad()
    def infer_forward_stage1(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, cal_loss=False):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        kp_s = kp_s[:,[0,8,16,27],:]
        kp_d = kp_d[:,[0,8,16,27],:]

        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=hparams.get("torso_mask_dilate_ksize", 7))
        if hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)
        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd)
        motion_estimator_grad_scale_factor = 0.1
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}
        ret['torso_appearance_feats'] = torso_appearance_feats
        ret['deformation'] = deformation
        ret['occlusion'] = occlusion
        return ret
    
    @torch.no_grad()
    def infer_forward_stage2(self, ret):
        torso_appearance_feats = ret['torso_appearance_feats']
        deformation = ret['deformation']
        occlusion = ret['occlusion']
        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        return deformed_torso_img
    
if __name__ == '__main__':
    from utils.nn.model_utils import num_params
    import tqdm
    model = WarpBasedTorsoModel('small')
    model.cuda()
    num_params(model)
    for n, m in model.named_children():
        num_params(m, model_name=n)
    torso_ref_img = torch.randn([2, 3, 256, 256]).cuda()
    ref_img = torch.randn([2, 3, 256, 256]).cuda()
    mv_img = torch.randn([2, 3, 256, 256]).cuda()
    out = model(torso_ref_img, ref_img, mv_img)
    for i in tqdm.trange(100):
        out_img, losses = model(torso_ref_img, ref_img, mv_img, cal_loss=True)
    print(" ")import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from modules.real3d.facev2v_warp.layers import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from modules.real3d.facev2v_warp.func_utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)

class AppearanceFeatureExtractor(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, in_dim=3, model_scale='standard', lora_args=None):
        super().__init__()
        use_weight_norm = False
        down_seq = [64, 128, 256]
        n_res = 6
        C = 32
        D = 16
        self.in_conv = ConvBlock2D("CNA", in_dim, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], C * D, 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock3D(C, use_weight_norm) for _ in range(n_res)])

        self.C, self.D = C, D

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.res(x)
        return x


class CanonicalKeypointDetector(nn.Module):
    # Canonical keypoints detector
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,512,16,16]
    # [N,1024,8,8]
    # [N,16384,8,8]
    # [N,1024,16,8,8]
    # [N,512,16,16,16]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False

        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [3, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
            D = 16 # depth_channel 
            K = 15
            scale_factor=0.25
        elif model_scale == 'small':
            down_seq = [3, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
            D = 6 # depth_channel 
            K = 15
            scale_factor=0.25
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        # [1, 3, 256, 256] ==> [1, 3, 64, 64]
        x = self.down(x) # ==> [1, 1024, 2, 2]
        x = self.mid_conv(x) # ==> [1, 16384, 2, 2]
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W) # ==> [1, 1024, 16, 2, 2]
        x = self.up(x) # ==> [1, 32, 16, 64, 64]
        x = self.out_conv(x) # ==> [1, 15, 16, 64, 64]
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class PoseExpressionEstimator(nn.Module):
    # Head pose estimator && expression deformation estimator
    # [N,3,256,256]
    # [N,64,64,64]
    # [N,256,64,64]
    # [N,512,32,32]
    # [N,1024,16,16]
    # [N,2048,8,8]
    # [N,2048]
    # [N,66] [N,66] [N,66] [N,3] [N,60]
    # [N,] [N,] [N,] [N,3] [N,20,3]
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False
        n_bins=66
        K=15
        if model_scale == 'standard' or model_scale == 'large':
            n_filters=[64, 256, 512, 1024, 2048]
            n_blocks=[3, 3, 5, 2]
        elif model_scale == 'small':
            n_filters=[32, 128, 256, 512, 512]
            n_blocks=[2, 2, 4, 2]

        self.pre_layers = nn.Sequential(ConvBlock2D("CNA", 3, n_filters[0], 7, 2, 3, use_weight_norm), nn.MaxPool2d(3, 2, 1))
        res_layers = []
        for i in range(len(n_filters) - 1):
            res_layers.extend(self._make_layer(i, n_filters[i], n_filters[i + 1], n_blocks[i], use_weight_norm))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc_yaw = nn.Linear(n_filters[-1], n_bins)
        self.fc_pitch = nn.Linear(n_filters[-1], n_bins)
        self.fc_roll = nn.Linear(n_filters[-1], n_bins)
        self.fc_t = nn.Linear(n_filters[-1], 3)
        self.fc_delta = nn.Linear(n_filters[-1], 3 * K)
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return [ResBottleneck(in_channels, out_channels, stride, use_weight_norm)] + [
            ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)
        ]

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw, pitch, roll, t, delta = self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x), self.fc_t(x), self.fc_delta(x)
        yaw = torch.softmax(yaw, dim=1)
        pitch = torch.softmax(pitch, dim=1)
        roll = torch.softmax(roll, dim=1)
        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta


class MotionFieldEstimator(nn.Module):
    # Motion field estimator
    # (4+1)x(20+1)=105
    # [N,105,16,64,64]
    # ...
    # [N,32,16,64,64]
    # [N,137,16,64,64]
    # 1.
    # [N,21,16,64,64] (mask)
    # 2.
    # [N,2192,64,64]
    # [N,1,64,64] (occlusion)
    def __init__(self, model_scale='standard', input_channels=32, num_keypoints=15, predict_multiref_occ=True):
        super().__init__()
        use_weight_norm=False
        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [(num_keypoints+1)*5, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
        elif model_scale == 'small':
            down_seq = [(num_keypoints+1)*5, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
        K = num_keypoints
        D = 16
        C1 = input_channels # appearance feats channel
        C2 = 4
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])

        tgt_head_in_dim = 3 + 1
        tgt_head_hid_dim = 32
        tgt_head_layers =  [ConvBlock2D("CNA", tgt_head_in_dim, tgt_head_hid_dim, 7, 1, 3, use_weight_norm)] + [ResBlock2D(tgt_head_hid_dim, use_weight_norm) for _ in range(3)]
        self.tgt_head_encoder = nn.Sequential(*tgt_head_layers)
        self.tgt_head_fuser = nn.Conv3d(tgt_head_hid_dim + down_seq[0] + up_seq[-1], tgt_head_hid_dim, 7, 1, 3)
        
        self.mask_conv = nn.Conv3d(tgt_head_hid_dim, K + 1, 7, 1, 3)
        self.predict_multiref_occ = predict_multiref_occ
        self.occlusion_conv = nn.Conv2d(tgt_head_hid_dim * D, 1, 7, 1, 3)
        self.occlusion_conv2 = nn.Conv2d(tgt_head_hid_dim * D, 1, 7, 1, 3)

        self.C, self.D = down_seq[0] + up_seq[-1], D
        
    def forward(self, fs, kp_s, kp_d, Rs, Rd, tgt_head_img, tgt_head_weights):
        # the original fs is compressed to 4 channels using a 1x1x1 conv
        fs_compressed = self.compress(fs)
        N, _, D, H, W = fs.shape
        # [N,21,1,16,64,64]
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d)
        # [N,21,16,64,64,3]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)
        # [N,21,4,16,64,64]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2).view(N, -1, D, H, W)
        output = self.down(input)
        output = self.up(output)
        x = torch.cat([input, output], dim=1)

        tgt_head_inp = torch.cat([tgt_head_img, tgt_head_weights], dim=1)
        tgt_head_inp = torch.nn.functional.interpolate(tgt_head_inp, size=(128,128), mode='bilinear')
        tgt_head_feats = self.tgt_head_encoder(tgt_head_inp) # [B, C=3+1, H=256, W=256]
        tgt_head_feats = torch.nn.functional.interpolate(tgt_head_feats, size=(64,64), mode='bilinear')

        fused_x = torch.cat([x, tgt_head_feats.unsqueeze(2).repeat([1,1,x.shape[2],1,1])], dim=1)
        x = self.tgt_head_fuser(fused_x)

        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        if self.predict_multiref_occ:
            occlusion, occlusion_2 = self.create_occlusion(x.view(N, -1, H, W))
            return deformation, occlusion, occlusion_2
        else:
            return deformation, x.view(N, -1, H, W)
        
    # x: torch.Tensor, N, M, H, W
    def create_occlusion(self, x, deformed_source=None):
        occlusion = self.occlusion_conv(x)
        occlusion_2 = self.occlusion_conv2(x)
        occlusion = torch.sigmoid(occlusion)
        occlusion_2 = torch.sigmoid(occlusion_2)
        return occlusion, occlusion_2
    


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, input_channels=32, model_scale='standard', more_res=False):
        super().__init__()
        use_weight_norm=True
        C=input_channels
        
        if model_scale == 'large':
            n_res = 12
            up_seq = [256, 128, 64]
            D = 16
            use_up_res = True
        elif model_scale in ['standard', 'small']:
            n_res = 6
            up_seq = [256, 128, 64]
            D = 16 
            use_up_res = False
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        ups = []
        for i in range(len(up_seq) - 1):
            ups.append(UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm))
            if use_up_res:
                ups.append(ResBlock2D(up_seq[i + 1], up_seq[i + 1]))
        self.up = nn.Sequential(*ups)
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)
               
    def forward(self, fs, deformation, occlusion, return_hid=False):
        deformed_fs = self.get_deformed_feature(fs, deformation)
        return self.forward_with_deformed_feature(deformed_fs, occlusion, return_hid=return_hid)
    
    def forward_with_deformed_feature(self, deformed_fs, occlusion, return_hid=False):
        fs = deformed_fs
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        fs = self.res(fs)
        fs = self.up(fs)
        rgb = self.out_conv(fs)
        if return_hid:
            return rgb, fs
        return rgb
    
    @staticmethod
    def get_deformed_feature(fs, deformation):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True, padding_mode='border').view(N, -1, H, W)
        return fs


class Discriminator(nn.Module):
    # Patch Discriminator

    def __init__(self, use_weight_norm=True, down_seq=[64, 128, 256, 512], K=15):
        super().__init__()
        layers = []
        layers.append(ConvBlock2D("CNA", 3 + K, down_seq[0], 3, 2, 1, use_weight_norm, "instance", "leakyrelu"))
        layers.extend(
            [
                ConvBlock2D("CNA", down_seq[i], down_seq[i + 1], 3, 2 if i < len(down_seq) - 2 else 1, 1, use_weight_norm, "instance", "leakyrelu")
                for i in range(len(down_seq) - 1)
            ]
        )
        layers.append(ConvBlock2D("CN", down_seq[-1], 1, 3, 1, 1, use_weight_norm, activation_type="none"))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, kp):
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])
        x = torch.cat([x, heatmap], dim=1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from utils.commons.hparams import hparams

from modules.real3d.facev2v_warp.layers import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from modules.real3d.facev2v_warp.func_utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)

class AppearanceFeatureExtractor(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [64, 128, 256]
        n_res = 6
        C = 32
        D = 16
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], C * D, 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock3D(C, use_weight_norm) for _ in range(n_res)])

        self.C, self.D = C, D

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.res(x)
        return x


class CanonicalKeypointDetector(nn.Module):
    # Canonical keypoints detector
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,512,16,16]
    # [N,1024,8,8]
    # [N,16384,8,8]
    # [N,1024,16,8,8]
    # [N,512,16,16,16]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False

        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [3, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
            D = 16 # depth_channel 
            K = 15
            scale_factor=0.25
        elif model_scale == 'small':
            down_seq = [3, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
            D = 6 # depth_channel 
            K = 15
            scale_factor=0.25
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        # [1, 3, 256, 256] ==> [1, 3, 64, 64]
        x = self.down(x) # ==> [1, 1024, 2, 2]
        x = self.mid_conv(x) # ==> [1, 16384, 2, 2]
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W) # ==> [1, 1024, 16, 2, 2]
        x = self.up(x) # ==> [1, 32, 16, 64, 64]
        x = self.out_conv(x) # ==> [1, 15, 16, 64, 64]
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class PoseExpressionEstimator(nn.Module):
    # Head pose estimator && expression deformation estimator
    # [N,3,256,256]
    # [N,64,64,64]
    # [N,256,64,64]
    # [N,512,32,32]
    # [N,1024,16,16]
    # [N,2048,8,8]
    # [N,2048]
    # [N,66] [N,66] [N,66] [N,3] [N,60]
    # [N,] [N,] [N,] [N,3] [N,20,3]
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False
        n_bins=66
        K=15
        if model_scale == 'standard' or model_scale == 'large':
            n_filters=[64, 256, 512, 1024, 2048]
            n_blocks=[3, 3, 5, 2]
        elif model_scale == 'small':
            n_filters=[32, 128, 256, 512, 512]
            n_blocks=[2, 2, 4, 2]

        self.pre_layers = nn.Sequential(ConvBlock2D("CNA", 3, n_filters[0], 7, 2, 3, use_weight_norm), nn.MaxPool2d(3, 2, 1))
        res_layers = []
        for i in range(len(n_filters) - 1):
            res_layers.extend(self._make_layer(i, n_filters[i], n_filters[i + 1], n_blocks[i], use_weight_norm))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc_yaw = nn.Linear(n_filters[-1], n_bins)
        self.fc_pitch = nn.Linear(n_filters[-1], n_bins)
        self.fc_roll = nn.Linear(n_filters[-1], n_bins)
        self.fc_t = nn.Linear(n_filters[-1], 3)
        self.fc_delta = nn.Linear(n_filters[-1], 3 * K)
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return [ResBottleneck(in_channels, out_channels, stride, use_weight_norm)] + [
            ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)
        ]

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw, pitch, roll, t, delta = self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x), self.fc_t(x), self.fc_delta(x)
        yaw = torch.softmax(yaw, dim=1)
        pitch = torch.softmax(pitch, dim=1)
        roll = torch.softmax(roll, dim=1)
        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta


class MotionFieldEstimator(nn.Module):
    # Motion field estimator
    # (4+1)x(20+1)=105
    # [N,105,16,64,64]
    # ...
    # [N,32,16,64,64]
    # [N,137,16,64,64]
    # 1.
    # [N,21,16,64,64] (mask)
    # 2.
    # [N,2192,64,64]
    # [N,1,64,64] (occlusion)
    def __init__(self, model_scale='standard', input_channels=32, num_keypoints=15, predict_multiref_occ=True, occ2_on_deformed_source=False):
        super().__init__()
        use_weight_norm=False
        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [(num_keypoints+1)*5, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
        elif model_scale == 'small':
            down_seq = [(num_keypoints+1)*5, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
        K = num_keypoints
        D = 16
        C1 = input_channels # appearance feats channel
        C2 = 4
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.mask_conv = nn.Conv3d(down_seq[0] + up_seq[-1], K + 1, 7, 1, 3)
        self.predict_multiref_occ = predict_multiref_occ
        self.occ2_on_deformed_source = occ2_on_deformed_source
        self.occlusion_conv = nn.Conv2d((down_seq[0] + up_seq[-1]) * D, 1, 7, 1, 3)
        if self.occ2_on_deformed_source:
            self.occlusion_conv2 = nn.Conv2d(3, 1, 7, 1, 3)
        else:
            self.occlusion_conv2 = nn.Conv2d((down_seq[0] + up_seq[-1]) * D, 1, 7, 1, 3)
        self.C, self.D = down_seq[0] + up_seq[-1], D

    def forward(self, fs, kp_s, kp_d, Rs, Rd):
        # the original fs is compressed to 4 channels using a 1x1x1 conv
        fs_compressed = self.compress(fs)
        N, _, D, H, W = fs.shape
        # [N,21,1,16,64,64]
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d)
        # [N,21,16,64,64,3]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)
        # [N,21,4,16,64,64]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2).view(N, -1, D, H, W)
        output = self.down(input)
        output = self.up(output)
        x = torch.cat([input, output], dim=1) # [B, C1=25 + C2=32, D, H, W]
        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        if self.predict_multiref_occ:
            occlusion, occlusion_2 = self.create_occlusion(x.view(N, -1, H, W))
            return deformation, occlusion, occlusion_2
        else:
            return deformation, x.view(N, -1, H, W)
        
    # x: torch.Tensor, N, M, H, W
    def create_occlusion(self, x, deformed_source=None):
        occlusion = self.occlusion_conv(x)
        if self.occ2_on_deformed_source:
            assert deformed_source is not None
            occlusion_2 = self.occlusion_conv2(deformed_source)
        else:
            occlusion_2 = self.occlusion_conv2(x)
        occlusion = torch.sigmoid(occlusion)
        occlusion_2 = torch.sigmoid(occlusion_2)
        return occlusion, occlusion_2
    


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, input_channels=32, model_scale='standard', more_res=False):
        super().__init__()
        use_weight_norm=True
        C=input_channels
        
        if model_scale == 'large':
            n_res = 12
            up_seq = [256, 128, 64]
            D = 16
            use_up_res = True
        elif model_scale in ['standard', 'small']:
            n_res = 6
            up_seq = [256, 128, 64]
            D = 16 
            use_up_res = False
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        ups = []
        for i in range(len(up_seq) - 1):
            ups.append(UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm))
            if use_up_res:
                ups.append(ResBlock2D(up_seq[i + 1], up_seq[i + 1]))
        self.up = nn.Sequential(*ups)
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)
               
    def forward(self, fs, deformation, occlusion, return_hid=False):
        deformed_fs = self.get_deformed_feature(fs, deformation)
        return self.forward_with_deformed_feature(deformed_fs, occlusion, return_hid=return_hid)
    
    def forward_with_deformed_feature(self, deformed_fs, occlusion, return_hid=False):
        fs = deformed_fs
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        # if hparams.get("occlusion_fuse", True):
        #     blank = torch.full_like(fs, 0.)
        #     fs = fs * occlusion + blank * (1 - occlusion)
        # else:
        #     pass
        fs = self.res(fs)
        fs = self.up(fs)
        rgb = self.out_conv(fs)
        if return_hid:
            return rgb, fs
        return rgb
    
    @staticmethod
    def get_deformed_feature(fs, deformation):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True, padding_mode='border').view(N, -1, H, W)
        return fs


class Discriminator(nn.Module):
    # Patch Discriminator

    def __init__(self, use_weight_norm=True, down_seq=[64, 128, 256, 512], K=15):
        super().__init__()
        layers = []
        layers.append(ConvBlock2D("CNA", 3 + K, down_seq[0], 3, 2, 1, use_weight_norm, "instance", "leakyrelu"))
        layers.extend(
            [
                ConvBlock2D("CNA", down_seq[i], down_seq[i + 1], 3, 2 if i < len(down_seq) - 2 else 1, 1, use_weight_norm, "instance", "leakyrelu")
                for i in range(len(down_seq) - 1)
            ]
        )
        layers.append(ConvBlock2D("CN", down_seq[-1], 1, 3, 1, 1, use_weight_norm, activation_type="none"))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, kp):
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])
        x = torch.cat([x, heatmap], dim=1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
