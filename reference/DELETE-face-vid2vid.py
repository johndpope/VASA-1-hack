"""
Code from https://github.com/hassony2/torch_videovision
"""

import random
import numpy as np
import PIL
import torchvision

import warnings

from skimage import img_as_ubyte, img_as_float


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]
        if random.random() < 0.5 and self.horizontal_flip:
            return [np.fliplr(img) for img in clip]

        return clip


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype("float32"))
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))
        return jittered_clip


class AllAugmentationTransform:
    def __init__(self, flip_param=None, jitter_param=None):
        self.transforms = []

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob


def read_video(name, frame_shape):
    """
    Read video which can be:
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, str(frames[idx], encoding="utf-8")))) for idx in range(num_frames)])
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4"):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(
        self,
        root_dir="datasets/vox",
        frame_shape=(256, 256, 3),
        id_sampling=True,
        is_train=True,
        random_seed=0,
        pairs_list=None,
        augmentation_params={
            "flip_param": {"horizontal_flip": True, "time_flip": True},
            "jitter_param": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.1},
        },
    ):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, "train")):
            assert os.path.exists(os.path.join(root_dir, "test"))
            if id_sampling:
                train_videos = {os.path.basename(video).split("#")[0] for video in os.listdir(os.path.join(root_dir, "train"))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, "train"))
            test_videos = os.listdir(os.path.join(root_dir, "test"))
            self.root_dir = os.path.join(self.root_dir, "train" if is_train else "test")
        else:
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + "*.mp4")))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, str(frames[idx], encoding="utf-8")))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            driving = driving.transpose((2, 0, 1))
            source = source.transpose((2, 0, 1))
            return source, driving
        else:
            video = np.array(video_array, dtype="float32")
            video = video.transpose((3, 0, 1, 2))
            return video


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=75):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs["source"].isin(videos), pairs["driving"].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[pairs["driving"].iloc[ind]], name_to_index[pairs["source"].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {"driving_" + key: value for key, value in first.items()}
        second = {"source_" + key: value for key, value in second.items()}

        return {**first, **second}
import functools
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def init_seeds(cuda_deterministic=True):
    seed = 1 + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def init_dist(local_rank, world_size, backend="nccl"):
    r"""Initialize distributed training"""
    torch.autograd.set_detect_anomaly(True)
    if dist.is_available():
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=local_rank)
    print("Rank", get_rank(), "initialized.")


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def master_only(func):
    r"""Apply this function only to the master GPU."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args, **kwargs):
    r"""master-only print"""
    print(*args, **kwargs)
import argparse
from models import AFE, CKD, HPE_EDE, MFE, Generator
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
from skimage import io, img_as_float32
from utils import transform_kp, transform_kp_with_new_pose


@torch.no_grad()
def eval(args):
    g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    ckp_path = os.path.join(args.ckp_dir, "%s-checkpoint.pth.tar" % str(args.ckp).zfill(8))
    checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    output_frames = []
    if args.source == "r":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        s = np.array(video_array[0], dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw_s, pitch_s, roll_s, t_s, delta_s = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw_s, pitch_s, roll_s, t_s, delta_s)
        for img in video_array[1:]:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    elif args.source == "f":
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            fs = g_models["afe"](img)
            kp_c = g_models["ckd"](img)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            kp_d, Rd = transform_kp_with_new_pose(kp_c, yaw, pitch, roll, t, delta, 0 * yaw, 0 * pitch, 0 * roll)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            # generated_d = F.interpolate(generated_d, scale_factor=0.5)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    else:
        s = img_as_float32(io.imread(args.source))[:, :, :3]
        s = np.array(s, dtype="float32").transpose((2, 0, 1))
        s = torch.from_numpy(s).cuda().unsqueeze(0)
        s = F.interpolate(s, size=(256, 256))
        fs = g_models["afe"](s)
        kp_c = g_models["ckd"](s)
        yaw, pitch, roll, t, delta = g_models["hpe_ede"](s)
        kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)
        frames = sorted(os.listdir(args.driving))[: args.num_frames]
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(args.driving, frames[idx]))) for idx in range(num_frames)]
        for img in video_array:
            img = np.array(img, dtype="float32").transpose((2, 0, 1))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
            kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
            deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
            generated_d = g_models["generator"](fs, deformation, occlusion)
            generated_d = torch.cat((img, generated_d), dim=3)
            generated_d = generated_d.squeeze(0).data.cpu().numpy()
            generated_d = np.transpose(generated_d, [1, 2, 0])
            generated_d = generated_d.clip(0, 1)
            generated_d = (255 * generated_d).astype(np.uint8)
            output_frames.append(generated_d)
    imageio.mimsave(args.output, output_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="ckp", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="output.gif", help="Output video")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")
    parser.add_argument("--source", type=str, default="r", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, help="Driving dir")
    parser.add_argument("--num_frames", type=int, default=90, help="Number of frames")

    args = parser.parse_args()
    eval(args)
import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from distributed import master_only, master_only_print, get_rank, is_master
from models import AFE, CKD, HPE_EDE, MFE, Generator, Discriminator
from trainer import GeneratorFull, DiscriminatorFull
from tqdm import tqdm


def to_cpu(losses):
    return {key: value.detach().data.cpu().numpy() for key, value in losses.items()}


class Logger:
    def __init__(
        self,
        ckp_dir,
        vis_dir,
        dataloader,
        lr,
        checkpoint_freq=1,
        visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow"},
        zfill_num=8,
        log_file_name="log.txt",
    ):

        self.g_losses, self.d_losses = [], []
        self.ckp_dir = ckp_dir
        self.vis_dir = vis_dir
        if is_master():
            if not os.path.exists(self.ckp_dir):
                os.makedirs(self.ckp_dir)
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            self.log_file = open(log_file_name, "a")
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float("inf")
        self.g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
        self.d_models = {"discriminator": Discriminator()}
        for name, model in self.g_models.items():
            self.g_models[name] = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[get_rank()])
        for name, model in self.d_models.items():
            self.d_models[name] = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[get_rank()])
        self.g_optimizers = {name: torch.optim.Adam(self.g_models[name].parameters(), lr=lr, betas=(0.5, 0.999)) for name in self.g_models.keys()}
        self.d_optimizers = {name: torch.optim.Adam(self.d_models[name].parameters(), lr=lr, betas=(0.5, 0.999)) for name in self.d_models.keys()}
        self.g_full = GeneratorFull(**self.g_models, **self.d_models)
        self.d_full = DiscriminatorFull(**self.d_models)
        self.g_loss_names, self.d_loss_names = None, None
        self.dataloader = dataloader

    def __del__(self):
        self.save_cpk()
        if is_master():
            self.log_file.close()

    @master_only
    def log_scores(self):
        loss_mean = np.array(self.g_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.g_loss_names, loss_mean)])
        loss_string = "G" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.g_losses = []
        loss_mean = np.array(self.d_losses).mean(axis=0)
        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.d_loss_names, loss_mean)])
        loss_string = "D" + str(self.epoch).zfill(self.zfill_num) + ") " + loss_string
        print(loss_string, file=self.log_file)
        self.d_losses = []
        self.log_file.flush()

    @master_only
    def visualize_rec(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        image = self.visualizer.visualize(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)
        imageio.imsave(os.path.join(self.vis_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    @master_only
    def save_cpk(self):
        ckp = {
            **{k: v.module.state_dict() for k, v in self.g_models.items()},
            **{k: v.module.state_dict() for k, v in self.d_models.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.g_optimizers.items()},
            **{"optimizer_" + k: v.state_dict() for k, v in self.d_optimizers.items()},
            "epoch": self.epoch,
        }
        ckp_path = os.path.join(self.ckp_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        torch.save(ckp, ckp_path)

    def load_cpk(self, epoch):
        ckp_path = os.path.join(self.ckp_dir, "%s-checkpoint.pth.tar" % str(epoch).zfill(self.zfill_num))
        checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
        for k, v in self.g_models.items():
            v.module.load_state_dict(checkpoint[k])
        for k, v in self.d_models.items():
            v.module.load_state_dict(checkpoint[k])
        for k, v in self.g_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        for k, v in self.d_optimizers.items():
            v.load_state_dict(checkpoint["optimizer_" + k])
        self.epoch = checkpoint["epoch"] + 1

    @master_only
    def log_iter(self, g_losses, d_losses):
        g_losses = collections.OrderedDict(g_losses.items())
        d_losses = collections.OrderedDict(d_losses.items())
        if self.g_loss_names is None:
            self.g_loss_names = list(g_losses.keys())
        if self.d_loss_names is None:
            self.d_loss_names = list(d_losses.keys())
        self.g_losses.append(list(g_losses.values()))
        self.d_losses.append(list(d_losses.values()))

    @master_only
    def log_epoch(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores()
        self.visualize_rec(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)

    def step(self):
        master_only_print("Epoch", self.epoch)
        with tqdm(total=len(self.dataloader.dataset)) as progress_bar:
            for s, d in self.dataloader:
                s = s.cuda(non_blocking=True)
                d = d.cuda(non_blocking=True)
                for optimizer in self.g_optimizers.values():
                    optimizer.zero_grad()
                losses_g, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion = self.g_full(s, d)
                loss_g = sum(losses_g.values())
                loss_g.backward()
                for optimizer in self.g_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                for optimizer in self.d_optimizers.values():
                    optimizer.zero_grad()
                losses_d = self.d_full(d, generated_d, kp_d)
                loss_d = sum(losses_d.values())
                loss_d.backward()
                for optimizer in self.d_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                self.log_iter(to_cpu(losses_g), to_cpu(losses_d))
                if is_master():
                    progress_bar.update(len(s))
        self.log_epoch(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)
        self.epoch += 1


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
        images = []
        # Source image with keypoints
        source = s.data.cpu()
        kp_source = kp_s.data.cpu().numpy()[:, :, :2]
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        transformed = transformed_d.data.cpu().numpy()
        transformed = np.transpose(transformed, [0, 2, 3, 1])
        transformed_kp = transformed_kp.data.cpu().numpy()[:, :, :2]
        images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = kp_d.data.cpu().numpy()[:, :, :2]
        driving = d.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result with and without keypoints
        prediction = generated_d.data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Occlusion map
        occlusion_map = occlusion.data.cpu().repeat(1, 3, 1, 1)
        occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
        occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
        images.append(occlusion_map)

        image = self.create_image_grid(*images)
        image = image.clip(0, 1)
        image = (255 * image).astype(np.uint8)
        return image
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import nn
from utils import apply_imagenet_normalization, apply_vggface_normalization


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
    def __init__(self, layers_weight={"relu_1_1": 0.03125, "relu_2_1": 0.0625, "relu_3_1": 0.125, "relu_4_1": 0.25, "relu_5_1": 1.0}, n_scale=3):
        super().__init__()
        self.vgg19 = _vgg19(layers_weight.keys())
        self.vggface = _vgg_face(layers_weight.keys())
        self.criterion = nn.L1Loss()
        self.layers_weight, self.n_scale = layers_weight, n_scale

    def forward(self, input, target):
        self.vgg19.eval()
        self.vggface.eval()
        loss = 0
        loss += self.criterion(input, target)
        features_vggface_input = self.vggface(apply_vggface_normalization(input))
        features_vggface_target = self.vggface(apply_vggface_normalization(target))
        input = apply_imagenet_normalization(input)
        target = apply_imagenet_normalization(target)
        features_vgg19_input = self.vgg19(input)
        features_vgg19_target = self.vgg19(target)
        for layer, weight in self.layers_weight.items():
            loss += weight * self.criterion(features_vggface_input[layer], features_vggface_target[layer].detach()) / 255
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
        for i in range(self.n_scale):
            input = F.interpolate(input, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            target = F.interpolate(target, mode="bilinear", scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
            features_vgg19_input = self.vgg19(input)
            features_vgg19_target = self.vgg19(target)
            loss += weight * self.criterion(features_vgg19_input[layer], features_vgg19_target[layer].detach())
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
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from modules import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)


class AFE(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, use_weight_norm=False, down_seq=[64, 128, 256], n_res=6, C=32, D=16):
        super().__init__()
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


class CKD(nn.Module):
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
    def __init__(
        self, use_weight_norm=False, down_seq=[3, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], D=16, K=15, scale_factor=0.25
    ):
        super().__init__()
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        # self.out_conv = nn.Conv3d(up_seq[-1], K, 7, 1, 3)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.up(x)
        x = self.out_conv(x)
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class HPE_EDE(nn.Module):
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
    def __init__(self, use_weight_norm=False, n_filters=[64, 256, 512, 1024, 2048], n_blocks=[3, 3, 5, 2], n_bins=66, K=15):
        super().__init__()
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


class MFE(nn.Module):
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
    def __init__(self, use_weight_norm=False, down_seq=[80, 64, 128, 256, 512, 1024], up_seq=[1024, 512, 256, 128, 64, 32], K=15, D=16, C1=32, C2=4):
        super().__init__()
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.mask_conv = nn.Conv3d(down_seq[0] + up_seq[-1], K + 1, 7, 1, 3)
        self.occlusion_conv = nn.Conv2d((down_seq[0] + up_seq[-1]) * D, 1, 7, 1, 3)
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
        x = torch.cat([input, output], dim=1)
        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        occlusion = self.occlusion_conv(x.view(N, -1, H, W))
        occlusion = torch.sigmoid(occlusion)
        return deformation, occlusion


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, use_weight_norm=True, n_res=6, up_seq=[256, 128, 64], D=16, C=32):
        super().__init__()
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        self.up = nn.Sequential(*[UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)

    def forward(self, fs, deformation, occlusion):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True).view(N, -1, H, W)
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        fs = fs * occlusion
        fs = self.res(fs)
        fs = self.up(fs)
        fs = self.out_conv(fs)
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
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu"
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 2, activation_type, nonlinearity_type)


class ConvBlock3D(_ConvBlock):
    def __init__(
        self, pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, activation_type="batch", nonlinearity_type="relu"
    ):
        super().__init__(pattern, in_channels, out_channels, kernel_size, stride, padding, use_weight_norm, 3, activation_type, nonlinearity_type)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight_norm, base_conv, base_pooling, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(base_conv("CNA", in_channels, out_channels, 3, 1, 1, use_weight_norm), base_pooling(kernel_size))

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
import math
import torchvision
import numpy as np
import torch.nn.functional as F
from torch import nn
from models import AFE, CKD, HPE_EDE, MFE, Generator, Discriminator
from losses import PerceptualLoss, GANLoss, FeatureMatchingLoss, EquivarianceLoss, KeypointPriorLoss, HeadPoseLoss, DeformationPriorLoss
from utils import transform_kp, make_coordinate_grid_2d, apply_imagenet_normalization


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


class GeneratorFull(nn.Module):
    def __init__(
        self,
        afe: AFE,
        ckd: CKD,
        hpe_ede: HPE_EDE,
        mfe: MFE,
        generator: Generator,
        discriminator: Discriminator,
        pretrained_path="hopenet_robust_alpha1.pkl",
        n_bins=66,
    ):
        super().__init__()
        pretrained = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], n_bins).cuda()
        pretrained.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        for parameter in pretrained.parameters():
            parameter.requires_grad = False
        self.pretrained = pretrained
        self.afe = afe
        self.ckd = ckd
        self.hpe_ede = hpe_ede
        self.mfe = mfe
        self.generator = generator
        self.discriminator = discriminator
        self.weights = {
            "P": 10,
            "G": 1,
            "F": 10,
            "E": 20,
            "L": 10,
            "H": 20,
            "D": 5,
        }
        self.losses = {
            "P": PerceptualLoss(),
            "G": GANLoss(),
            "F": FeatureMatchingLoss(),
            "E": EquivarianceLoss(),
            "L": KeypointPriorLoss(),
            "H": HeadPoseLoss(),
            "D": DeformationPriorLoss(),
        }

    def forward(self, s, d):
        fs = self.afe(s)
        kp_c = self.ckd(s)
        transform = Transform(d.shape[0])
        transformed_d = transform.transform_frame(d)
        cated = torch.cat([s, d, transformed_d], dim=0)
        yaw, pitch, roll, t, delta = self.hpe_ede(cated)
        [t_s, t_d, t_tran], [delta_s, delta_d, delta_tran] = (
            torch.chunk(t, 3, dim=0),
            torch.chunk(delta, 3, dim=0),
        )
        with torch.no_grad():
            self.pretrained.eval()
            real_yaw, real_pitch, real_roll = self.pretrained(F.interpolate(apply_imagenet_normalization(cated), size=(224, 224)))
        [yaw_s, yaw_d, yaw_tran], [pitch_s, pitch_d, pitch_tran], [roll_s, roll_d, roll_tran] = (
            torch.chunk(yaw, 3, dim=0),
            torch.chunk(pitch, 3, dim=0),
            torch.chunk(roll, 3, dim=0),
        )
        kp_s, Rs = transform_kp(kp_c, yaw_s, pitch_s, roll_s, t_s, delta_s)
        kp_d, Rd = transform_kp(kp_c, yaw_d, pitch_d, roll_d, t_d, delta_d)
        transformed_kp, _ = transform_kp(kp_c, yaw_tran, pitch_tran, roll_tran, t_tran, delta_tran)
        reverse_kp = transform.warp_coordinates(transformed_kp[:, :, :2])
        deformation, occlusion = self.mfe(fs, kp_s, kp_d, Rs, Rd)
        generated_d = self.generator(fs, deformation, occlusion)
        output_d, features_d = self.discriminator(d, kp_d)
        output_gd, features_gd = self.discriminator(generated_d, kp_d)
        loss = {
            "P": self.weights["P"] * self.losses["P"](generated_d, d),
            "G": self.weights["G"] * self.losses["G"](output_gd, True, False),
            "F": self.weights["F"] * self.losses["F"](features_gd, features_d),
            "E": self.weights["E"] * self.losses["E"](kp_d, reverse_kp),
            "L": self.weights["L"] * self.losses["L"](kp_d),
            "H": self.weights["H"] * self.losses["H"](yaw, pitch, roll, real_yaw, real_pitch, real_roll),
            "D": self.weights["D"] * self.losses["D"](delta_d),
        }
        return loss, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion


class DiscriminatorFull(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.weights = {
            "G": 1,
        }
        self.losses = {
            "G": GANLoss(),
        }

    def forward(self, d, generated_d, kp_d):
        output_d, _ = self.discriminator(d, kp_d)
        output_gd, _ = self.discriminator(generated_d.detach(), kp_d)
        loss = {
            "G1": self.weights["G"] * self.losses["G"](output_gd, False, True),
            "G2": self.weights["G"] * self.losses["G"](output_d, True, True),
        }
        return loss
import os
import argparse
import torch.utils.data as data
import torch.multiprocessing as mp
from logger import Logger
from dataset import FramesDataset, DatasetRepeater
from distributed import init_seeds, init_dist


def main(proc, args):
    world_size = len(args.gpu_ids)
    init_seeds(not args.benchmark)
    init_dist(proc, world_size)
    trainset = DatasetRepeater(FramesDataset(), num_repeats=100)
    trainsampler = data.distributed.DistributedSampler(trainset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=trainsampler)
    logger = Logger(args.ckp_dir, args.vis_dir, trainloader, args.lr)
    if args.ckp > 0:
        logger.load_cpk(args.ckp)
    for i in range(args.num_epochs):
        logger.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU")
    parser.add_argument("--benchmark", type=str2bool, default=True, help="Turn on CUDNN benchmarking")
    parser.add_argument("--gpu_ids", default=[0], type=eval, help="IDs of GPUs to use")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loader threads")
    parser.add_argument("--ckp_dir", type=str, default="ckp", help="Checkpoint dir")
    parser.add_argument("--vis_dir", type=str, default="vis", help="Visualization dir")
    parser.add_argument("--ckp", type=int, default=0, help="Checkpoint epoch")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    mp.spawn(main, nprocs=len(args.gpu_ids), args=(args,))
import torch
import torch.nn.functional as F


def rotation_matrix_x(theta):
    theta = theta.view(-1, 1, 1)
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
    theta = theta.view(-1, 1, 1)
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
    theta = theta.view(-1, 1, 1)
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
    xx = x.view(-1, 1).repeat(1, w)
    yy = y.view(1, -1).repeat(h, 1)
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
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    xx = x.view(1, -1, 1).repeat(d, 1, w)
    yy = y.view(1, 1, -1).repeat(d, h, 1)
    meshed = torch.cat([yy.unsqueeze(3), xx.unsqueeze(3), zz.unsqueeze(3)], 3)
    return meshed


def out2heatmap(out, temperature=0.1):
    final_shape = out.shape
    heatmap = out.view(final_shape[0], final_shape[1], -1)
    heatmap = F.softmax(heatmap / temperature, dim=2)
    heatmap = heatmap.view(*final_shape)
    return heatmap


def heatmap2kp(heatmap):
    shape = heatmap.shape
    grid = make_coordinate_grid_3d(shape[2:]).unsqueeze(0).unsqueeze(0)
    kp = (heatmap.unsqueeze(-1) * grid).sum(dim=(2, 3, 4))
    return kp


def kp2gaussian_2d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_2d(spatial_size).view(1, 1, *spatial_size, 2).repeat(N, K, 1, 1, 1)
    mean = kp.view(N, K, 1, 1, 2)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def kp2gaussian_3d(kp, spatial_size, kp_variance=0.01):
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_3d(spatial_size).view(1, 1, *spatial_size, 3).repeat(N, K, 1, 1, 1, 1)
    mean = kp.view(N, K, 1, 1, 1, 3)
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
    identity_grid = make_coordinate_grid_3d((D, H, W)).view(1, 1, D, H, W, 3).repeat(N, 1, 1, 1, 1, 1)
    # [N,20,16,64,64,3]
    coordinate_grid = identity_grid.repeat(1, K, 1, 1, 1, 1) - kp_d.view(N, K, 1, 1, 1, 3)
    # [N,1,1,1,1,3,3]
    jacobian = torch.matmul(Rs, torch.inverse(Rd)).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
    coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1)).squeeze(-1)
    driving_to_source = coordinate_grid + kp_s.view(N, K, 1, 1, 1, 3)
    sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
    # sparse_motions = driving_to_source
    # [N,21,16,64,64,3]
    return sparse_motions


def create_deformed_source_image(fs, sparse_motions):
    N, _, D, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    # [N*21,4,16,64,64]
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1, 1).view(N * (K + 1), -1, D, H, W)
    # [N*21,16,64,64,3]
    sparse_motions = sparse_motions.view((N * (K + 1), D, H, W, -1))
    # [N*21,4,16,64,64]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = sparse_deformed.view((N, K + 1, -1, D, H, W))
    # [N,21,4,16,64,64]
    return sparse_deformed


def apply_imagenet_normalization(input):
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output


def apply_vggface_normalization(input):
    mean = input.new_tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).view(1, 3, 1, 1)
    std = input.new_tensor([1, 1, 1]).view(1, 3, 1, 1)
    output = (input * 255 - mean) / std
    return output
