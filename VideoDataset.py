from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import gc
import lmdb
import pickle
import os
from tqdm import tqdm
from rembg import remove
import io
from skimage.transform import PiecewiseAffineTransform, warp
from torchvision.transforms.functional import to_pil_image, to_tensor
import random
class VideoDataset(Dataset):
    def __init__(self, 
                video_dir: str,
                width: int,
                height: int,
                initial_pairs: int = 1000,  # Start with fewer pairs
                max_pairs: int = 10000,     # Maximum pairs to reach
                growth_rate: float = 1.5,   # How fast to grow
                cache_dir: str = None,
                transform: transforms.Compose = None,
                remove_background: bool = False,
                use_greenscreen: bool = False,
                apply_warping: bool = False,
                max_frames: int = 100,
                duplicate_short: bool = True,
                warp_strength: float = 0.01):
        
        self.video_dir = Path(video_dir)
        self.width = width
        self.height = height
        self.initial_pairs = initial_pairs
        self.max_pairs = max_pairs
        self.growth_rate = growth_rate
        self.current_pairs = initial_pairs
        self.epoch = 0
        
        self.transform = transform
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.apply_warping = apply_warping
        self.max_frames = max_frames
        self.duplicate_short = duplicate_short
        self.warp_strength = warp_strength
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path("frame_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Get list of video files
        self.video_files = list(self.video_dir.glob("*.mp4"))
        print(f"Found {len(self.video_files)} video files")
        
        # Process videos in batches as needed
        self.processed_videos = set()
        self._process_initial_videos()
        
        # Create initial pairs
        self.pairs = self._create_epoch_pairs()

    def _process_initial_videos(self):
        """Process an initial batch of videos"""
        initial_videos = self._get_next_video_batch(self.initial_pairs * 2)  # 2x for pairs
        for video_path in tqdm(initial_videos, desc="Processing initial videos"):
            self._process_video(video_path)
            self.processed_videos.add(video_path)

    def _get_next_video_batch(self, n_videos):
        """Get next batch of unprocessed videos"""
        remaining = set(self.video_files) - self.processed_videos
        return random.sample(list(remaining), min(n_videos, len(remaining)))

    def _process_video(self, video_path):
        """Process single video and cache frames"""
        cache_path = self.cache_dir / video_path.stem
        if cache_path.exists():
            return

        try:
            cache_path.mkdir(exist_ok=True)
            vr = VideoReader(str(video_path), ctx=cpu())
            n_frames = len(vr)
            
            indices = range(min(n_frames, self.max_frames))
            if n_frames > self.max_frames:
                indices = sorted(random.sample(range(n_frames), self.max_frames))
            
            for idx in indices:
                out_path = cache_path / f"{idx:06d}.png"
                if not out_path.exists():
                    frame = vr[idx]
                    if hasattr(frame, 'asnumpy'):
                        frame = frame.asnumpy()
                    frame = Image.fromarray(np.uint8(frame))
                    frame = frame.resize((self.width, self.height))
                    
                    if self.remove_background:
                        frame = self.remove_bg(frame)
                    
                    frame.save(out_path)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
            return None

    def _create_epoch_pairs(self):
        """Create pairs based on current dataset size"""
        valid_videos = [v for v in self.processed_videos 
                       if (self.cache_dir / v.stem).exists()]
        
        pairs = []
        for _ in range(self.current_pairs):
            if len(valid_videos) < 2:  # Need at least 2 videos
                break
            vid1, vid2 = random.sample(valid_videos, 2)
            pairs.append((vid1, vid2))
        return pairs

    def grow_dataset(self):
        """Increase dataset size for next epoch"""
        self.epoch += 1
        
        # Calculate new size
        self.current_pairs = min(
            int(self.current_pairs * self.growth_rate),
            self.max_pairs
        )
        
        # Process more videos if needed
        videos_needed = self.current_pairs * 2  # 2x for pairs
        if len(self.processed_videos) < videos_needed:
            new_videos = self._get_next_video_batch(
                videos_needed - len(self.processed_videos)
            )
            for video_path in tqdm(new_videos, desc=f"Processing videos for epoch {self.epoch}"):
                self._process_video(video_path)
                self.processed_videos.add(video_path)
        
        # Create new pairs
        self.pairs = self._create_epoch_pairs()
        print(f"Epoch {self.epoch}: Dataset size = {len(self.pairs)} pairs")

    def __getitem__(self, index):
        """Get a pair of frames"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                vid1_path, vid2_path = self.pairs[index]
                
                # Load frames
                source_frame = self._load_random_frame(vid1_path)
                driving_frame = self._load_random_frame(vid2_path)
                
                return {
                    "source_frame": source_frame,
                    "driving_frame": driving_frame,
                    "source_vid": vid1_path.stem,
                    "driving_vid": vid2_path.stem
                }
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} retries for index {index}: {e}")
                    index = random.randrange(len(self))
                continue
    
    def _load_random_frame(self, video_path):
        """Load a random frame from a video"""
        frame_dir = self.cache_dir / video_path.stem
        frame_files = list(frame_dir.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No frames found for {video_path}")
            
        frame_path = random.choice(frame_files)
        frame = Image.open(frame_path).convert('RGB')
        
        if self.transform:
            frame = self.transform(frame)
            if self.apply_warping:
                frame = self.apply_warp_transform(frame)
        else:
            frame = to_tensor(frame)
            
        return frame
    
    def __len__(self):
        return len(self.pairs)

    def reset_pairs(self):
        """Grow dataset and create new pairs for next epoch"""
        self.grow_dataset()

    def remove_bg(self, image):
        """Remove background from image"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")

        if self.use_greenscreen:
            green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))
            final_image = Image.alpha_composite(green_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        return final_image.convert("RGB")

    def apply_warp_transform(self, image_tensor):
        """Apply warping transformation to image"""
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        
        image = to_pil_image(image_tensor)
        image_array = np.array(image)
        
        # Generate random control points for warping
        rows, cols = image_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * self.warp_strength)
        
        # Create and apply the warping transform
        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)
        
        # Apply warping to each channel
        warped_array = np.zeros_like(image_array)
        for i in range(image_array.shape[2]):
            warped_array[..., i] = warp(image_array[..., i], tps, output_shape=(rows, cols))
        
        warped_image = Image.fromarray((warped_array * 255).astype(np.uint8))
        return to_tensor(warped_image)

    def duplicate_frames(self, frames, target_length):
        """Duplicate frames to reach target length"""
        if not frames or len(frames) >= target_length:
            return frames[:target_length]
            
        result = []
        while len(result) < target_length:
            result.extend(frames)
        return result[:target_length]
    