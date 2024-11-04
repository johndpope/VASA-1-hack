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
                n_pairs: int = 10000,
                cache_dir: str = None,
                transform: transforms.Compose = None,
                remove_background: bool = False,
                use_greenscreen: bool = False,
                apply_warping: bool = False,
                max_frames: int = 100,
                duplicate_short: bool = True,
                warp_strength: float = 0.01):
        """
        Enhanced dataset with background removal and warping support.
        
        Args:
            video_dir: Directory containing .mp4 files
            width: Target frame width 
            height: Target frame height
            n_pairs: Number of pairs to sample per epoch
            cache_dir: Directory for LMDB cache
            transform: Torchvision transforms to apply
            remove_background: Whether to remove background from frames
            use_greenscreen: Add green background after removal
            apply_warping: Apply warping augmentation
            max_frames: Maximum frames to keep per video
            duplicate_short: Whether to duplicate frames for short videos
            warp_strength: Strength of warping transform
        """
        self.video_dir = Path(video_dir)
        self.width = width
        self.height = height
        self.n_pairs = n_pairs
        self.transform = transform
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.apply_warping = apply_warping
        self.max_frames = max_frames
        self.duplicate_short = duplicate_short
        self.warp_strength = warp_strength
        
        # Get list of video files
        self.video_files = list(self.video_dir.glob("*.mp4"))
        print(f"Found {len(self.video_files)} video files")
        
        # Setup LMDB cache
        if cache_dir:
            self.cache_env = self._init_cache(cache_dir)
        else:
            self.cache_env = None
            
        # Create random pairs for this epoch
        self.pairs = self._create_epoch_pairs()
        
    def _init_cache(self, cache_dir):
        """Initialize LMDB cache"""
        os.makedirs(cache_dir, exist_ok=True)
        return lmdb.open(
            cache_dir,
            map_size=1024*1024*1024*1024,  # 1TB max size
            create=True,
            readonly=False,
            meminit=False,
            map_async=True
        )
        
    def _create_epoch_pairs(self):
        """Create random pairs for this epoch"""
        pairs = []
        for _ in range(self.n_pairs):
            # Sample two different videos
            vid1, vid2 = random.sample(self.video_files, 2)
            pairs.append((vid1, vid2))
        return pairs

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
    
    def _get_cached_frames(self, video_path):
        """Try to get frames from cache"""
        if not self.cache_env:
            return None
            
        try:
            with self.cache_env.begin(write=False) as txn:
                cached = txn.get(str(video_path).encode())
                if cached:
                    return pickle.loads(cached)
        except:
            return None
            
    def _cache_frames(self, video_path, frames):
        """Cache frames to LMDB"""
        if not self.cache_env:
            return
            
        try:
            with self.cache_env.begin(write=True) as txn:
                txn.put(
                    str(video_path).encode(),
                    pickle.dumps(frames)
                )
        except Exception as e:
            print(f"Cache error for {video_path}: {e}")
        
    def _load_video_frames(self, video_path: Path):
        """Load frames with caching and proper numpy handling"""
        # Try cache first
        frames = self._get_cached_frames(video_path)
        if frames is not None:
            return frames
            
        try:
            vr = VideoReader(str(video_path), ctx=cpu())
            n_frames = len(vr)
            
            # Sample frames
            if n_frames > self.max_frames:
                indices = sorted(random.sample(range(n_frames), self.max_frames))
            else:
                indices = range(n_frames)
                
            frames = []
            for idx in indices:
                # Handle numpy conversion properly
                frame = vr[idx]
                if hasattr(frame, 'asnumpy'):  # Handle decord NDArray
                    frame = frame.asnumpy()
                elif isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                    
                # Convert to PIL
                frame = Image.fromarray(np.uint8(frame))
                frame = frame.resize((self.width, self.height))
                
                # Remove background if needed
                if self.remove_background:
                    frame = self.remove_bg(frame)
                
                # Apply transform
                if self.transform:
                    frame_tensor = self.transform(frame)
                    
                    # Apply warping if enabled
                    if self.apply_warping:
                        frame_tensor = self.apply_warp_transform(frame_tensor)
                        
                    frames.append(frame_tensor)
                else:
                    frames.append(to_tensor(frame))
                    
                # Clean up
                del frame
                if len(frames) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Handle short videos
            if self.duplicate_short and len(frames) < self.max_frames:
                frames = self.duplicate_frames(frames, self.max_frames)
                
            # Cache processed frames
            self._cache_frames(video_path, frames)
            
            return frames
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def __getitem__(self, index):
        """Get a pair of videos with proper error handling"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                vid1_path, vid2_path = self.pairs[index]
                
                # Load frames with retries
                frames1 = self._load_video_frames(vid1_path)
                if frames1 is None:
                    raise ValueError(f"Failed to load {vid1_path}")
                    
                frames2 = self._load_video_frames(vid2_path)
                if frames2 is None:
                    raise ValueError(f"Failed to load {vid2_path}")
                    
                if len(frames1) == 0 or len(frames2) == 0:
                    raise ValueError("Empty frames list")
                    
                # Sample frames
                source_idx = random.randrange(len(frames1))
                driving_idx = random.randrange(len(frames2))
                
                return {
                    "source_frame": frames1[source_idx],
                    "driving_frame": frames2[driving_idx],
                    "source_vid": vid1_path.stem,
                    "driving_vid": vid2_path.stem
                }
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} retries for index {index}: {e}")
                    # Get a new random pair
                    index = random.randrange(len(self))
                continue
            
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    def __len__(self):
        return self.n_pairs
        
    def reset_pairs(self):
        """Reset pairs for new epoch"""
        self.pairs = self._create_epoch_pairs()