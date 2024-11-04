import os
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import torch
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
import numpy as np
from tqdm import tqdm
from rembg import remove
import io
import argparse

def process_single_video(args):
    video_path, output_dir, config = args
    try:
        # Create output directory for this video
        video_id = Path(video_path).stem
        frames_dir = Path(output_dir) / video_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if (Path(output_dir) / video_id / "processed.txt").exists():
            return f"Skipped {video_id} - already processed"

        # Initialize video reader
        video_reader = VideoReader(str(video_path))
        total_frames = len(video_reader)
        processed_frames = []

        # Process frames
        for frame_idx in range(min(total_frames, config['max_frames'])):
            # Extract frame
            frame = video_reader[frame_idx].asnumpy()
            pil_frame = Image.fromarray(frame)
            
            # Apply basic transforms
            if config['width'] and config['height']:
                pil_frame = pil_frame.resize((config['width'], config['height']))
            
            # Remove background if requested
            if config['remove_background']:
                img_byte_arr = io.BytesIO()
                pil_frame.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                bg_removed_bytes = remove(img_byte_arr)
                pil_frame = Image.open(io.BytesIO(bg_removed_bytes)).convert('RGB')

            # Save frame
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            pil_frame.save(frame_path)
            
            # Create and save tensor
            tensor = transforms.ToTensor()(pil_frame)
            processed_frames.append(tensor)

        # Save tensors in batch
        tensor_path = Path(output_dir) / video_id / "frames.npz"
        np.savez_compressed(str(tensor_path), *[t.numpy() for t in processed_frames])
        
        # Mark as processed
        with open(Path(output_dir) / video_id / "processed.txt", "w") as f:
            f.write("done")
            
        return f"Processed {video_id}"
        
    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def preprocess_dataset(video_dir: str, json_file: str, output_dir: str, config: dict):
    """
    Preprocesses all videos in the dataset in parallel
    """
    # Load video information
    with open(json_file, 'r') as f:
        dataset_info = json.load(f)
    
    # Prepare processing tasks
    tasks = []
    for video_id in dataset_info['clips'].keys():
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            tasks.append((video_path, output_dir, config))
    
    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=config['num_workers']) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), 
                          total=len(tasks), 
                          desc="Processing videos"))
    
    # Report results
    print("\nProcessing Summary:")
    for result in results:
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing input videos')
    parser.add_argument('--json_file', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed frames')
    parser.add_argument('--width', type=int, default=256, help='Target frame width')
    parser.add_argument('--height', type=int, default=256, help='Target frame height')
    parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to extract per video')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--remove_background', action='store_true', help='Remove background from frames')
    
    args = parser.parse_args()
    
    config = {
        'width': args.width,
        'height': args.height,
        'max_frames': args.max_frames,
        'num_workers': args.num_workers,
        'remove_background': args.remove_background
    }
    
    preprocess_dataset(args.video_dir, args.json_file, args.output_dir, config)