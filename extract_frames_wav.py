import os
import subprocess
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_audio_stream(video_path):
    """Check if video has audio stream"""
    try:
        command = [
            'ffprobe',
            '-loglevel', 'error',
            '-show_streams',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            check=True
        )
        
        data = json.loads(result.stdout)
        return bool(data.get('streams', []))
        
    except Exception as e:
        logger.error(f"Error checking audio in {video_path}: {e}")
        return False

def extract_audio(video_path, output_path, sampling_rate=16000):
    """Extract audio from video file"""
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sampling_rate),  # Sampling rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            output_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return False

def process_video(video_path, output_folder, cache_info):
    """Process a single video"""
    try:
        # Create output folder if needed
        os.makedirs(output_folder, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = os.path.join(output_folder, f"{video_name}.wav")
        
        # Skip if already processed
        if video_path in cache_info.get('processed', {}):
            logger.info(f"Skipping already processed video: {video_name}")
            return True
        
        # Check for audio stream
        if not check_audio_stream(video_path):
            logger.info(f"No audio stream found in {video_name}")
            cache_info['no_audio'] = cache_info.get('no_audio', []) + [video_path]
            return False
        
        # Extract audio
        success = extract_audio(video_path, output_path)
        
        if success:
            cache_info['processed'] = cache_info.get('processed', {})
            cache_info['processed'][video_path] = output_path
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        return False

def process_videos(
    input_folder,
    output_base_folder,
    max_videos=None,
    sampling_rate=16000,
    cache_file='audio_extraction_cache.json'
):
    """Process multiple videos"""
    # Setup folders
    input_folder = Path(input_folder)
    output_base_folder = Path(output_base_folder)
    output_base_folder.mkdir(exist_ok=True)
    
    # Load cache if exists
    cache_path = output_base_folder / cache_file
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache_info = json.load(f)
        logger.info("Loaded extraction cache")
    else:
        cache_info = {'processed': {}, 'no_audio': []}
    
    # Get video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(input_folder.rglob(f"*{ext}")))
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Filter already processed videos
    video_files = [
        str(v) for v in video_files 
        if str(v) not in cache_info['processed'] 
        and str(v) not in cache_info['no_audio']
    ]
    
    logger.info(f"Found {len(video_files)} unprocessed videos")
    
    # Shuffle and limit if needed
    if max_videos:
        random.shuffle(video_files)
        video_files = video_files[:max_videos]
        logger.info(f"Processing {len(video_files)} videos")
    
    # Process videos
    successful = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Extracting audio"):
        relative_path = Path(video_path).relative_to(input_folder)
        output_folder = output_base_folder / relative_path.parent
        
        if process_video(video_path, output_folder, cache_info):
            successful += 1
        else:
            failed += 1
            
        # Save cache periodically
        if (successful + failed) % 10 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache_info, f)
    
    # Save final cache
    with open(cache_path, 'w') as f:
        json.dump(cache_info, f)
    
    logger.info(f"""
    Audio extraction completed:
    - Successfully processed: {successful} videos
    - Failed/No audio: {failed} videos
    - Total processed: {successful + failed} videos
    """)

if __name__ == "__main__":
    # Example usage
    input_folder = "/path/to/videos"
    output_base_folder = "/path/to/output"
    
    process_videos(
        input_folder=input_folder,
        output_base_folder=output_base_folder,
        max_videos=100,  # Limit number of videos to process
        sampling_rate=16000,  # Audio sampling rate
        cache_file='audio_extraction_cache.json'  # Cache file name
    )