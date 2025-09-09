#!/usr/bin/env python3
"""
Bulk process videos using nemo/pipeline2.py with high-quality identity image.
This will process all videos in junk/ folder using nemo/data/IMG_1.png as the identity source.
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_single_video(source_image_path, video_path, output_path, fps=25, max_len=1000):
    """
    Process a single video using pipeline2.py
    
    Args:
        source_image_path: Path to identity image (IMG_1.png)
        video_path: Path to driving video
        output_path: Path to save result
        fps: Output video FPS
        max_len: Maximum frames to process
    """
    # Change to nemo directory for pipeline2.py to work properly
    original_dir = os.getcwd()
    nemo_dir = os.path.join(original_dir, 'nemo')
    
    try:
        os.chdir(nemo_dir)
        
        # Build the command
        cmd = [
            sys.executable,  # Use current Python interpreter
            'pipeline2.py',
            '--source_image_path', source_image_path,
            '--driven_video_path', video_path,
            '--saved_to_path', output_path,
            '--fps', str(fps),
            '--max_len', str(max_len)
        ]
        
        logger.info(f"Processing: {os.path.basename(video_path)}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully processed: {os.path.basename(video_path)}")
            return True
        else:
            logger.error(f"✗ Failed to process: {os.path.basename(video_path)}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception processing {video_path}: {str(e)}")
        return False
    finally:
        # Always change back to original directory
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="Bulk process videos with high-quality identity")
    parser.add_argument('--identity_image', type=str, 
                        default='data/IMG_1.png',
                        help='Path to identity image (relative to nemo/)')
    parser.add_argument('--input_dir', type=str, 
                        default='../junk',
                        help='Input video directory (relative to nemo/)')
    parser.add_argument('--output_dir', type=str, 
                        default='data/bulk_results',
                        help='Output directory (relative to nemo/)')
    parser.add_argument('--video_pattern', type=str, 
                        default='*.mp4',
                        help='Video file pattern to match')
    parser.add_argument('--fps', type=float, 
                        default=25.0,
                        help='Output video FPS')
    parser.add_argument('--max_len', type=int, 
                        default=300,
                        help='Maximum frames per video')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have output')
    parser.add_argument('--max_videos', type=int, 
                        default=None,
                        help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent.absolute()
    nemo_dir = script_dir / 'nemo'
    
    # Verify nemo directory exists
    if not nemo_dir.exists():
        logger.error(f"nemo directory not found at: {nemo_dir}")
        sys.exit(1)
    
    # Build full paths relative to nemo directory
    identity_path = nemo_dir / args.identity_image
    input_dir = nemo_dir / args.input_dir
    output_dir = nemo_dir / args.output_dir
    
    # Verify identity image exists
    if not identity_path.exists():
        logger.error(f"Identity image not found: {identity_path}")
        sys.exit(1)
    
    logger.info(f"Using identity image: {identity_path}")
    
    # Verify input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Find all videos
    video_pattern = str(input_dir / args.video_pattern)
    video_files = sorted(glob.glob(video_pattern))
    
    if not video_files:
        logger.error(f"No videos found matching pattern: {video_pattern}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Limit number of videos if specified
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        logger.info(f"Processing only first {args.max_videos} videos")
    
    # Process statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = Path(video_path).stem
        output_filename = f"{video_name}_identity.mp4"
        output_path = output_dir / output_filename
        
        # Skip if output exists and skip_existing is True
        if args.skip_existing and output_path.exists():
            logger.info(f"Skipping existing: {output_filename}")
            skipped += 1
            continue
        
        # Convert paths to relative paths from nemo directory
        identity_rel = os.path.relpath(identity_path, nemo_dir)
        video_rel = os.path.relpath(video_path, nemo_dir)
        output_rel = os.path.relpath(output_path, nemo_dir)
        
        # Process the video
        success = process_single_video(
            identity_rel,
            video_rel, 
            output_rel,
            args.fps,
            args.max_len
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Small delay between videos to avoid overwhelming the system
        time.sleep(1)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    
    if successful > 0:
        logger.info(f"\nResults saved in: {output_dir}")
        
        # List output files
        output_files = list(output_dir.glob("*.mp4"))
        if output_files:
            logger.info(f"Generated {len(output_files)} output videos:")
            for f in output_files[:5]:  # Show first 5
                logger.info(f"  - {f.name}")
            if len(output_files) > 5:
                logger.info(f"  ... and {len(output_files) - 5} more")

if __name__ == "__main__":
    main()