#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

def merge_audio_video(video_path, audio_path, output_path):
    """Merge audio and video using ffmpeg"""
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'aac',   # Convert audio to AAC
        '-shortest',     # Match shortest stream
        '-y',            # Overwrite output
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging {video_path.name}: {e.stderr}")
        return False

def process_folder(base_path, folder_name):
    """Process a single folder (age, ethnicity, gender, or language)"""
    folder_path = Path(base_path) / folder_name

    audio_dir = folder_path / "audios"
    video_dir = folder_path / "videos_crop_512x512"
    output_dir = folder_path / "videos_512x512_with_audio"

    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return False

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return False

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get all video files
    video_files = list(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return False

    print(f"\nProcessing {folder_name} folder...")
    print(f"Found {len(video_files)} video files")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in tqdm(video_files, desc=f"Processing {folder_name}"):
        # Get corresponding audio file
        video_stem = video_path.stem

        # Try different audio extensions
        audio_path = None
        for ext in ['.m4a', '.mp3', '.wav', '.aac']:
            potential_audio = audio_dir / f"{video_stem}{ext}"
            if potential_audio.exists():
                audio_path = potential_audio
                break

        if not audio_path:
            print(f"No audio found for {video_path.name}")
            fail_count += 1
            continue

        # Output path
        output_path = output_dir / video_path.name

        # Skip if already exists
        if output_path.exists():
            skip_count += 1
            continue

        # Merge audio and video
        if merge_audio_video(video_path, audio_path, output_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"Completed {folder_name}:")
    print(f"  - Success: {success_count}")
    print(f"  - Skipped (already exists): {skip_count}")
    print(f"  - Failed: {fail_count}")

    return True

def main():
    parser = argparse.ArgumentParser(description='Merge audio with 512x512 videos for TalkVid dataset')
    parser.add_argument('--base-path', default='/media/12TB/TalkVid/datasets/TalkVid/TalkVid-bench',
                        help='Base path to TalkVid-bench folder')
    parser.add_argument('--folders', nargs='+', default=['age', 'ethnicity', 'gender', 'language'],
                        help='Folders to process')
    args = parser.parse_args()

    base_path = Path(args.base_path)

    if not base_path.exists():
        print(f"Base path not found: {base_path}")
        return

    print(f"Processing TalkVid dataset at: {base_path}")
    print(f"Folders to process: {args.folders}")

    for folder in args.folders:
        process_folder(base_path, folder)

    print("\nAll done!")

if __name__ == "__main__":
    main()