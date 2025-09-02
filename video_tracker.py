from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from logger import logger

class VideoEvent(Enum):
    """Enum for different video processing events"""
    VIDEO_TOO_SHORT = "video_too_short"
    NO_AUDIO = "no_audio"
    INVALID_FRAMES = "invalid_frames"
    FACE_DETECTION_FAILED = "face_detection_failed"
    LANDMARK_DETECTION_FAILED = "landmark_detection_failed"
    PROCESSING_ERROR = "processing_error"
    NO_VALID_WINDOWS = "no_valid_windows"

@dataclass
class VideoEventData:
    """Data class for video events"""
    video_path: str
    event_type: VideoEvent
    details: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            from time import time
            self.timestamp = time()

class ProblematicVideosTracker:
    """Singleton tracker for problematic video events."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, output_dir: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super(ProblematicVideosTracker, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, output_dir: Optional[Path] = None):
        # Only initialize once
        if not ProblematicVideosTracker._initialized:
            if output_dir is None:
                raise ValueError("output_dir must be provided for initial initialization")
                
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize files
            self.invalid_videos_path = self.output_dir / "invalid_videos.txt"
            self.event_log_path = self.output_dir / "video_events.log"
            self.summary_path = self.output_dir / "processing_summary.txt"
            
            # Initialize tracking collections
            self.problematic_videos = set()
            self.failed_videos: Dict[str, List[VideoEventData]] = {}
            self.event_counts: Dict[VideoEvent, int] = {
                event: 0 for event in VideoEvent
            }
            
            # Initialize event handlers
            self._handlers: Dict[VideoEvent, List[Callable]] = {
                event: [] for event in VideoEvent
            }
            
            # Register default handlers
            self.register_default_handlers()
            
            ProblematicVideosTracker._initialized = True
            logger.info(f"Initialized ProblematicVideosTracker with output dir: {output_dir}")

    @classmethod
    def get_instance(cls) -> 'ProblematicVideosTracker':
        """Get singleton instance."""
        if cls._instance is None:
            raise RuntimeError(
                "ProblematicVideosTracker not initialized. "
                "Create an instance with output_dir first."
            )
        return cls._instance

    def register_default_handlers(self):
        """Register the default event handlers."""
        for event in VideoEvent:
            self.register_handler(event, self._log_event)
            self.register_handler(event, self._track_failure)
            self.register_handler(event, self._update_counts)

    def register_handler(self, event: VideoEvent, handler: Callable):
        """Register a custom handler for an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def dispatch(self, event_data: VideoEventData):
        """Dispatch an event to all registered handlers."""
        for handler in self._handlers.get(event_data.event_type, []):
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in event handler: {str(e)}")

    def _log_event(self, event_data: VideoEventData):
        """Log event to file."""
        timestamp = datetime.fromtimestamp(event_data.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        with open(self.event_log_path, 'a') as f:
            f.write(f"{timestamp}\t{event_data.event_type.value}\t{event_data.video_path}\t{event_data.details}\n")

    def _track_failure(self, event_data: VideoEventData):
        """Track video failure."""
        if event_data.video_path not in self.failed_videos:
            self.failed_videos[event_data.video_path] = []
        self.failed_videos[event_data.video_path].append(event_data)
        self.problematic_videos.add(event_data.video_path)
        self._save_failures()

    def _update_counts(self, event_data: VideoEventData):
        """Update event counts."""
        self.event_counts[event_data.event_type] += 1

    def _save_failures(self):
        """Save failed videos report."""
        with open(self.invalid_videos_path, 'w') as f:
            f.write("# Failed Videos Report\n\n")
            for video_path, events in self.failed_videos.items():
                f.write(f"\n{video_path}:\n")
                for event in sorted(events, key=lambda e: e.timestamp):
                    timestamp = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"  {timestamp} - {event.event_type.value}: {event.details}\n")

    def save_summary(self):
        """Save processing summary."""
        with open(self.summary_path, 'w') as f:
            f.write("# Video Processing Summary\n\n")
            f.write(f"Total problematic videos: {len(self.problematic_videos)}\n\n")
            f.write("Event Counts:\n")
            for event, count in self.event_counts.items():
                if count > 0:
                    f.write(f"  {event.value}: {count}\n")

    def get_problematic_videos(self) -> Set[str]:
        """Get set of all problematic video paths."""
        return self.problematic_videos

    def get_event_stats(self) -> Dict[str, int]:
        """Get statistics about events."""
        return {
            event.value: count 
            for event, count in self.event_counts.items()
            if count > 0
        }

    def add_video(self, video_path: str, issue: str, event_type: Optional[VideoEvent] = None):
        """Legacy method for compatibility - converts to event dispatch."""
        if event_type is None:
            event_type = VideoEvent.PROCESSING_ERROR
            
        self.dispatch(VideoEventData(
            video_path=video_path,
            event_type=event_type,
            details={"issue": issue}
        ))

    def print_summary(self):
        """Print summary of processing issues."""
        logger.info("\nVideo Processing Summary:")
        logger.info(f"Total problematic videos: {len(self.problematic_videos)}")
        
        for event, count in self.event_counts.items():
            if count > 0:
                logger.info(f"  {event.value}: {count}")

    def reset(self):
        """Reset all tracking data."""
        self.problematic_videos.clear()
        self.failed_videos.clear()
        self.event_counts = {event: 0 for event in VideoEvent}
        logger.info("Reset all tracking data")


import os
import json
import re

def parse_log_file(log_path="./bad_videos/video_events.log"):
    """
    Parse the video events log file to extract failed video paths
    """
    failed_videos = set()  # Using set to avoid duplicates

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Parse the line (tab-separated)
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        timestamp, error_type, filepath = parts[:3]
                        
                        # Check for either type of failure
                        if error_type in ['face_detection_failed', 'landmark_detection_failed', 'video_too_short']:
                            failed_videos.add(filepath)
                            
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    print(f"Error: {str(e)}")
                    continue

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return set()
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return set()

    return failed_videos

def remove_failed_videos(log_path="./bad_videos/video_events.log"):
    """
    Remove videos that failed processing based on the event log
    """
    failed_videos = parse_log_file(log_path)
    
    if not failed_videos:
        print("No failed videos found in the log file.")
        return
    
    removed_count = 0
    failed_count = 0
    skipped_count = 0
    
    print(f"\nAttempting to remove {len(failed_videos)} unique failed videos...")
    
    for filepath in failed_videos:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Removed: {filepath}")
                removed_count += 1
            else:
                print(f"Skipped (not found): {filepath}")
                skipped_count += 1
        except Exception as e:
            print(f"Error removing {filepath}: {str(e)}")
            failed_count += 1
    
    print(f"\nSummary:")
    print(f"Successfully removed: {removed_count} files")
    print(f"Failed to remove: {failed_count} files")
    print(f"Skipped (not found): {skipped_count} files")
    print(f"Total unique failed videos in log: {len(failed_videos)}")


import os
import glob
from pathlib import Path
import h5py
import logging

logger = logging.getLogger(__name__)

def clean_window_cache(cache_dir: str = "cache", pattern: str = "window_*.h5"):
    """
    Delete all window cache H5 files.
    
    Args:
        cache_dir: Directory containing cache files
        pattern: File pattern to match cache files
    """
    try:
        # Convert to Path object for better path handling
        cache_path = Path(cache_dir)
        
        if not cache_path.exists():
            logger.warning(f"Cache directory {cache_dir} does not exist")
            return
            
        # Find all matching cache files
        cache_files = list(cache_path.glob(pattern))
        
        if not cache_files:
            logger.info(f"No cache files found matching pattern {pattern}")
            return
            
        logger.info(f"Found {len(cache_files)} cache files to delete")
        
        # Delete each file
        for cache_file in cache_files:
            try:
                # Ensure file is closed if it's open
                try:
                    with h5py.File(cache_file, 'r') as f:
                        pass
                except Exception:
                    pass
                    
                # Delete the file
                cache_file.unlink()
                logger.info(f"Deleted cache file: {cache_file}")
                
            except Exception as e:
                logger.error(f"Error deleting {cache_file}: {str(e)}")
                
        logger.info("Cache cleaning complete")
        
    except Exception as e:
        logger.error(f"Error cleaning cache: {str(e)}")

def clean_specific_window(window_idx: int, cache_dir: str = "cache"):
    """
    Delete a specific window cache file.
    
    Args:
        window_idx: Index of window to delete
        cache_dir: Directory containing cache files
    """
    try:
        cache_path = Path(cache_dir)
        target_file = cache_path / f"window_{window_idx}.h5"
        
        if not target_file.exists():
            logger.warning(f"Cache file for window {window_idx} does not exist")
            return
            
        try:
            # Ensure file is closed
            with h5py.File(target_file, 'r') as f:
                pass
        except Exception:
            pass
            
        # Delete the file
        target_file.unlink()
        logger.info(f"Deleted cache file for window {window_idx}")
        
    except Exception as e:
        logger.error(f"Error deleting window {window_idx}: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Allow log path to be specified as command line argument
    log_path = sys.argv[1] if len(sys.argv) > 1 else "./bad_videos/video_events.log"

    confirm = input(f"This will delete all failed videos listed in {log_path}. Continue? [y/N]: ")
    if confirm.lower() == 'y':
        remove_failed_videos(log_path)
    else:
        print("Operation cancelled.")

    confirm = input(f"Clear all cached h5 files? {log_path}. Continue? [y/N]: ")
    if confirm.lower() == 'y':
        clean_window_cache(cache_dir = "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/window_cache")  # Clean all window cache
    else:
        print("Operation cancelled.")


      