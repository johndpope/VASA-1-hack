import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

class EnhancedLipAnalyzer:
    def __init__(self, static_mode=False):
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define lip landmark indices
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Define mouth states
        self.mouth_states = {
            'closed': {'color': 'red', 'threshold': 0.05},
            'slightly_open': {'color': 'yellow', 'threshold': 0.1},
            'open': {'color': 'green', 'threshold': 0.2},
            'wide_open': {'color': 'blue', 'threshold': float('inf')}
        }

    def analyze_frame(self, image):
        """Analyze a single frame and return lip metrics."""
        # Process image with MediaPipe
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract lip landmarks
        upper_lip = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] 
                             for idx in self.UPPER_LIP])
        lower_lip = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] 
                             for idx in self.LOWER_LIP])
        
        # Calculate metrics
        metrics = {
            'openness': self._calculate_openness(upper_lip, lower_lip),
            'symmetry': self._calculate_symmetry(upper_lip, lower_lip),
            'shape': self._analyze_shape(upper_lip, lower_lip)
        }
        
        # Determine mouth state
        metrics['state'] = self._determine_mouth_state(metrics['openness'])
        
        return {
            'upper_lip': upper_lip,
            'lower_lip': lower_lip,
            'metrics': metrics
        }

    def _calculate_openness(self, upper_lip, lower_lip):
        """Calculate vertical distance between lips."""
        # Use middle points for measurement
        upper_middle = upper_lip[len(upper_lip)//2]
        lower_middle = lower_lip[len(lower_lip)//2]
        return np.abs(upper_middle[1] - lower_middle[1])

    def _calculate_symmetry(self, upper_lip, lower_lip):
        """Calculate lip symmetry score."""
        def get_symmetric_halves(points):
            # Get center index
            center_idx = len(points) // 2
            
            # If odd number of points, exclude center point from comparison
            if len(points) % 2 == 1:
                left = points[:center_idx]
                right = points[center_idx + 1:]
            else:
                left = points[:center_idx]
                right = points[center_idx:]
            
            # Flip right side for comparison
            right = np.flip(right, axis=0)
            
            # Ensure equal length
            min_len = min(len(left), len(right))
            left = left[:min_len]
            right = right[:min_len]
            
            return left, right
        
        # Get symmetric halves for upper and lower lips
        upper_left, upper_right = get_symmetric_halves(upper_lip)
        lower_left, lower_right = get_symmetric_halves(lower_lip)
        
        # Compute symmetry scores
        if len(upper_left) > 0:
            upper_diff = np.mean(np.linalg.norm(upper_left - upper_right, axis=1))
        else:
            upper_diff = 0
            
        if len(lower_left) > 0:
            lower_diff = np.mean(np.linalg.norm(lower_left - lower_right, axis=1))
        else:
            lower_diff = 0
        
        # Return symmetry score (1 = perfect symmetry, 0 = asymmetric)
        if upper_diff == 0 and lower_diff == 0:
            return 1.0
        return 1.0 - np.mean([upper_diff, lower_diff])

    def _analyze_shape(self, upper_lip, lower_lip):
        """Analyze lip shape characteristics."""
        # Calculate area and perimeter
        area = self._calculate_area(np.vstack([upper_lip[:, :2], lower_lip[:, :2]]))
        perimeter = self._calculate_perimeter(upper_lip, lower_lip)
        
        # Calculate aspect ratio (width/height)
        width = np.linalg.norm(upper_lip[0] - upper_lip[-1])
        height = self._calculate_openness(upper_lip, lower_lip)
        aspect_ratio = width / (height + 1e-6)  # Avoid division by zero
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio
        }

    def _determine_mouth_state(self, openness):
        """Determine mouth state based on openness."""
        for state, params in self.mouth_states.items():
            if openness <= params['threshold']:
                return state
        return 'wide_open'

    def visualize(self, image, analysis_result):
        """Create visualization with metrics."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot original image with landmarks
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot lips
        upper_lip = analysis_result['upper_lip']
        lower_lip = analysis_result['lower_lip']
        
        # Scale points to image coordinates
        h, w = image.shape[:2]
        upper_lip_scaled = np.column_stack((upper_lip[:, 0] * w, upper_lip[:, 1] * h))
        lower_lip_scaled = np.column_stack((lower_lip[:, 0] * w, lower_lip[:, 1] * h))
        
        # Draw lips with color based on state
        state = analysis_result['metrics']['state']
        color = self.mouth_states[state]['color']
        
        ax1.plot(upper_lip_scaled[:, 0], upper_lip_scaled[:, 1], color=color, linewidth=2)
        ax1.plot(lower_lip_scaled[:, 0], lower_lip_scaled[:, 1], color=color, linewidth=2)
        ax1.set_title(f'Mouth State: {state}')
        
        # Create metrics visualization
        metrics = analysis_result['metrics']
        shape = metrics['shape']
        
        # Plot metrics
        metric_names = ['openness', 'symmetry', 'aspect_ratio', 'area', 'perimeter']
        metric_values = [
            metrics['openness'],
            metrics['symmetry'],
            shape['aspect_ratio'],
            shape['area'],
            shape['perimeter']
        ]
        
        # Normalize values for visualization
        normalized_values = [v / max(metric_values) for v in metric_values]
        
        # Create bar plot
        bars = ax2.bar(metric_names, normalized_values)
        ax2.set_ylim(0, 1.2)
        ax2.set_title('Lip Metrics (Normalized)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def _calculate_area(self, points):
        """Calculate area of lip region using shoelace formula."""
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _calculate_perimeter(self, upper_lip, lower_lip):
        """Calculate perimeter of lips."""
        def segment_length(points):
            return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
            
        upper_length = segment_length(upper_lip)
        lower_length = segment_length(lower_lip)
        return upper_length + lower_length



    def process_video(self, video_path: str, output_dir: str, save_frames: bool = True):
            """Process video and save analysis frames."""
            import os
            from datetime import datetime
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            frames_dir = os.path.join(output_dir, f'analysis_{timestamp}')
            if save_frames:
                os.makedirs(frames_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize metrics storage
            frame_metrics = []
            
            try:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Analyze frame
                    analysis = self.analyze_frame(frame)
                    if analysis is not None:
                        # Store metrics
                        metrics = analysis['metrics']
                        metrics['frame_number'] = frame_count
                        frame_metrics.append(metrics)
                        
                        if save_frames:
                            # Generate visualization
                            fig = self.visualize(frame, analysis)
                            
                            # Save frame
                            frame_path = os.path.join(frames_dir, f'frame_{frame_count:06d}.png')
                            fig.savefig(frame_path, bbox_inches='tight')
                            plt.close(fig)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # Progress update every 30 frames
                        print(f"Processed {frame_count}/{total_frames} frames")
                        
                # Generate summary metrics
                summary = self._generate_summary(frame_metrics)
                
                # Plot summary graphs
                summary_fig = self._plot_summary(frame_metrics)
                summary_path = os.path.join(output_dir, f'summary_{timestamp}.png')
                summary_fig.savefig(summary_path, bbox_inches='tight')
                plt.close(summary_fig)
                
                return summary, frames_dir
                
            finally:
                cap.release()
            
    def _generate_summary(self, frame_metrics):
        """Generate summary statistics from frame metrics."""
        if not frame_metrics:
            return {}
            
        summary = {
            'total_frames': len(frame_metrics),
            'mouth_states': {}
        }
        
        # Count mouth states
        for metrics in frame_metrics:
            state = metrics['state']
            summary['mouth_states'][state] = summary['mouth_states'].get(state, 0) + 1
            
        # Calculate percentage for each state
        total = len(frame_metrics)
        summary['mouth_states'] = {
            state: count / total * 100 
            for state, count in summary['mouth_states'].items()
        }
        
        # Calculate average metrics
        avg_metrics = {
            'avg_openness': np.mean([m['openness'] for m in frame_metrics]),
            'avg_symmetry': np.mean([m['symmetry'] for m in frame_metrics]),
            'avg_aspect_ratio': np.mean([m['shape']['aspect_ratio'] for m in frame_metrics])
        }
        summary.update(avg_metrics)
        
        return summary
        
    def _plot_summary(self, frame_metrics):
        """Create summary visualization of video analysis."""
        if not frame_metrics:
            return plt.figure()  # Return empty figure if no metrics
            
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Time series of metrics
        ax1 = plt.subplot(2, 1, 1)
        frames = [m['frame_number'] for m in frame_metrics]
        
        # Plot metrics over time
        ax1.plot(frames, [m['openness'] for m in frame_metrics], 
                label='Openness', color='blue')
        ax1.plot(frames, [m['symmetry'] for m in frame_metrics], 
                label='Symmetry', color='green')
        ax1.plot(frames, [m['shape']['aspect_ratio'] for m in frame_metrics], 
                label='Aspect Ratio', color='red')
        
        ax1.set_title('Metrics Over Time')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # 2. Mouth state distribution
        ax2 = plt.subplot(2, 1, 2)
        state_counts = {}
        for metrics in frame_metrics:
            state = metrics['state']
            state_counts[state] = state_counts.get(state, 0) + 1
            
        # Create pie chart of states
        states = list(state_counts.keys())
        counts = list(state_counts.values())
        colors = [self.mouth_states[state]['color'] for state in states]
        
        ax2.pie(counts, labels=states, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Distribution of Mouth States')
        
        plt.tight_layout()
        return fig

def process_video_file(video_path: str, output_dir: str, save_frames: bool = True):
    """Helper function to process a video file."""
    analyzer = EnhancedLipAnalyzer(static_mode=False)
    summary, frames_dir = analyzer.process_video(video_path, output_dir, save_frames)
    
    # Print summary
    print("\nVideo Analysis Summary:")
    print(f"Total Frames: {summary['total_frames']}")
    print("\nMouth State Distribution:")
    for state, percentage in summary['mouth_states'].items():
        print(f"{state}: {percentage:.1f}%")
    print("\nAverage Metrics:")
    print(f"Average Openness: {summary['avg_openness']:.3f}")
    print(f"Average Symmetry: {summary['avg_symmetry']:.3f}")
    print(f"Average Aspect Ratio: {summary['avg_aspect_ratio']:.3f}")
    
    return summary, frames_dir


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load('vasa_config.yaml')
    
    summary, frames_dir = process_video_file(
        video_path=f"{config.paths.video_folder}/3.mp4",
        output_dir="output",
        save_frames=True  # Set to False if you don't want to save individual frames
    )