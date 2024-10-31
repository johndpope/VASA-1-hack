
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import logging
from typing import Dict, List
import seaborn as sns
from tqdm import tqdm
from dataset import VASADataset 
from rich.console import Console
import torch.nn.functional as F
from rich.traceback import install
console = Console(width=3000)
# Install Rich traceback handling
# install(show_locals=True)
install()

class VASADatasetTester:
    """Comprehensive testing suite for VASA dataset implementation"""
    def __init__(self, dataset: VASADataset, save_dir: str = "test_results"):
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'test_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_all_tests(self):
        """Run all sanity checks"""
        self.logger.info("Starting VASA dataset sanity checks...")
        
        # Basic dataset checks
        self.test_dataset_initialization()
        
        # Test single sample
        sample = self.test_single_sample()
        
        # Test batch loading
        self.test_batch_loading()
        
        # Test specific components
        self.test_emotion_extraction(sample)
        self.test_audio_features(sample)
        self.test_face_attributes(sample)
        
        # Test data distributions
        self.analyze_distributions()
        
        self.logger.info("All sanity checks completed!")

    def test_dataset_initialization(self):
        """Test basic dataset setup"""
        self.logger.info("\nTesting dataset initialization...")
        
        try:
            # Check video paths
            assert len(self.dataset.video_paths) > 0, "No video paths found"
            self.logger.info(f"Found {len(self.dataset.video_paths)} videos")
            
            # Check file existence
            for path in self.dataset.video_paths[:5]:  # Check first 5
                assert Path(path).exists(), f"Video file not found: {path}"
                
                # Get audio path if available
                audio_path = self.dataset.get_audio_path(path)
                if audio_path is not None:
                    assert Path(audio_path).exists(), f"Audio file not found: {audio_path}"
            
            # Check components initialization
            assert self.dataset.face_analyzer is not None, "Face analyzer not initialized"
            assert self.dataset.emotion_recognizer is not None, "Emotion recognizer not initialized"
            assert self.dataset.audio_model is not None, "Audio model not initialized"
            
            self.logger.info("✅ Dataset initialization tests passed")
            
        except Exception as e:
            self.logger.error(f"🔥Dataset initialization failed: {str(e)}")
            raise
    def test_single_sample(self) -> Dict[str, torch.Tensor]:
        """Test single sample extraction"""
        self.logger.info("\nTesting single sample extraction...")
        
        try:
            sample = self.dataset[0]
            
            # Check all required keys
            required_keys = ['frames', 'audio_features', 'gaze', 'distance', 'emotion']
            for key in required_keys:
                assert key in sample, f"Missing key in sample: {key}"
            
            # Check shapes
            self.logger.info("Sample shapes:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"{key}: {value.shape}")
                    
                    # Basic tensor checks
                    assert not torch.isnan(value).any(), f"NaN values found in {key}"
                    assert not torch.isinf(value).any(), f"Inf values found in {key}"
            
            self.visualize_sample(sample)
            self.logger.info("✅ Single sample test passed")
            return sample
            
        except Exception as e:
            self.logger.error(f"🔥Single sample test failed: {str(e)}")
            raise

    def test_batch_loading(self):
        """Test batch loading functionality"""
        self.logger.info("\nTesting batch loading...")
        
        try:
            loader = DataLoader(
                self.dataset,
                batch_size=4,
                shuffle=True,
                num_workers=2
            )
            
            batch = next(iter(loader))
            
            # Check batch shapes
            self.logger.info("Batch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"{key}: {value.shape}")
                    assert value.shape[0] == 4, f"Incorrect batch size for {key}"
            
            self.logger.info("✅ Batch loading test passed")
            
        except Exception as e:
            self.logger.error(f"🔥Batch loading test failed: {str(e)}")
            raise

    def test_emotion_extraction(self, sample: Dict[str, torch.Tensor]):
        """Test emotion extraction specifics"""
        self.logger.info("\nTesting emotion extraction...")
        
        try:
            emotion_tensor = sample['emotion']
            
            # Check emotion tensor properties
            assert emotion_tensor.shape[-1] == 8, "Incorrect number of emotion categories"
            assert torch.allclose(torch.sum(F.softmax(emotion_tensor, dim=-1), dim=-1), 
                                torch.ones_like(torch.sum(F.softmax(emotion_tensor, dim=-1), dim=-1))), \
                "Emotion probabilities don't sum to 1"
            
            # Visualize emotion distributions
            self.visualize_emotions(emotion_tensor)
            
            self.logger.info("✅ Emotion extraction test passed")
            
        except Exception as e:
            self.logger.error(f"🔥Emotion extraction test failed: {str(e)}")
            raise

    def test_audio_features(self, sample: Dict[str, torch.Tensor]):
        """Test audio feature extraction"""
        self.logger.info("\nTesting audio features...")
        
        try:
            audio_features = sample['audio_features']
            
            # Check audio feature properties
            assert len(audio_features.shape) == 3, "Incorrect audio feature dimensions"
            assert not torch.isnan(audio_features).any(), "NaN values in audio features"
            
            # Visualize audio features
            self.visualize_audio_features(audio_features)
            
            self.logger.info("✅ Audio feature test passed")
            
        except Exception as e:
            self.logger.error(f"🔥Audio feature test failed: {str(e)}")
            raise

    def test_face_attributes(self, sample: Dict[str, torch.Tensor]):
        """Test face attribute extraction"""
        self.logger.info("\nTesting face attributes...")
        
        try:
            # Check gaze properties
            assert len(sample['gaze'].shape) == 2, "Incorrect gaze dimensions"
            assert sample['gaze'].shape[-1] == 2, "Incorrect gaze vector size"
            
            # Check distance properties
            assert len(sample['distance'].shape) == 2, "Incorrect distance dimensions"
            assert 0 <= sample['distance'].min() and sample['distance'].max() <= 1, \
                "Distance not normalized"
            
            self.visualize_face_attributes(sample)
            
            self.logger.info("✅ Face attribute test passed")
            
        except Exception as e:
            self.logger.error(f"🔥Face attribute test failed: {str(e)}")
            raise

    def analyze_distributions(self):
        """Analyze data distributions"""
        self.logger.info("\nAnalyzing data distributions...")
        
        try:
            # Sample a subset for distribution analysis
            n_samples = min(100, len(self.dataset))
            samples = [self.dataset[i] for i in tqdm(range(n_samples))]
            
            # Analyze distributions
            distributions = {
                'gaze_theta': torch.cat([s['gaze'][:, 0] for s in samples]),
                'gaze_phi': torch.cat([s['gaze'][:, 1] for s in samples]),
                'distance': torch.cat([s['distance'] for s in samples]),
                'emotion_max': torch.cat([torch.max(F.softmax(s['emotion'], dim=-1), dim=-1)[0] 
                                       for s in samples])
            }
            
            self.visualize_distributions(distributions)
            
            self.logger.info("✅ Distribution analysis completed")
            
        except Exception as e:
            self.logger.error(f"🔥Distribution analysis failed: {str(e)}")
            raise

    def visualize_sample(self, sample: Dict[str, torch.Tensor]):
        """Visualize a single sample"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Plot frames
        for i in range(4):
            frame = sample['frames'][i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(frame)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Frame {i}')
        
        # Plot emotion probabilities
        emotions = F.softmax(sample['emotion'][0], dim=-1)
        axes[1, 0].bar(range(8), emotions.numpy())
        axes[1, 0].set_title('Emotions')
        
        # Plot gaze
        axes[1, 1].scatter(sample['gaze'][:, 0], sample['gaze'][:, 1])
        axes[1, 1].set_title('Gaze Distribution')
        
        # Plot distance
        axes[1, 2].plot(sample['distance'].numpy())
        axes[1, 2].set_title('Head Distance')
        
        # Plot audio features
        axes[1, 3].imshow(sample['audio_features'][0].numpy().T)
        axes[1, 3].set_title('Audio Features')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_visualization.png')
        plt.close()

    def visualize_emotions(self, emotion_tensor: torch.Tensor):
        """Visualize emotion distributions"""
        emotions_prob = F.softmax(emotion_tensor, dim=-1)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=emotions_prob.numpy())
        plt.title('Emotion Distribution')
        emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 
                             'Happy', 'Sad', 'Surprise', 'Neutral']
        plt.xticks(range(8), emotion_labels, rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'emotion_distribution.png')
        plt.close()

    def visualize_audio_features(self, audio_features: torch.Tensor):
        """Visualize audio features"""
        plt.figure(figsize=(12, 4))
        plt.imshow(audio_features[0].numpy().T, aspect='auto')
        plt.colorbar()
        plt.title('Audio Features')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'audio_features.png')
        plt.close()

    def visualize_face_attributes(self, sample: Dict[str, torch.Tensor]):
        """Visualize face attributes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Gaze plot
        ax1.scatter(sample['gaze'][:, 0], sample['gaze'][:, 1])
        ax1.set_title('Gaze Distribution')
        ax1.set_xlabel('θ (yaw)')
        ax1.set_ylabel('φ (pitch)')
        
        # Distance plot
        ax2.plot(sample['distance'].numpy())
        ax2.set_title('Head Distance')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'face_attributes.png')
        plt.close()

    def visualize_distributions(self, distributions: Dict[str, torch.Tensor]):
        """Visualize data distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, (name, data) in enumerate(distributions.items()):
            sns.histplot(data=data.numpy(), ax=axes[i])
            axes[i].set_title(name)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'distributions.png')
        plt.close()

if __name__ == "__main__":
    # Example usage


    dataset = VASADataset(
        video_folder="./junk/",
        max_videos=100, 
        frame_size=(512, 512),
        sequence_length=25,
        cache_audio=True,  # Enable audio caching
        preextract_audio=True,
        random_seed=42
    )
    tester = VASADatasetTester(dataset)
    tester.run_all_tests()

