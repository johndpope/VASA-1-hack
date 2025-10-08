#!/usr/bin/env python3
"""
Expression Audit Tool - Compare ground truth vs generated expressions

Usage:
    python audit_expressions.py \
        --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
        --identity ./data/IMG_1.png \
        --config overfit_config.yaml \
        --checkpoint checkpoints_overfit/best_checkpoint.pt
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from omegaconf import OmegaConf
import torchaudio
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class ExpressionAuditor:
    def __init__(self, config_path: str, checkpoint_path: str):
        """Initialize the auditor with config and checkpoint."""
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the VASA model from checkpoint."""
        from vasa_model import VASAModel
        import importlib

        logger.info(f"Loading volumetric avatar...")
        model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
        emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
        self.volumetric_avatar = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)

        model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        self.volumetric_avatar.load_state_dict(model_dict, strict=False)
        self.volumetric_avatar = self.volumetric_avatar.cuda()
        self.volumetric_avatar.eval()
        for param in self.volumetric_avatar.parameters():
            param.requires_grad = False

        logger.info(f"Loading VASA model...")
        self.model = VASAModel(self.config, self.volumetric_avatar)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.cuda()

        logger.info(f"✅ Model loaded successfully")

    def load_ground_truth_from_cache(self, video_path: str = None):
        """Load ground truth expressions and theta from single bucket cache.

        Args:
            video_path: If provided, only load windows from this specific video.
                       If None, loads ALL windows.
        """
        cache_path = Path("cache_single_bucket/all_windows_cache.h5")

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        logger.info(f"Loading ground truth from cache: {cache_path}")
        if video_path:
            logger.info(f"Filtering for video: {video_path}")

        gt_data = {
            'theta': [],
            'expression': [],
            'audio': [],
        }

        with h5py.File(cache_path, 'r') as f:
            # Load windows - filter by video path if provided
            window_keys = sorted([k for k in f.keys() if k.startswith('window_')],
                                key=lambda x: int(x.split('_')[1]))

            logger.info(f"Found {len(window_keys)} total windows in cache")

            windows_loaded = 0
            for key in window_keys:
                window = f[key]

                # Filter by video path if provided
                if video_path:
                    window_video_path = window.attrs.get('video_path', '')
                    if window_video_path != video_path:
                        continue

                windows_loaded += 1
                # Extract data - note: theta and expression_embed are the targets
                gt_data['theta'].append(torch.tensor(window['theta'][:]).unsqueeze(0))  # [1, T, 3, 4]
                gt_data['expression'].append(torch.tensor(window['expression_embed'][:]).unsqueeze(0))  # [1, T, 128]
                gt_data['audio'].append(torch.tensor(window['audio_features'][:]).unsqueeze(0))  # [1, T, 768]

            logger.info(f"Loaded {windows_loaded} windows matching criteria")

        # Concatenate all windows
        gt_data['theta'] = torch.cat(gt_data['theta'], dim=1)  # [1, total_T, 3, 4]
        gt_data['expression'] = torch.cat(gt_data['expression'], dim=1)  # [1, total_T, 128]
        gt_data['audio'] = torch.cat(gt_data['audio'], dim=1)  # [1, total_T, 768]

        logger.info(f"Loaded ground truth: theta {gt_data['theta'].shape}, expr {gt_data['expression'].shape}")

        return gt_data

    def generate_predictions(self, gt_audio_features: torch.Tensor, identity_path: str):
        """Generate expressions and theta using the model with GT audio features.

        Args:
            gt_audio_features: Ground truth audio features from cache [1, T, 768]
            identity_path: Path to identity image
        """
        from PIL import Image
        import torchvision.transforms as transforms

        num_frames = gt_audio_features.shape[1]
        logger.info(f"Generating predictions for {num_frames} frames using GT audio features")

        # Load identity image
        identity_img = Image.open(identity_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        identity = transform(identity_img).unsqueeze(0).cuda()  # [1, 3, 512, 512]

        # Use the EXACT same audio features as ground truth
        audio_features = gt_audio_features.cuda()
        logger.info(f"Using GT audio features: shape={audio_features.shape}, mean={audio_features.mean():.6f}")

        # Generate motion parameters
        logger.info(f"Running inference with audio shape: {audio_features.shape}")

        with torch.no_grad():
            # Prepare conditions
            B, T = audio_features.shape[:2]
            device = audio_features.device

            conditions = {
                'audio_features': audio_features,
                'gaze': torch.zeros(B, T, 2, device=device),
                'emotion': torch.zeros(B, T, 2, device=device),
                'blink': torch.zeros(B, T, 3, device=device),
                'speed_bucket': torch.zeros(B, T, 1, device=device, dtype=torch.long),
            }

            # Initial pose and dynamics (zeros for start)
            initial_pose = {
                'theta': torch.zeros(B, 3, 4, device=device),
            }
            initial_dynamics = torch.zeros(B, self.config.model.expression_dim, device=device)

            # Generate identity embedding from identity image
            with torch.no_grad():
                # Extract face mask and get identity embedding
                face_mask, _, _, _ = self.model.volumetric_avatar.face_idt.forward(identity)
                face_mask = (face_mask > 0.6).float()
                source_masked = identity * face_mask
                idt_embed = self.model.volumetric_avatar.idt_embedder_nw(source_masked)  # [B, 128]

            # Generate motion using DDIM sampling
            outputs = self.model.generate_sequence(
                initial_pose=initial_pose,
                initial_dynamics=initial_dynamics,
                conditions=conditions,
                num_steps=self.config.inference.get('num_diffusion_steps', 50),
                eta=self.config.inference.get('eta', 0.8),
                cfg_scales=None,  # Use default CFG scales
                idt_embed=idt_embed,  # CRITICAL: Pass identity embedding for warp generation
            )

        pred_data = {
            'theta': outputs['theta'].cpu(),  # [B, T, 3, 4]
            'expression': outputs['expression_embed'].cpu(),  # [B, T, 128]
            'audio': gt_audio_features.cpu(),  # Same as GT (should be identical)
        }

        logger.info(f"✅ Generated predictions: theta {pred_data['theta'].shape}, expr {pred_data['expression'].shape}")

        return pred_data

    def compare_expressions(self, gt_data: dict, pred_data: dict, output_dir: Path):
        """Compare ground truth vs predicted expressions."""
        output_dir.mkdir(exist_ok=True, parents=True)

        # Extract arrays
        gt_expr = gt_data['expression'][0].numpy()  # [T, 128]
        pred_expr = pred_data['expression'][0].numpy()  # [T, 128]

        gt_theta = gt_data['theta'][0].numpy()  # [T, 3, 4]
        pred_theta = pred_data['theta'][0].numpy()  # [T, 3, 4]

        # Ensure same length
        min_len = min(gt_expr.shape[0], pred_expr.shape[0])
        gt_expr = gt_expr[:min_len]
        pred_expr = pred_expr[:min_len]
        gt_theta = gt_theta[:min_len]
        pred_theta = pred_theta[:min_len]

        logger.info(f"Comparing {min_len} frames")

        # 1. Expression L2 distance per frame
        expr_l2 = np.linalg.norm(gt_expr - pred_expr, axis=1)  # [T]

        # 2. Theta L2 distance per frame (flatten 3x4 to 12)
        gt_theta_flat = gt_theta.reshape(min_len, -1)  # [T, 12]
        pred_theta_flat = pred_theta.reshape(min_len, -1)  # [T, 12]
        theta_l2 = np.linalg.norm(gt_theta_flat - pred_theta_flat, axis=1)  # [T]

        # 3. Cosine similarity per frame
        from scipy.spatial.distance import cosine
        expr_cosine = np.array([1 - cosine(gt_expr[i], pred_expr[i]) for i in range(min_len)])

        # Statistics
        logger.info(f"Expression L2 Distance: mean={expr_l2.mean():.4f}, std={expr_l2.std():.4f}")
        logger.info(f"Theta L2 Distance: mean={theta_l2.mean():.4f}, std={theta_l2.std():.4f}")
        logger.info(f"Expression Cosine Sim: mean={expr_cosine.mean():.4f}, std={expr_cosine.std():.4f}")

        # Plot comparison
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # Expression L2 distance
        axes[0].plot(expr_l2, label='Expression L2 Distance', color='blue')
        axes[0].axhline(expr_l2.mean(), color='red', linestyle='--', label=f'Mean: {expr_l2.mean():.4f}')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('L2 Distance')
        axes[0].set_title('Expression L2 Distance (GT vs Pred)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Theta L2 distance
        axes[1].plot(theta_l2, label='Theta L2 Distance', color='green')
        axes[1].axhline(theta_l2.mean(), color='red', linestyle='--', label=f'Mean: {theta_l2.mean():.4f}')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('L2 Distance')
        axes[1].set_title('Theta L2 Distance (GT vs Pred)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Cosine similarity
        axes[2].plot(expr_cosine, label='Expression Cosine Similarity', color='purple')
        axes[2].axhline(expr_cosine.mean(), color='red', linestyle='--', label=f'Mean: {expr_cosine.mean():.4f}')
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('Cosine Similarity')
        axes[2].set_title('Expression Cosine Similarity (GT vs Pred) - Higher is Better')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'expression_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Saved comparison plot: {plot_path}")
        plt.close()

        # Save detailed CSV
        import pandas as pd
        df = pd.DataFrame({
            'frame': range(min_len),
            'expr_l2': expr_l2,
            'theta_l2': theta_l2,
            'expr_cosine': expr_cosine,
        })
        csv_path = output_dir / 'expression_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"📝 Saved metrics CSV: {csv_path}")

        # Identify worst frames
        worst_expr_idx = np.argsort(expr_l2)[-10:][::-1]
        logger.info(f"🔴 Top 10 worst expression frames: {worst_expr_idx.tolist()}")
        logger.info(f"   L2 distances: {expr_l2[worst_expr_idx].tolist()}")

        return {
            'expr_l2': expr_l2,
            'theta_l2': theta_l2,
            'expr_cosine': expr_cosine,
        }

    def check_audio_alignment(self, gt_data: dict, pred_data: dict):
        """Check if audio features are aligned."""
        gt_audio = gt_data['audio'][0].detach().cpu().numpy()  # [T, 768]
        pred_audio = pred_data['audio'][0].detach().cpu().numpy()  # [T, 768]

        min_len = min(gt_audio.shape[0], pred_audio.shape[0])
        gt_audio = gt_audio[:min_len]
        pred_audio = pred_audio[:min_len]

        # Check if audio is identical (should be since we're using GT audio)
        audio_diff = np.abs(gt_audio - pred_audio).mean()

        logger.info(f"Audio feature difference: {audio_diff:.6f}")

        if audio_diff < 1e-6:
            logger.info("✅ Audio features are IDENTICAL (using same GT audio)")
        else:
            logger.error(f"⚠️  Audio features DIFFER by {audio_diff:.6f} - THIS IS A BUG!")
            logger.error("   Pred and GT should be using same audio tensor!")

        return audio_diff


def main():
    parser = argparse.ArgumentParser(description='Audit expression generation vs ground truth')
    parser.add_argument('--video', type=str, required=True,
                        help='Video path (same as training video)')
    parser.add_argument('--identity', type=str, default='./data/IMG_1.png',
                        help='Identity image path')
    parser.add_argument('--config', type=str, default='overfit_config.yaml',
                        help='Config path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_overfit/best_checkpoint.pt',
                        help='Checkpoint path')
    parser.add_argument('--output-dir', type=str, default='expression_audit',
                        help='Output directory for audit results')

    args = parser.parse_args()

    # Initialize auditor
    auditor = ExpressionAuditor(args.config, args.checkpoint)

    # Load ground truth
    logger.info(f"📂 Loading ground truth from cache...")
    gt_data = auditor.load_ground_truth_from_cache(args.video)

    # Generate predictions using EXACT same audio features as GT
    logger.info(f"🎯 Generating predictions with GT audio features...")
    pred_data = auditor.generate_predictions(gt_data['audio'], args.identity)

    # Compare expressions
    logger.info(f"📊 Comparing expressions...")
    output_dir = Path(args.output_dir)
    metrics = auditor.compare_expressions(gt_data, pred_data, output_dir)

    # Check audio alignment
    logger.info(f"🎵 Checking audio alignment...")
    audio_diff = auditor.check_audio_alignment(gt_data, pred_data)

    logger.info(f"✅ Audit complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
