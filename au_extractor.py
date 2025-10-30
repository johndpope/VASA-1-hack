"""
Action Unit (AU) Extraction from MediaPipe Face Mesh Landmarks

This module computes 16 facial Action Units based on geometric relationships
between MediaPipe Face Mesh landmarks (468 points). Each AU represents a
specific facial muscle movement with intensity in range [0, 1].

Reference: Facial Action Coding System (FACS) by Ekman & Friesen (1978)
MediaPipe: https://google.github.io/mediapipe/solutions/face_mesh
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from logger import logger  # Import logger first (handles MediaPipe log suppression)
import mediapipe as mp


# MediaPipe Face Mesh landmark indices (468 total landmarks)
# Key regions for AU computation:
LANDMARKS = {
    # Eyes
    'left_eye_outer': 33,
    'left_eye_inner': 133,
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_outer': 263,
    'right_eye_inner': 362,
    'right_eye_top': 386,
    'right_eye_bottom': 374,

    # Eyebrows
    'left_eyebrow_inner': 70,
    'left_eyebrow_middle': 107,
    'left_eyebrow_outer': 66,
    'right_eyebrow_inner': 300,
    'right_eyebrow_middle': 336,
    'right_eyebrow_outer': 296,

    # Nose
    'nose_tip': 1,
    'nose_bridge': 6,
    'left_nostril': 98,
    'right_nostril': 327,

    # Mouth
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'upper_lip_top': 0,
    'upper_lip_bottom': 12,
    'lower_lip_top': 15,
    'lower_lip_bottom': 17,
    'mouth_corner_left': 61,
    'mouth_corner_right': 291,

    # Face outline
    'chin': 152,
    'forehead': 10,
    'left_cheek': 234,
    'right_cheek': 454,
}


class ActionUnitExtractor:
    """
    Extract 16 Action Units from MediaPipe Face Mesh landmarks.

    AUs Implemented:
    - AU1: Inner Brow Raiser
    - AU2: Outer Brow Raiser
    - AU4: Brow Lowerer
    - AU5: Upper Lid Raiser
    - AU6: Cheek Raiser
    - AU7: Lid Tightener
    - AU9: Nose Wrinkler
    - AU10: Upper Lip Raiser
    - AU12: Lip Corner Puller
    - AU15: Lip Corner Depressor
    - AU17: Chin Raiser
    - AU20: Lip Stretcher
    - AU23: Lip Tightener
    - AU25: Lips Part
    - AU26: Jaw Drop
    - AU27: Mouth Stretch
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Store baseline measurements (computed from first frame or neutral expression)
        self.baseline = None

    def compute_distance(self, lm1: np.ndarray, lm2: np.ndarray) -> float:
        """Euclidean distance between two landmarks."""
        return np.linalg.norm(lm1 - lm2)

    def compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline measurements from neutral expression.

        Args:
            landmarks: Face landmarks [468, 3] from neutral frame
        """
        self.baseline = {
            'eye_height_left': self.compute_distance(
                landmarks[LANDMARKS['left_eye_top']],
                landmarks[LANDMARKS['left_eye_bottom']]
            ),
            'eye_height_right': self.compute_distance(
                landmarks[LANDMARKS['right_eye_top']],
                landmarks[LANDMARKS['right_eye_bottom']]
            ),
            'brow_to_eye_left': self.compute_distance(
                landmarks[LANDMARKS['left_eyebrow_middle']],
                landmarks[LANDMARKS['left_eye_top']]
            ),
            'brow_to_eye_right': self.compute_distance(
                landmarks[LANDMARKS['right_eyebrow_middle']],
                landmarks[LANDMARKS['right_eye_top']]
            ),
            'mouth_height': self.compute_distance(
                landmarks[LANDMARKS['mouth_top']],
                landmarks[LANDMARKS['mouth_bottom']]
            ),
            'mouth_width': self.compute_distance(
                landmarks[LANDMARKS['mouth_left']],
                landmarks[LANDMARKS['mouth_right']]
            ),
            'face_height': self.compute_distance(
                landmarks[LANDMARKS['forehead']],
                landmarks[LANDMARKS['chin']]
            ),
        }
        logger.info("✅ AU baseline measurements set from neutral expression")

    def extract_aus(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract 16 Action Units from face landmarks.

        Args:
            landmarks: MediaPipe face landmarks [468, 3] in normalized coordinates

        Returns:
            AU intensities [16] with values in [0, 1]
        """
        # Set baseline if not already set
        if self.baseline is None:
            self.set_baseline(landmarks)

        aus = np.zeros(16, dtype=np.float32)

        # Get key landmark positions
        left_eye_top = landmarks[LANDMARKS['left_eye_top']]
        left_eye_bottom = landmarks[LANDMARKS['left_eye_bottom']]
        right_eye_top = landmarks[LANDMARKS['right_eye_top']]
        right_eye_bottom = landmarks[LANDMARKS['right_eye_bottom']]

        left_brow_inner = landmarks[LANDMARKS['left_eyebrow_inner']]
        left_brow_middle = landmarks[LANDMARKS['left_eyebrow_middle']]
        left_brow_outer = landmarks[LANDMARKS['left_eyebrow_outer']]
        right_brow_inner = landmarks[LANDMARKS['right_eyebrow_inner']]
        right_brow_middle = landmarks[LANDMARKS['right_eyebrow_middle']]
        right_brow_outer = landmarks[LANDMARKS['right_eyebrow_outer']]

        mouth_left = landmarks[LANDMARKS['mouth_left']]
        mouth_right = landmarks[LANDMARKS['mouth_right']]
        mouth_top = landmarks[LANDMARKS['mouth_top']]
        mouth_bottom = landmarks[LANDMARKS['mouth_bottom']]

        nose_tip = landmarks[LANDMARKS['nose_tip']]
        nose_bridge = landmarks[LANDMARKS['nose_bridge']]
        chin = landmarks[LANDMARKS['chin']]
        forehead = landmarks[LANDMARKS['forehead']]

        # AU1: Inner Brow Raiser
        # Measure vertical distance between inner brows and eyes
        inner_brow_dist = (
            self.compute_distance(left_brow_inner, left_eye_top) +
            self.compute_distance(right_brow_inner, right_eye_top)
        ) / 2
        aus[0] = np.clip((inner_brow_dist - self.baseline['brow_to_eye_left']) /
                         self.baseline['brow_to_eye_left'], 0, 1)

        # AU2: Outer Brow Raiser
        # Measure vertical distance between outer brows and eyes
        outer_brow_dist = (
            self.compute_distance(left_brow_outer, left_eye_top) +
            self.compute_distance(right_brow_outer, right_eye_top)
        ) / 2
        aus[1] = np.clip((outer_brow_dist - self.baseline['brow_to_eye_left']) /
                         self.baseline['brow_to_eye_left'], 0, 1)

        # AU4: Brow Lowerer
        # Opposite of AU1/AU2 - negative change means brow lowered
        brow_lower = -(inner_brow_dist - self.baseline['brow_to_eye_left']) / \
                     self.baseline['brow_to_eye_left']
        aus[2] = np.clip(brow_lower, 0, 1)

        # AU5: Upper Lid Raiser
        # Measure eye opening height
        eye_height = (
            self.compute_distance(left_eye_top, left_eye_bottom) +
            self.compute_distance(right_eye_top, right_eye_bottom)
        ) / 2
        aus[3] = np.clip((eye_height - self.baseline['eye_height_left']) /
                         self.baseline['eye_height_left'], 0, 1)

        # AU6: Cheek Raiser
        # Measure distance from nose to mouth (decreases when cheeks raise)
        nose_to_mouth = self.compute_distance(nose_tip, mouth_top)
        face_height = self.baseline['face_height']
        aus[4] = np.clip(1.0 - nose_to_mouth / (face_height * 0.3), 0, 1)

        # AU7: Lid Tightener
        # Eye opening becomes smaller (opposite of AU5)
        aus[5] = np.clip(-(eye_height - self.baseline['eye_height_left']) /
                         self.baseline['eye_height_left'], 0, 1)

        # AU9: Nose Wrinkler
        # Measure nose width (increases when wrinkling)
        left_nostril = landmarks[LANDMARKS['left_nostril']]
        right_nostril = landmarks[LANDMARKS['right_nostril']]
        nose_width = self.compute_distance(left_nostril, right_nostril)
        baseline_nose_width = self.baseline['face_height'] * 0.1
        aus[6] = np.clip((nose_width - baseline_nose_width) / baseline_nose_width, 0, 1)

        # AU10: Upper Lip Raiser
        # Measure vertical distance between nose and upper lip
        upper_lip = landmarks[LANDMARKS['upper_lip_top']]
        nose_to_lip = self.compute_distance(nose_tip, upper_lip)
        aus[7] = np.clip(1.0 - nose_to_lip / (face_height * 0.15), 0, 1)

        # AU12: Lip Corner Puller (smile)
        # Measure angle of mouth corners (increases for smile)
        left_angle = self.compute_angle(nose_tip, mouth_left, chin)
        right_angle = self.compute_angle(nose_tip, mouth_right, chin)
        avg_angle = (left_angle + right_angle) / 2
        aus[8] = np.clip((avg_angle - np.pi * 0.4) / (np.pi * 0.2), 0, 1)

        # AU15: Lip Corner Depressor (frown)
        # Opposite of AU12 - corners pull down
        aus[9] = np.clip((np.pi * 0.4 - avg_angle) / (np.pi * 0.2), 0, 1)

        # AU17: Chin Raiser
        # Measure distance from lower lip to chin (decreases when chin raises)
        lower_lip = landmarks[LANDMARKS['lower_lip_bottom']]
        lip_to_chin = self.compute_distance(lower_lip, chin)
        aus[10] = np.clip(1.0 - lip_to_chin / (face_height * 0.15), 0, 1)

        # AU20: Lip Stretcher
        # Measure mouth width increase
        mouth_width = self.compute_distance(mouth_left, mouth_right)
        aus[11] = np.clip((mouth_width - self.baseline['mouth_width']) /
                          self.baseline['mouth_width'], 0, 1)

        # AU23: Lip Tightener
        # Mouth width decreases (opposite of AU20)
        aus[12] = np.clip(-(mouth_width - self.baseline['mouth_width']) /
                          self.baseline['mouth_width'], 0, 1)

        # AU25: Lips Part
        # Measure mouth opening height
        mouth_height = self.compute_distance(mouth_top, mouth_bottom)
        aus[13] = np.clip((mouth_height - self.baseline['mouth_height']) /
                          self.baseline['mouth_height'], 0, 1)

        # AU26: Jaw Drop
        # Measure jaw opening (large mouth opening)
        aus[14] = np.clip((mouth_height - self.baseline['mouth_height']) /
                          (self.baseline['mouth_height'] * 2), 0, 1)

        # AU27: Mouth Stretch
        # Combined width and height increase
        mouth_area = mouth_width * mouth_height
        baseline_area = self.baseline['mouth_width'] * self.baseline['mouth_height']
        aus[15] = np.clip((mouth_area - baseline_area) / baseline_area, 0, 1)

        return aus

    def extract_aus_from_video(
        self,
        frames: List[np.ndarray],
        num_queries: int = 8
    ) -> torch.Tensor:
        """
        Extract AU sequence from video frames and pool to num_queries.

        Args:
            frames: List of RGB frames [H, W, 3] as numpy arrays
            num_queries: Number of queries to pool AUs to (default: 8)

        Returns:
            AU tensor [num_queries, 16] with pooled AU intensities
        """
        au_sequence = []

        # Process each frame
        for frame in frames:
            # Convert to RGB if needed
            if frame.shape[-1] == 4:  # RGBA
                frame = frame[:, :, :3]

            # Run MediaPipe face mesh
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                # Extract landmarks as numpy array [468, 3]
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in results.multi_face_landmarks[0].landmark
                ])

                # Compute AUs
                aus = self.extract_aus(landmarks)
                au_sequence.append(aus)
            else:
                # No face detected - use zeros
                logger.warning("No face detected in frame, using zero AUs")
                au_sequence.append(np.zeros(16, dtype=np.float32))

        # Convert to tensor [T, 16]
        au_tensor = torch.tensor(np.array(au_sequence), dtype=torch.float32)

        # Pool to num_queries using average pooling
        if len(au_tensor) < num_queries:
            # Pad if too short
            padded = torch.zeros(num_queries, 16)
            padded[:len(au_tensor)] = au_tensor
            return padded
        else:
            # Average pool to match num_queries
            # Reshape to [1, 16, T] for pooling
            au_reshaped = au_tensor.T.unsqueeze(0)  # [1, 16, T]

            kernel_size = max(1, len(au_tensor) // num_queries)
            stride = kernel_size

            pooled = torch.nn.functional.avg_pool1d(
                au_reshaped,
                kernel_size=kernel_size,
                stride=stride
            )  # [1, 16, num_queries]

            pooled_aus = pooled.squeeze(0).T  # [num_queries, 16]

            # Ensure exactly num_queries
            if len(pooled_aus) > num_queries:
                pooled_aus = pooled_aus[:num_queries]
            elif len(pooled_aus) < num_queries:
                padded = torch.zeros(num_queries, 16)
                padded[:len(pooled_aus)] = pooled_aus
                pooled_aus = padded

            return pooled_aus


# AU names for visualization and logging
AU_NAMES = [
    "AU1_Inner_Brow_Raiser",
    "AU2_Outer_Brow_Raiser",
    "AU4_Brow_Lowerer",
    "AU5_Upper_Lid_Raiser",
    "AU6_Cheek_Raiser",
    "AU7_Lid_Tightener",
    "AU9_Nose_Wrinkler",
    "AU10_Upper_Lip_Raiser",
    "AU12_Lip_Corner_Puller",
    "AU15_Lip_Corner_Depressor",
    "AU17_Chin_Raiser",
    "AU20_Lip_Stretcher",
    "AU23_Lip_Tightener",
    "AU25_Lips_Part",
    "AU26_Jaw_Drop",
    "AU27_Mouth_Stretch",
]


def compute_au_statistics(au_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for AU tensor for validation.

    Args:
        au_tensor: AU intensities [T, 16] or [num_queries, 16]

    Returns:
        Dictionary with mean, std, min, max per AU
    """
    stats = {}
    for i, name in enumerate(AU_NAMES):
        au_values = au_tensor[:, i]
        stats[name] = {
            'mean': au_values.mean().item(),
            'std': au_values.std().item(),
            'min': au_values.min().item(),
            'max': au_values.max().item(),
        }
    return stats


if __name__ == "__main__":
    # Test AU extraction on dummy data
    import cv2

    extractor = ActionUnitExtractor()

    # Create dummy frame (black image)
    frame = np.zeros((512, 512, 3), dtype=np.uint8)

    # Test with dummy frames
    frames = [frame] * 50  # 50 frames

    try:
        au_tensor = extractor.extract_aus_from_video(frames, num_queries=8)
        print(f"✅ AU extraction successful!")
        print(f"   Output shape: {au_tensor.shape}")
        print(f"   Value range: [{au_tensor.min():.3f}, {au_tensor.max():.3f}]")

        # Print statistics
        stats = compute_au_statistics(au_tensor)
        print("\n📊 AU Statistics:")
        for name, s in stats.items():
            print(f"   {name}: mean={s['mean']:.3f}, std={s['std']:.3f}")
    except Exception as e:
        print(f"❌ AU extraction failed: {e}")
        import traceback
        traceback.print_exc()
