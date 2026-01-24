# Product Requirements Document (PRD) for VASA-1 Model/Class

## 2. Product Overview

### 2.1 Description
VASA-1 generates talking face videos by decomposing facial images into a latent space and modeling motion sequences conditioned on audio and optional signals. Instead of direct video frame generation, it produces latent codes for head pose (z_pose) and facial dynamics (z_dyn), which are decoded into videos using a pre-trained decoder.

The overall framework (as illustrated in the research):  
- **Input:** Single static face image, speech audio clip, optional controls (gaze, head distance, emotion offset).  
- **Process:** Extracts latent variables (appearance volume V_app, identity z_id), generates motion sequences using a Diffusion Transformer, and decodes to video.  
- **Output:** High-quality video (512x512) with synchronized lips, nuanced expressions (e.g., eye blinking, non-lip movements), and natural head motions.  

The model is identity-agnostic, allowing generation for any subject from massive video datasets. Tensor shapes from dataset processing (e.g., motion parameters and audio features) ensure consistency in batch (B), time (T), and feature dimensions, as detailed in Section 4.4.

### 2.2 Objectives
- **Functional:** Produce videos with exquisite lip-audio sync, diverse facial nuances, and lifelike head movements.  
- **Performance:** Real-time generation (40 FPS, negligible latency) on standard hardware (e.g., GPU with ≥8GB VRAM).  
- **Usability:** Simple API/class interface for integration into applications (e.g., Python class with `generate_sequence` method).  
- **Ethical:** Include built-in safeguards (e.g., watermarks, usage restrictions) to prevent harmful applications like deepfakes.

### 2.3 Scope
- **In Scope:** Core model architecture (latent space encoders/decoders, Diffusion Transformer), training pipeline, inference with sliding windows, optional conditioning, and tensor shape validations for data consistency.  
- **Out of Scope:** Full video editing suite, multi-subject generation in one video, integration with external avatars (e.g., MetaHuman), real-time streaming server.

---

## 3. Target Audience and Use Cases

### 3.1 Target Users
- **Researchers/Developers:** AI/ML engineers building talking head systems.  
- **Application Developers:** For apps in education (AI tutors), healthcare (therapeutic avatars), accessibility (sign language interpreters), and entertainment (virtual characters).  
- **End-Users:** Indirectly, via integrated products (e.g., video call enhancements).  
- **Constraints:** Restricted to verified users/researchers due to ethical risks; not for public consumer use.

### 3.2 User Stories
- As a developer, I want to input a photo and audio clip so that I can generate a talking video with natural expressions.  
- As a researcher, I want to control gaze/emotion so that I can test disentangled facial dynamics.  
- As an educator, I want real-time generation so that AI avatars can respond interactively in lessons.  
- As an ethical reviewer, I want built-in detection mechanisms so that generated content is identifiable as AI.

### 3.3 Use Cases
- **Digital Communication:** Enhance video calls with AI avatars for bandwidth savings or anonymity.  
- **Accessibility:** Generate sign language videos or read-aloud content with expressive faces.  
- **Education/Training:** Interactive AI tutors with lifelike responses.  
- **Healthcare:** Therapeutic companions for social interaction (e.g., for autism or elderly care).  
- **Entertainment:** Virtual influencers or character animation in games/movies.

---

## 4. Features and Functionality

### 4.1 Core Features
- **Input Processing:**  
  - Single static face image (any subject).  
  - Speech audio clip (any speaker; processed to features like MFCC with shape [B, T, 13]).  
  - Optional controls: Eye gaze (spherical coordinates), head-camera distance (scalar), emotion offset (vector).  

- **Latent Space Construction:**  
  - Disentangled decomposition: Canonical 3D appearance volume (V_app), identity code (z_id), head pose (z_pose), facial dynamics (z_dyn).  
  - Encoders for extraction; decoder for reconstruction.  
  - Training with swapping losses for disentanglement (e.g., pairwise transfer loss l_consist, cross-identity similarity l_cross_id using deep face features [16] for cross-subject transfers).  

- **Motion Generation:**  
  - Holistic Facial Dynamics Generation (HFDG) using Diffusion Transformer.  
  - Models all facial movements (lips, expressions, gaze, blinking) as a unified latent.  
  - Sliding-window approach for long sequences, with previous K frames as context for seamless transitions.  
  - Output motion tensors: e.g., theta [B, T, 3, 4], expression_embed [B, T, 128].  

- **Conditioning and Guidance:**  
  - Primary: Audio features (Wav2Vec2 or MFCC [B, T, 13]).  
  - Additional: Gaze (g), distance (d), emotion offset (e).  
  - Classifier-Free Guidance (CFG) with dropout (0.1 for most, 0.5 for previous context).  

- **Output:**  
  - Video frames (512x512, up to 40 FPS).  
  - Latent motion sequences { [z_pose_i, z_dyn_i] }, with shapes like rotation/translation/scale [B, T, 3].  

### 4.2 Technical Architecture
- **Face Latent Space (Section 3.1):**  
  - Based on 3D-aware face reenactment [64, 19].  
  - Decompose image into V_app (3D volume), z_id, z_pose (3D head pose), z_dyn (facial dynamics).  
  - Encoders: Independent for each component; V_app uses posed-to-canonical warping.  
  - Decoder: Warps V_app back and reconstructs image.  
  - Training Losses: Reconstruction with swapping; l_consist (pairwise pose/dynamics transfer); l_cross_id (cosine similarity on face identity features [16] for cross-subject transfers).  
  - Achieves high disentanglement and expressiveness for identity-agnostic generation.  

- **Diffusion Transformer (Section 3.2):**  
  - Transformer architecture [58, 38, 52] for sequence generation.  
  - Motion Sequence: X = { [z_pose_i, z_dyn_i] } for window length W (e.g., T=50).  
  - Diffusion Formulation: Denoising score matching; predicts clean signal X_0 (not noise).  
  - Input: Noisy latent X_t + conditions C concatenated temporally.  
  - Conditioning Signals: Audio A (Wav2Vec2 features [B, T, 768] or MFCC [B, T, 13]); gaze g (θ, φ) [B, T, 2]; distance d (scalar) [B, T, 1]; emotion e (offset from [43]) [B, T, 2]; previous context (last K frames of audio/motion).  
  - CFG: Dropout during training (0.1 general); inference scales λ_c per condition.  
  - Handles variable-length audio by dropping last frames during training.  

- **Inference Process (Section 3.3):**  
  - Extract V_app and z_id from input image.  
  - Split audio into windows of length W.  
  - Generate motion sequences sliding-window style using Diffusion Transformer.  
  - Decode latents to video frames using trained decoder.  
  - Real-time: DDIM sampling for efficiency; negligible latency.  

### 4.3 Non-Functional Requirements
- **Performance:** 40 FPS at 512x512; low latency (<100ms start).  
- **Scalability:** Handle sequences up to 1000 frames (configurable max_motion_length).  
- **Reliability:** Robust to short audio; seamless window transitions; validate tensor dtypes (e.g., float32 for features, ensure device consistency like cuda:0 during extraction).  
- **Security/Ethics:** Watermark outputs; restrict access; detect generated content.  

### 4.4 Tensor Shape Specifications
Based on dataset processing logs, the following tensor shapes must be enforced for consistency in batch (B=1 for single samples), time (T=50 for window size), and feature dimensions. All tensors should be float32 dtype unless specified, with device management (e.g., cuda:0 during extraction, cpu for final storage).

- **Motion Parameters (from EMO Feature Extraction):**  
  - `theta`: [B, T, 3, 4] – 3D pose matrix (rotation + translation).  
  - `scale`: [B, T, 3] – Scaling factors.  
  - `rotation`: [B, T, 3] – Euler angles or rotation components.  
  - `translation`: [B, T, 3] – Translation vectors.  
  - `expression_embed`: [B, T, 128] – Facial dynamics embeddings.  

- **Audio Features (from MFCC Processing):**  
  - Raw MFCC: [B, 13, interpolated_T] (e.g., [1, 13, 134] before interpolation).  
  - Interpolated MFCC: [B, T, 13] (e.g., [1, 50, 13] after resizing to match window size).  
  - Value Range: Typically [-114.698, 92.191] raw; [-106.956, 90.613] interpolated (normalize if needed).  

- **Frame Data:**  
  - Individual Frames: [B, 3, 512, 512] – RGB images (float32, range [0, 1] or [0, 255]).  

- **Audio Segments:**  
  - Raw Audio: [B, samples] (e.g., [1, 26666] for 1.67s at 16kHz).  

These shapes ensure compatibility with the Diffusion Transformer input (concatenated temporally) and decoder. Validation should include checks for NaN/Inf values and device consistency during extraction.

---

## 5. Technical Requirements

### 5.1 System Requirements
- **Hardware:** GPU (e.g., NVIDIA RTX 30xx+ with ≥8GB VRAM) for training/inference.  
- **Software:** Python 3.12+; PyTorch 2.0+; Diffusers library; Wav2Vec2 for audio features.  
- **Dependencies:** NumPy, OpenCV, MediaPipe (for gaze/distance extraction in training); pre-trained models (e.g., [2] for gaze, [17] for distance, [43] for emotion).  
- **Data:** Unlabeled talking face videos for training (e.g., VoxCeleb dataset).  

### 5.2 Implementation Details
- **Class Structure:** VASAModel class with forward (training), generate_sequence (inference).  
- **Hyperparameters:** From config (e.g., d_model=512, n_layers=8, diffusion_steps=1000).  
- **Optimization:** AdamW; mixed precision (AMP); gradient checkpointing optional.  
- **Tensor Handling:** Enforce shapes as in 4.4; use torch.float32 dtype; manage devices (e.g., cuda:0 for extraction).

---