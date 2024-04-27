from dataset import VideoMAE # TBD - https://github.com/search?q=VideoMAE+voxceleb&type=code
# voxceleb2 dataset here - https://github.com/johndpope/VASA-1-hack/issues/5#issuecomment-2077007921
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import get_cosine_schedule_with_warmup
from Net import FaceEncoder, FaceDecoder, DiffusionTransformer
import torchvision.transforms as transforms
from FaceHelper import estimate_gaze, detect_emotions,head_distance_estimator,mediapipe_lip_landmark_detector

# Training configuration
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Initialize models
encoder = FaceEncoder()
decoder = FaceDecoder()
diffusion_transformer = DiffusionTransformer(num_layers=6, num_heads=8, hidden_size=512)

# Initialize optimizer
params = list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion_transformer.parameters())
optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-5)  # Added weight decay for regularization

# Data loading and transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = VideoMAE(
    root='/path/to/voxceleb2/root',
    setting='/path/to/voxceleb2/train.txt',
    train=True,
    image_size=256,
    audio_conf={
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'noise': False,
        'mean': -4.6476,  
        'std': 4.5699,
    },
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Learning rate scheduler setup
total_steps = len(dataloader) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

def compute_lip_sync_loss(original_landmarks, generated_landmarks):
    # Assuming landmarks are tensors of shape [batch_size, num_landmarks, 2 (x, y coordinates)]
    loss_fn = torch.nn.MSELoss()
    return loss_fn(original_landmarks, generated_landmarks)

# Example usage in the training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        video_frames, _, audio_features, _ = batch
        video_frames = video_frames.cuda()
        audio_features = audio_features.cuda()
        
        # Process frames
        for frame_idx in range(video_frames.shape[1]):
            frame = video_frames[:, frame_idx]
            
            # Forward pass through encoder
            appearance_volume, identity_code, head_pose, facial_dynamics = encoder(frame)
            
            # Generate dynamics using structured inputs
            gaze_direction = estimate_gaze(frame)  # Updated to use FaceHelper
            head_distance = head_distance_estimator(frame)  
            emotion_offset = detect_emotions(frame)  # Updated to use FaceHelper hsemotion_onnx
            
            # Diffusion transformer for generating dynamics
            generated_dynamics = diffusion_transformer(
                facial_dynamics, audio_features[:, frame_idx], gaze_direction, head_distance, emotion_offset)
            
            # Face reconstruction
            reconstructed_face = decoder(appearance_volume, identity_code, head_pose, generated_dynamics)
            
            # Get lip landmarks for original and reconstructed face
            original_lip_landmarks = mediapipe_lip_landmark_detector(frame)
            generated_lip_landmarks = mediapipe_lip_landmark_detector(reconstructed_face.detach())  # Detach for inference
            
            # Compute lip sync loss
            lip_sync_loss = compute_lip_sync_loss(original_lip_landmarks, generated_lip_landmarks)
            
            # Reconstruction loss
            reconstruction_loss = nn.L1Loss()(frame, reconstructed_face)
            total_loss = reconstruction_loss + 0.5 * lip_sync_loss  # Weighted sum of losses
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item()}")


