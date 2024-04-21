import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader
from omegaconf import OmegaConf
from typing import List, Dict, Any
from EmoDataset import EMODataset
from decord import AudioReader




def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"

    # Unpack and flatten the images and audio frames from the batch
    all_images = []
    all_audio_frames = []
    for item in batch:
        all_images.extend(item['images'])
        all_audio_frames.extend(item['audio_frames'])

    assert all(isinstance(img, torch.Tensor) for img in all_images), "All images must be PyTorch tensors"
    assert all(isinstance(audio, torch.Tensor) for audio in all_audio_frames), "All audio frames must be PyTorch tensors"

    # Determine the maximum dimensions
    assert all(img.ndim == 3 for img in all_images), "All images must be 3D tensors"
    max_height = max(img.shape[1] for img in all_images)
    max_width = max(img.shape[2] for img in all_images)

    # Pad the images
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in all_images]

    # Stack the padded images and audio frames
    images_tensor = torch.stack(padded_images)
    audio_frames_tensor = torch.stack(all_audio_frames)

    # Assert the correct shape of the output tensors
    assert images_tensor.ndim == 4, "Images tensor should be 4D"
    assert audio_frames_tensor.ndim == 3, "Audio frames tensor should be 3D"

    return {'images': images_tensor, 'audio_frames': audio_frames_tensor}




def train_model(model, data_loader, optimizer, criterion, device, num_epochs, cfg):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in data_loader:
            reference_images = batch['images'].to(device)
            audio_frames = batch['audio_frames'].to(device)
            head_rotation_speeds = batch['head_rotation_speeds'].to(device)

            batch_size = reference_images.size(0)
            total_loss = 0.0

            for i in range(batch_size):
                reference_image = reference_images[i].unsqueeze(0)
                audio_frame = audio_frames[i].unsqueeze(0)
                head_rotation_speed = head_rotation_speeds[i].unsqueeze(0)

                optimizer.zero_grad()  # Zero the parameter gradients

                # Forward pass using the reference image, audio frame, and head rotation speed
                generated_frame = model(reference_image, audio_frame, head_rotation_speed)

                # Compute the loss
                loss = criterion(generated_frame, reference_image)

                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Perform a single optimization step (parameter update)

                total_loss += loss.item()

            batch_loss = total_loss / batch_size
            running_loss += batch_loss

        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model



def main(cfg: OmegaConf) -> None:
    transform = transforms.Compose([
        transforms.Resize((cfg.data.train_height, cfg.data.train_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = EMODataset(
        use_gpu=cfg.training.use_gpu_video_tensor,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        data_dir='./images_folder',
        video_dir=cfg.training.video_data_dir,
        json_file='./data/overfit.json',
        stage='stage3-speedlayers',
        transform=transform
    )

    # Configuration and Hyperparameters
    num_epochs = 10  # Example number of epochs
    learning_rate = 1e-3  # Example learning rate

    # Initialize Dataset and DataLoader
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=gpu_padded_collate)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    # Save the model
    torch.save(trained_model.state_dict(), 'emo_model_stage3.pth')
    print("Model saved to emo_model_stage3.pth")

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage3.yaml")
    main(config)