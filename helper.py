import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import wandb
import os
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def consistent_sub_sample(tensor1, tensor2, sub_sample_size):
    """
    Consistently sub-sample two tensors with the same random offset.
    
    Args:
    tensor1 (torch.Tensor): First input tensor of shape (B, C, H, W)
    tensor2 (torch.Tensor): Second input tensor of shape (B, C, H, W)
    sub_sample_size (tuple): Desired sub-sample size (h, w)
    
    Returns:
    tuple: Sub-sampled versions of tensor1 and tensor2
    """
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    assert tensor1.ndim == 4, "Input tensors should have 4 dimensions (B, C, H, W)"
    
    batch_size, channels, height, width = tensor1.shape
    sub_h, sub_w = sub_sample_size
    
    assert height >= sub_h and width >= sub_w, "Sub-sample size should not exceed the tensor dimensions"
    
    offset_x = torch.randint(0, height - sub_h + 1, (1,)).item()
    offset_y = torch.randint(0, width - sub_w + 1, (1,)).item()
    
    tensor1_sub = tensor1[..., offset_x:offset_x+sub_h, offset_y:offset_y+sub_w]
    tensor2_sub = tensor2[..., offset_x:offset_x+sub_h, offset_y:offset_y+sub_w]
    
    return tensor1_sub, tensor2_sub


def plot_loss_landscape(model, loss_fns, dataloader, num_points=20, alpha=1.0):
    # Store original parameters
    original_params = [p.clone() for p in model.parameters()]
    
    # Calculate two random directions
    direction1 = [torch.randn_like(p) for p in model.parameters()]
    direction2 = [torch.randn_like(p) for p in model.parameters()]
    
    # Normalize directions
    norm1 = torch.sqrt(sum(torch.sum(d**2) for d in direction1))
    norm2 = torch.sqrt(sum(torch.sum(d**2) for d in direction2))
    direction1 = [d / norm1 for d in direction1]
    direction2 = [d / norm2 for d in direction2]
    
    # Create grid
    x = np.linspace(-alpha, alpha, num_points)
    y = np.linspace(-alpha, alpha, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate loss for each point and each loss function
    Z = {f'loss_{i}': np.zeros_like(X) for i in range(len(loss_fns))}
    Z['total_loss'] = np.zeros_like(X)
    
    for i in range(num_points):
        for j in range(num_points):
            # Update model parameters
            for p, d1, d2 in zip(model.parameters(), direction1, direction2):
                p.data = p.data + X[i,j] * d1 + Y[i,j] * d2
            
            # Calculate loss for each loss function
            total_loss = 0
            num_batches = 0
            for batch in dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                for k, loss_fn in enumerate(loss_fns):
                    loss = loss_fn(outputs, targets)
                    Z[f'loss_{k}'][i,j] += loss.item()
                    total_loss += loss.item()
                num_batches += 1
            
            # Average the losses
            for k in range(len(loss_fns)):
                Z[f'loss_{k}'][i,j] /= num_batches
            Z['total_loss'][i,j] = total_loss / num_batches
            
            # Reset model parameters
            for p, orig_p in zip(model.parameters(), original_params):
                p.data = orig_p.clone()
    
    # Plot the loss landscapes
    figs = []
    for loss_key in Z.keys():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z[loss_key], cmap='viridis')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_zlabel('Loss')
        ax.set_title(f'Loss Landscape - {loss_key}')
        fig.colorbar(surf)
        figs.append(fig)
    
    # Save the plots to buffers
    bufs = []
    for fig in figs:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        bufs.append(buf)
        plt.close(fig)
    
    return bufs

def log_loss_landscape(model, loss_fns, dataloader, step):
    # Generate the loss landscape plots
    bufs = plot_loss_landscape(model, loss_fns, dataloader)
    
    # Log the plots to wandb
    log_dict = {
        f"loss_landscape_{i}": wandb.Image(buf, caption=f"Loss Landscape - Loss {i}")
        for i, buf in enumerate(bufs[:-1])
    }
    log_dict["loss_landscape_total"] = wandb.Image(bufs[-1], caption="Loss Landscape - Total Loss")
    log_dict["step"] = step
    
    wandb.log(log_dict)




# Global variable to store the current table structure
current_table_columns = None
def log_grad_flow(named_parameters, global_step):
    global current_table_columns

    grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            grads.append(p.grad.abs().mean().item())
    
    if not grads:
        print("No valid gradients found for logging.")
        return
    
    # Normalize gradients
    max_grad = max(grads)
    if max_grad == 0:
        print("👿👿👿 Warning: All gradients are zero. 👿👿👿")
        normalized_grads = grads  # Use unnormalized grads if max is zero
        raise ValueError(f"👿👿👿 Warning: All gradients are zero. 👿👿👿")
    else:
        normalized_grads = [g / max_grad for g in grads]

    # Create the matplotlib figure
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(grads)), normalized_grads, alpha=0.5)
    plt.xticks(range(len(grads)), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.title(f"Gradient Flow (Step {global_step})")
    if max_grad == 0:
        plt.title(f"Gradient Flow (Step {global_step}) - All Gradients Zero")
    plt.tight_layout()

    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Create a wandb.Image from the buffer
    img = wandb.Image(Image.open(buf))
    
    plt.close()

    # Calculate statistics
    stats = {
        "max_gradient": max_grad,
        "min_gradient": min(grads),
        "mean_gradient": np.mean(grads),
        "median_gradient": np.median(grads),
        "gradient_variance": np.var(grads),
    }

    # Check for gradient issues
    issues = check_gradient_issues(grads, layers)

    # Log everything
    log_dict = {
        "gradient_flow_plot": img,
        **stats,
        "gradient_issues": wandb.Html(issues),
        "step": global_step
    }

    # Log other metrics
    wandb.log(log_dict)


def check_gradient_issues(grads, layers):
    issues = []
    mean_grad = np.mean(grads)
    std_grad = np.std(grads)
    
    for layer, grad in zip(layers, grads):
        if grad > mean_grad + 3 * std_grad:
            issues.append(f"🔥 Potential exploding gradient in {layer}: {grad:.2e}")
        elif grad < mean_grad - 3 * std_grad:
            issues.append(f"🥶 Potential vanishing gradient in {layer}: {grad:.2e}")
    
    if issues:
        return "<br>".join(issues)
    else:
        return "✅ No significant gradient issues detected"




def count_model_params(model, trainable_only=False, verbose=False):
    """
    Count the number of parameters in a PyTorch model, distinguishing between system native and custom modules.
    
    Args:
    model (nn.Module): The PyTorch model to analyze.
    trainable_only (bool): If True, count only trainable parameters. Default is False.
    verbose (bool): If True, print detailed breakdown of parameters. Default is False.
    
    Returns:
    float: Total number of (trainable) parameters in millions.
    dict: Breakdown of parameters by layer type.
    """
    total_params = 0
    trainable_params = 0
    param_counts = defaultdict(int)
    
    # List of PyTorch native modules
    native_modules = set([name for name, obj in nn.__dict__.items() if isinstance(obj, type)])

    for name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
            
            # Count parameters for each layer type
            layer_type = module.__class__.__name__
            if layer_type in native_modules:
                layer_type = f"Native_{layer_type}"
            else:
                layer_type = f"Custom_{layer_type}"
            param_counts[layer_type] += param.numel()
    
    if verbose:
     
        
        native_counts = {k: v for k, v in param_counts.items() if k.startswith("Native_")}
        custom_counts = {k: v for k, v in param_counts.items() if k.startswith("Custom_")}
        
        native_total = sum(native_counts.values())
        custom_total = sum(custom_counts.values())
        
        print("-" * 55)
        print(f"{'☕ Native Modules Total':<30} {native_total:<15,d} {native_total/total_params*100:.2f}%")
        print("-" * 55)
        
        # Print native modules
        for i, (layer_type, count) in enumerate(sorted(native_counts.items(), key=lambda x: x[1], reverse=True), 1):
            percentage = count / total_params * 100
            print(f"    {i}. ⅀ {layer_type[7:]:<23} {count:<15,d} {percentage:.2f}%")
        
        print("-" * 55)
        print(f"{'🍄 Custom Modules Total':<30} {custom_total:<15,d} {custom_total/total_params*100:.2f}%")
        print("-" * 55)
        # Print custom modules
        for i, (layer_type, count) in enumerate(sorted(custom_counts.items(), key=lambda x: x[1], reverse=True), 1):
            percentage = count / total_params * 100
            print(f"   {i}. {layer_type[7:]:<23} {count:<15,d} {percentage:.2f}%")
        
        print(f"{'Layer Type':<30} {'Parameter Count':<15} {'% of Total':<10}")
        print("-" * 55)
        print(f"{'Total':<30} {total_params:<15,d} 100.00%")
        print(f"{'Trainable':<30} {trainable_params:<15,d} {trainable_params/total_params*100:.2f}%")
    
    if trainable_only:
        return trainable_params / 1e6, dict(param_counts)
    else:
        return total_params / 1e6, dict(param_counts)
    
def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def sample_recon(model, data, accelerator, output_path, num_samples=1):
    model.eval()
    with torch.no_grad():
        try:
            x_reconstructed,x_current, x_reference = data
            batch_size = x_reconstructed.size(0)
            num_samples = min(num_samples, batch_size)
            
            # Select a subset of images if batch_size > num_samples
            x_reconstructed = x_reconstructed[:num_samples]
            x_reference = x_reference[:num_samples]
            x_current = x_current[:num_samples]
            
            # Prepare frames for saving (2 rows: clamped reconstructed and original reference)
            frames = torch.cat((x_reconstructed,x_current, x_reference), dim=0)
            
            # Ensure we have a valid output directory
            if output_path:
                output_dir = os.path.dirname(output_path)
                if not output_dir:
                    output_dir = '.'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save frames as a grid (2 rows, num_samples columns)
                save_image(accelerator.gather(frames), output_path, nrow=num_samples, padding=2, normalize=False)
                # accelerator.print(f"Saved sample reconstructions to {output_path}")
            else:
                accelerator.print("Warning: No output path provided. Skipping image save.")
            
            # Log images to wandb
            wandb_images = []
            for i in range(num_samples):
                wandb_images.extend([
                    wandb.Image(x_reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reconstructed {i}"),
                    wandb.Image(x_current[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_current {i}"),
                    wandb.Image(x_reference[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reference {i}")
                ])
            
            wandb.log({"Sample Reconstructions": wandb_images})
            
            return frames
        except RuntimeError as e:
            print(f"🔥 e:{e}")
        return None
                       

def monitor_gradients(model, epoch, batch_idx, log_interval=10):
    """
    Monitor gradients of the model parameters.
    
    :param model: The neural network model
    :param epoch: Current epoch number
    :param batch_idx: Current batch index
    :param log_interval: How often to log gradient statistics
    """
    if batch_idx % log_interval == 0:
        grad_stats = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats['norm'].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                
                grad_stats['names'].append(name)
        
        if grad_stats['norm']:
            avg_norm = np.mean(grad_stats['norm'])
            max_norm = np.max(grad_stats['norm'])
            min_norm = np.min(grad_stats['norm'])
            
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"Gradient norms - Avg: {avg_norm:.4f}, Max: {max_norm:.4f}, Min: {min_norm:.4f}")
            
            # Identify layers with unusually high or low gradients
            threshold_high = avg_norm * 10  # Adjust this multiplier as needed
            threshold_low = avg_norm * 0.1  # Adjust this multiplier as needed
            
            for name, norm in zip(grad_stats['names'], grad_stats['norm']):
                if norm > threshold_high:
                    print(f"High gradient in {name}: {norm:.4f}")
                elif norm < threshold_low:
                    print(f"Low gradient in {name}: {norm:.4f}")
        else:
            print("No gradients to monitor")



# helper for gradient vanishing / explosion
def hook_fn(name):
    def hook(grad):
        if torch.isnan(grad).any():
            # print(f"🔥 NaN gradient detected in {name}")
            return torch.zeros_like(grad)  # Replace NaN with zero
        elif torch.isinf(grad).any():
            # print(f"🔥 Inf gradient detected in {name}")
            return torch.clamp(grad, -1e6, 1e6)  # Clamp infinite values
        #else:
            # You can add more conditions or logging here
         #  grad_norm = grad.norm().item()
         #   print(f"Gradient norm for {name}: {grad_norm}")
        return grad
    return hook

def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))




def visualize_latent_token(token, save_path):
    """
    Visualize a 1D latent token as a colorful bar.
    
    Args:
    token (torch.Tensor): A 1D tensor representing the latent token.
    save_path (str): Path to save the visualization.
    """
    # Ensure the token is on CPU and convert to numpy
    token_np = token.cpu().detach().numpy()
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 0.5))
    
    # Normalize the token values to [0, 1] for colormap
    token_normalized = (token_np - token_np.min()) / (token_np.max() - token_np.min())
    
    # Create a colorful representation
    cmap = plt.get_cmap('viridis')
    colors = cmap(token_normalized)
    
    # Plot the token as a colorful bar
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a title
    plt.title(f"Latent Token (dim={len(token_np)})")
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()