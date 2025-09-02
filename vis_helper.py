import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import glob 
import re

def save_expression_embed(expression_embed, frame_idx, name, save_dir='expression_comparison'):
    """
    Save expression embedding tensor to disk for comparison.
    
    Args:
        expression_embed (torch.Tensor): Expression embedding tensor
        frame_idx (int): Frame index
        name (str): Identifier (e.g., 'vasa' or 'wrapper')
        save_dir (str): Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure we're working with CPU tensor
    if isinstance(expression_embed, torch.Tensor):
        expression_embed = expression_embed.cpu().detach()
    
    # Save raw tensor
    torch.save(expression_embed, 
              os.path.join(save_dir, f'expression_{name}_frame_{frame_idx}.pt'))

def visualize_expression_comparison(frame_idx, save_dir='expression_comparison'):
    """
    Create visualization comparing VASA and InferenceWrapper expression embeddings.
    
    Args:
        frame_idx (int): Frame index to compare
        save_dir (str): Directory containing saved embeddings
    """
    # Load saved tensors
    vasa_path = os.path.join(save_dir, f'expression_vasa_frame_{frame_idx}.pt')
    wrapper_path = os.path.join(save_dir, f'expression_wrapper_target_frame_{frame_idx}.pt')
    
    if not (os.path.exists(vasa_path) and os.path.exists(wrapper_path)):
        print(f"Missing expression embeddings for frame {frame_idx}")
        return
        
    vasa_embed = torch.load(vasa_path)
    wrapper_embed = torch.load(wrapper_path)
    
    # Convert to numpy for visualization
    vasa_np = vasa_embed.numpy()
    wrapper_np = wrapper_embed.numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    
    # Plot VASA embedding
    im1 = ax1.imshow(vasa_np.reshape(1, -1), aspect='auto', cmap='viridis')
    ax1.set_title(f'VASA Expression Embedding (Frame {frame_idx})')
    plt.colorbar(im1, ax=ax1)
    ax1.set_yticks([])
    
    # Plot Wrapper embedding
    im2 = ax2.imshow(wrapper_np.reshape(1, -1), aspect='auto', cmap='viridis')
    ax2.set_title(f'InferenceWrapper Expression Embedding (Frame {frame_idx})')
    plt.colorbar(im2, ax=ax2)
    ax2.set_yticks([])
    
    # Plot difference
    diff = vasa_np - wrapper_np
    im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto', cmap='RdBu', norm=plt.Normalize(vmin=-np.abs(diff).max(), vmax=np.abs(diff).max()))
    ax3.set_title('Difference (VASA - Wrapper)')
    plt.colorbar(im3, ax=ax3)
    ax3.set_yticks([])
    
    # Add statistics as text
    stats_text = f'Max Diff: {np.abs(diff).max():.4f}\n'
    stats_text += f'Mean Diff: {np.abs(diff).mean():.4f}\n'
    stats_text += f'Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_frame_{frame_idx}.png'))
    plt.close()

def analyze_expressions(frame_idx, save_dir='expression_comparison'):
    """
    Analyze and print detailed statistics about the expression embeddings.
    
    Args:
        frame_idx (int): Frame index to analyze
        save_dir (str): Directory containing saved embeddings
    """
    # Load saved tensors
    vasa_path = os.path.join(save_dir, f'expression_vasa_frame_{frame_idx}.pt')
    wrapper_path = os.path.join(save_dir, f'expression_wrapper_target_frame_{frame_idx}.pt')
    
    vasa_embed = torch.load(vasa_path)
    wrapper_embed = torch.load(wrapper_path)
    
    # Convert to numpy
    vasa_np = vasa_embed.numpy()
    wrapper_np = wrapper_embed.numpy()
    
    # Compute statistics
    print(f"\nExpression Embedding Analysis for Frame {frame_idx}")
    print("-" * 50)
    print(f"VASA Embedding:")
    print(f"  Shape: {vasa_np.shape}")
    print(f"  Mean: {vasa_np.mean():.4f}")
    print(f"  Std: {vasa_np.std():.4f}")
    print(f"  Min: {vasa_np.min():.4f}")
    print(f"  Max: {vasa_np.max():.4f}")
    
    print(f"\nWrapper Embedding:")
    print(f"  Shape: {wrapper_np.shape}")
    print(f"  Mean: {wrapper_np.mean():.4f}")
    print(f"  Std: {wrapper_np.std():.4f}")
    print(f"  Min: {wrapper_np.min():.4f}")
    print(f"  Max: {wrapper_np.max():.4f}")
    
    diff = vasa_np - wrapper_np
    print(f"\nDifferences:")
    print(f"  Max Absolute Diff: {np.abs(diff).max():.4f}")
    print(f"  Mean Absolute Diff: {np.abs(diff).mean():.4f}")
    print(f"  Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}")
    
    # Save detailed analysis
    with open(os.path.join(save_dir, f'analysis_frame_{frame_idx}.txt'), 'w') as f:
        f.write(f"Expression Embedding Analysis for Frame {frame_idx}\n")
        f.write("-" * 50 + "\n")
        f.write(f"VASA Embedding Statistics:\n")
        f.write(f"  Shape: {vasa_np.shape}\n")
        f.write(f"  Mean: {vasa_np.mean():.4f}\n")
        f.write(f"  Std: {vasa_np.std():.4f}\n")
        f.write(f"  Min: {vasa_np.min():.4f}\n")
        f.write(f"  Max: {vasa_np.max():.4f}\n\n")
        
        f.write(f"Wrapper Embedding Statistics:\n")
        f.write(f"  Shape: {wrapper_np.shape}\n")
        f.write(f"  Mean: {wrapper_np.mean():.4f}\n")
        f.write(f"  Std: {wrapper_np.std():.4f}\n")
        f.write(f"  Min: {wrapper_np.min():.4f}\n")
        f.write(f"  Max: {wrapper_np.max():.4f}\n\n")
        
        f.write(f"Differences:\n")
        f.write(f"  Max Absolute Diff: {np.abs(diff).max():.4f}\n")
        f.write(f"  Mean Absolute Diff: {np.abs(diff).mean():.4f}\n")
        f.write(f"  Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}\n")

def analyze_all_expressions(save_dir='expression_comparison'):
    """
    Find all saved expression embeddings and analyze them.
    """
    # Check if directory exists
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found!")
        return
    
    # Find all VASA expression files
    vasa_files = glob.glob(os.path.join(save_dir, 'expression_vasa_frame_*.pt'))
    
    # Extract frame numbers
    frame_numbers = []
    for file in vasa_files:
        match = re.search(r'frame_(\d+)\.pt', file)
        if match:
            frame_numbers.append(int(match.group(1)))
    
    frame_numbers = sorted(frame_numbers)
    print(f"Found {len(frame_numbers)} frames to analyze")
    
    # Process each frame
    for frame_idx in frame_numbers:
        print(f"\nProcessing frame {frame_idx}")
        print("-" * 50)
        try:
            visualize_expression_comparison(frame_idx)
            analyze_expressions(frame_idx)
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

# Run the analysis
if __name__ == "__main__":
    analyze_all_expressions()

#     visualize_expression_comparison(0)
#     analyze_expressions(0) 
