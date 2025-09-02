import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from logger import logger
import traceback
from typing import *

class ParallelizedNeuralMemory(nn.Module):
    def __init__(
        self, 
        dim: int,
        depth: int = 2,
        memory_size: Optional[int] = None,
        batch_size: int = 32
    ):
        super().__init__()
        logger.info("\n=== Initializing Neural Memory Module ===")
        logger.info(f"Input dimension: {dim}")
        logger.info(f"Depth: {depth}")
        logger.info(f"Memory size: {memory_size or dim}")
        
        self.dim = dim
        self.depth = depth
        self.memory_size = memory_size or dim
        
        # Memory layers with improved initialization
        self.layers = nn.ModuleList([])
        for i in range(depth):
            input_dim = dim if i == 0 else self.memory_size
            output_dim = dim if i == depth-1 else self.memory_size
            
            logger.info(f"\nLayer {i}:")
            logger.info(f"Input dimension: {input_dim}")
            logger.info(f"Output dimension: {output_dim}")
            
            layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU() if i < depth-1 else nn.Identity(),
                nn.Dropout(0.1)
            )
            
            nn.init.xavier_uniform_(layer[0].weight, gain=0.1)
            nn.init.zeros_(layer[0].bias)
            
            self.layers.append(layer)
            logger.info(f"Layer weights shape: {layer[0].weight.shape}")
        
        # Learning parameters
        self.register_parameter('theta', nn.Parameter(torch.ones(1) * 0.01))
        self.register_parameter('eta', nn.Parameter(torch.ones(1) * 0.99))
        self.register_parameter('alpha', nn.Parameter(torch.ones(1) * 0.01))
        
        # Initialize buffers with correct shapes
        self.register_buffer('momentum', torch.zeros(1, self.dim))
        self.register_buffer('grad_acc', torch.zeros(1, self.dim))

    def _expand_batch_params(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand parameters to match batch size."""
        return (
            self.theta.expand(batch_size, 1),
            self.eta.expand(batch_size, 1),
            self.alpha.expand(batch_size, 1)
        )

    def parallel_update(self, x: torch.Tensor) -> torch.Tensor:
        try:
            B, N, D = x.shape
            logger.info(f"\n=== Starting parallel_update ===")
            logger.info(f"Input shape: {x.shape}")
            
            # Ensure input tensor is contiguous
            x = x.contiguous()
            
            # Expand parameters
            theta_batch, eta_batch, alpha_batch = self._expand_batch_params(B)
            
            # Reshape for parallel processing
            x_flat = x.reshape(-1, D)
            logger.info(f"Reshaped input shape: {x_flat.shape}")
            
            # Forward pass through memory layers
            h = x_flat
            residuals = []
            
            for i, layer in enumerate(self.layers):
                logger.info(f"Layer {i} input shape: {h.shape}")
                if i == 0:
                    residuals.append(h)
                h = layer(h)
                logger.info(f"Layer {i} output shape: {h.shape}")
                if i > 0 and i % 2 == 0 and residuals:
                    h = h + residuals.pop()
            
            # Compute loss with a small regularization term to ensure all parameters are used
            loss = 0.5 * ((h - x_flat) ** 2).mean()
            # Add small regularization for learning parameters
            loss = loss + 1e-6 * (self.theta.sum() + self.eta.sum() + self.alpha.sum())
            logger.info(f"Loss value: {loss.item()}")
            
            # Get gradients with allow_unused=True
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                loss, 
                trainable_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            
            # Process gradients
            if grads:
                logger.info("\nProcessing gradients...")
                
                # Reset gradient accumulator
                self.grad_acc.zero_()
                
                # Process each gradient
                for param, grad in zip(trainable_params, grads):
                    if grad is not None:
                        # Ensure gradient tensor is contiguous
                        grad = grad.contiguous()
                        
                        # Handle different parameter shapes
                        if grad.dim() == 1:  # For bias terms
                            grad = grad.unsqueeze(0)
                        elif grad.dim() > 2:  # For layer weights
                            grad = grad.reshape(B * N, -1)
                        
                        # Average gradients
                        grad_mean = grad.mean(0, keepdim=True)
                        
                        # Update parameter-specific momentum
                        param_momentum = self.momentum[:, :grad_mean.size(1)]
                        param_momentum = eta_batch.mean(0) * param_momentum - theta_batch.mean(0) * grad_mean
                        
                        # Update parameter
                        with torch.no_grad():
                            forget_factor = 1 - alpha_batch.mean(0)
                            param.data.mul_(forget_factor).add_(param_momentum.reshape(param.shape))
            
            # Reshape output back to original dimensions
            output = h.reshape(B, N, D)
            logger.info(f"Final output shape: {output.shape}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in parallel_update: {str(e)}")
            raise

    def forward(self, x: torch.Tensor, update_weights: bool = True) -> torch.Tensor:
        try:
            logger.info(f"\n=== Memory Forward Pass ===")
            logger.info(f"Input shape: {x.shape}")
            
            if len(x.shape) != 3:
                raise ValueError(f"Expected 3D input (batch, seq, dim), got shape {x.shape}")
            
            if x.shape[-1] != self.dim:
                raise ValueError(f"Input dimension {x.shape[-1]} doesn't match expected {self.dim}")
            
            # Set to training mode during updates
            if update_weights:
                self.train()
            else:
                self.eval()
                
            if self.training and update_weights:
                return self.parallel_update(x)
            
            # Standard inference pass
            h = x
            for layer in self.layers:
                h = layer(h)
            return h
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
   



class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, 
                                  padding=kernel_size//2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        
    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise(x)
        x = self.pointwise(x)
        return rearrange(x, 'b d n -> b n d')

class EnhancedSelfAttention(nn.Module):
    """Enhanced self-attention with convolution and improved efficiency."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Convolution for local context
        self.conv_q = DepthwiseSeparableConv1d(dim)
        self.conv_k = DepthwiseSeparableConv1d(dim)
        self.conv_v = DepthwiseSeparableConv1d(dim)
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # Apply convolutions
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        # Project and reshape
        q = rearrange(self.q_proj(q), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(k), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(v), 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.out_proj(out)

class FeedForward(nn.Module):
    """Enhanced feed-forward network with SwiGLU activation."""
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.w1 = nn.Linear(dim, dim * expansion_factor)
        self.w2 = nn.Linear(dim, dim * expansion_factor)
        self.w3 = nn.Linear(dim * expansion_factor, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)

class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with improved normalization and gating."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attention = EnhancedSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)
        
        # Output gating
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU()
        )
        
    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out * self.gate(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out * self.gate(ffn_out)
        
        return x

class TitansMAC(nn.Module):
    """Enhanced Titans MAC architecture."""
    def __init__(
        self,
        dim,
        num_layers=12,
        num_heads=8,
        memory_depth=2,
        memory_size=None,
        persistent_tokens=16,
        max_seq_len=4096,
        chunk_size=512,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.persistent_tokens = persistent_tokens
        
        # Persistent memory with improved initialization
        self.persistent_mem = nn.Parameter(
            torch.randn(persistent_tokens, dim) / math.sqrt(dim)
        )
        
        # Enhanced memory module
        self.memory = ParallelizedNeuralMemory(
            dim=dim,
            depth=memory_depth,
            memory_size=memory_size
        )
        
        # Enhanced transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Enhanced projections
        self.input_proj = nn.Sequential(
            DepthwiseSeparableConv1d(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Rotary position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(max_seq_len + persistent_tokens, dim) / math.sqrt(dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def create_attention_mask(self, seq_len):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Process in chunks with memory
        chunk_outputs = []
        memory_state = None
        
        for i in range(0, N, self.chunk_size):
            chunk = x[:, i:min(i+self.chunk_size, N)]
            
            # Get memory output
            if memory_state is None:
                memory_out = self.memory(chunk)
            else:
                memory_out = self.memory(chunk, memory_state)
            
            # Combine context
            persistent = repeat(self.persistent_mem, 'n d -> b n d', b=B)
            context = torch.cat([persistent, memory_out, chunk], dim=1)
            
            # Add position embeddings
            pos_ids = torch.arange(context.size(1), device=x.device)
            context = context + self.pos_embed[pos_ids]
            context = self.dropout(context)
            
            # Create attention mask
            mask = self.create_attention_mask(context.size(1))
            
            # Process through transformer layers
            for layer in self.layers:
                context = layer(context)
            
            # Update memory state
            memory_state = context[:, self.persistent_tokens:].detach()
            
            # Extract chunk output
            chunk_out = context[:, -chunk.size(1):]
            chunk_outputs.append(chunk_out)
        
        # Combine chunks
        out = torch.cat(chunk_outputs, dim=1)
        
        return self.output_proj(out)

# Example usage
if __name__ == "__main__":
    # Model configuration
    config = {
        'dim': 512,
        'num_layers': 12,
        'num_heads': 8,
        'memory_depth': 2,
        'persistent_tokens': 16,
        'max_seq_len': 4096,
        'chunk_size': 512,
        'dropout': 0.1
    }
    
    # Create model
    model = TitansMAC(**config)
    
    # Example input
    x = torch.randn(2, 1024, 512)  # batch_size=2, seq_len=1024, dim=512
    
    # Forward pass
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be same as input shape