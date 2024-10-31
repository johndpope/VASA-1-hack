import os
import torch


class Checkpoint():
    """
    Handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
        device: PyTorch device onto which loaded weights should be mapped
        kwargs: PyTorch modules whose state should be checkpointed
    """
    def __init__(self, checkpoint_dir='./chkpts', device=None, **kwargs):
        self.module_dict = kwargs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save(self, filename, **kwargs):
        """ Saves the current module states
        Args:
            filename (str): name of checkpoint file
            kwargs: Additional state to save
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            if k in outdict:
                print(f"Warning: Checkpoint key '{k}' overloaded. Defaulting to saving state_dict {v}.")
            if v is not None:
                outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        """Loads a checkpoint from file.
        Args:
            filename (str): Name of checkpoint file.
        Returns:
            Dictionary containing checkpoint data which does not correspond to registered modules.
        """

        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        print(f'Loading checkpoint from file {filename}...')
        state_dict = torch.load(filename, map_location=self.device)

        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print(f'Warning: Could not find "{k}" in checkpoint!')

        remaining_state = {k: v for k, v in state_dict.items()
                           if k not in self.module_dict}
        return remaining_state


from torch import nn

from srt.encoder import ImprovedFSRTEncoder
from srt.decoder import ImprovedFSRTDecoder
from srt.small_decoder import ImprovedFSRTDecoder as SmallImprovedFSRTDecoder

class FSRT(nn.Module):
    def __init__(self, cfg, expression_encoder=None):
        super().__init__()
            
        self.encoder = ImprovedFSRTEncoder(expression_size=cfg['expression_size'],  **cfg['encoder_kwargs'])
        
        if cfg['small_decoder']:
            self.decoder = SmallImprovedFSRTDecoder(expression_size=cfg['expression_size'], **cfg['decoder_kwargs'])
            print('Loading small decoder')
        else:
            self.decoder = ImprovedFSRTDecoder(expression_size=cfg['expression_size'], **cfg['decoder_kwargs'])
            
        self.expression_encoder = expression_encoder

import numpy as np
import torch
import torch.nn as nn

from srt.layers import  Transformer, FSRTPosEncoder


class FSRTPixelPredictor(nn.Module): 
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, out_dims=3,
                 z_dim=768, input_mlp=True, output_mlp=False, num_kp=10, expression_size=0, kp_octaves=4, kp_start_octave=-1):
        super().__init__()

        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)
        self.expression_size = expression_size
        self.num_kp = num_kp
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 720),
                nn.ReLU(),
                nn.Linear(720, self.feat_dim))
        else:
            self.input_mlp = None
        

        self.transformer = Transformer(self.feat_dim, depth=num_att_blocks, heads=6, dim_head=z_dim // 12,
                                       mlp_dim=z_dim, selfatt=False, kv_dim=z_dim)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, pixels, keypoints, expression_vector=None):
        """
        Args:
            z: set-latent scene repres. [batch_size, num_patches, patch_dim]
            pixels: query pixels [batch_size, num_pixels, 2]
            keypoints: facial query keypoints [batch_size, num_pixels, num_kp, 2]
            expression_vector: latent repres. of the query expression [batch_size, expression_size]
        """
        bs = pixels.shape[0]
        nr = pixels.shape[1]
        nkp = keypoints.shape[-2]
        queries = self.positional_encoder(pixels, keypoints.view(bs,nr,nkp*2))
        
        if expression_vector is not None:
            queries = torch.cat([queries,expression_vector[:,None].repeat(1,queries.shape[1],1)],dim=-1)

        if self.input_mlp is not None:
            queries = self.input_mlp(queries)

        output = self.transformer(queries, z)
        
        if self.output_mlp is not None:
            output = self.output_mlp(output)
            
        return output
    

class ImprovedFSRTDecoder(nn.Module):
    """ Scene Representation Transformer Decoder with the improvements from Appendix A.4 in the OSRT paper."""
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, num_kp=10, kp_octaves=4, kp_start_octave=-1, expression_size=0):
        super().__init__()
        self.allocation_transformer = FSRTPixelPredictor(num_att_blocks=num_att_blocks,
                                                   pix_start_octave=pix_start_octave,
                                                   pix_octaves=pix_octaves,
                                                   z_dim=768,
                                                   input_mlp=True,
                                                   output_mlp=False,
                                                   expression_size=expression_size,
                                                   kp_octaves=kp_octaves,
                                                   kp_start_octave = kp_start_octave
                                                )
        self.expression_size = expression_size 
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size
        self.render_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 1536),
            nn.ReLU(),
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Linear(768, 3),
        )

    def forward(self, z, x, pixels, expression_vector=None):
        x = self.allocation_transformer(z, x, pixels, expression_vector = expression_vector)
        pixels = self.render_mlp(x)
        return pixels, {}

 

import numpy as np
from matplotlib import pyplot as plt

from colorsys import hsv_to_rgb
from skimage.draw import disk

import matplotlib.pyplot as plt


def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_kp = kp_array.shape[0]
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = disk((kp[1], kp[0]), 3, shape=image.shape[:2])
        image[rr, cc] = np.array(plt.get_cmap('gist_rainbow')(kp_ind / num_kp))[:3]
    return image



def background_image(shape, gridsize=2, lg=0.85, dg=0.5):
    bg = np.zeros(shape)
    c1 = np.array((lg, lg, lg))
    c2 = np.array((dg, dg, dg))

    for i, x in enumerate(range(0, shape[0], gridsize)):
        for j, y in enumerate(range(0, shape[1], gridsize)):
            c = c1 if (i + j) % 2 == 0 else c2
            bg[x:x+gridsize, y:y+gridsize] = c

    return bg


def visualize_2d_cluster(clustering, colors=None):
    if colors is None:
        num_clusters = clustering.max()
        colors = get_clustering_colors(num_clusters)
    img = colors[clustering]
    return img


def get_clustering_colors(num_colors):
    colors = [(0., 0., 0.)]
    for i in range(num_colors):
        colors.append(hsv_to_rgb(i / num_colors, 0.45, 0.8))
    colors = np.array(colors)
    return colors


def setup_axis(axis):
    axis.tick_params(axis='both',       # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     right=False,
                     left=False,
                     labelbottom=False,
                     labelleft=False)   # labels along the bottom edge are off


def draw_visualization_grid(columns, outfile, row_labels=None, name=None):
    num_rows = columns[0][1].shape[0]
    num_cols = len(columns)
    num_segments = 1

    bg_image = None
    imshow_args = {'interpolation': 'none', 'cmap': 'gray'}

    for i in range(num_cols):
        column_type = columns[i][2]
        if column_type == 'clustering':
            num_segments = max(num_segments, columns[i][1].max())
        if column_type == 'image' and bg_image is None:
            bg_image = background_image(list(columns[i][1].shape[1:3]) + [3])

    colors = get_clustering_colors(num_segments)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for c in range(num_cols):
        axes[0, c].set_title(columns[c][0])
        col_type = columns[c][2]
        for r in range(num_rows):
            setup_axis(axes[r, c])
            img = columns[c][1][r]
            if col_type == 'image':
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                axes[r, c].imshow(bg_image, **imshow_args)
                axes[r, c].imshow(img, **imshow_args)
                if len(columns[c]) > 3:
                    axes[r, c].set_xlabel(columns[c][3][r])
            elif col_type == 'clustering':
                axes[r, c].imshow(visualize_2d_cluster(img, colors), **imshow_args)

    if row_labels is not None:
        for r in range(num_rows):
            axes[r, 0].set_ylabel(row_labels[r])

    plt.savefig(f'{outfile}.png')
    plt.close()



import numpy as np
import torch
import torch.nn as nn

from srt.layers import  Transformer, FSRTPosEncoder


class FSRTPixelPredictor(nn.Module): 
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, out_dims=3,
                 z_dim=768, input_mlp=True, output_mlp=False, num_kp=10, expression_size=0, kp_octaves=4, kp_start_octave=-1):
        super().__init__()

        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)
        self.expression_size = expression_size
        self.num_kp = num_kp
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 720),
                nn.ReLU(),
                nn.Linear(720, self.feat_dim))
        else:
            self.input_mlp = None
        

        self.transformer = Transformer(self.feat_dim, depth=num_att_blocks, heads=12, dim_head=z_dim // 12,
                                       mlp_dim=z_dim * 2, selfatt=False, kv_dim=z_dim)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, pixels, keypoints, expression_vector=None):
        """
        Args:
            z: set-latent scene repres. [batch_size, num_patches, patch_dim]
            pixels: query pixels [batch_size, num_pixels, 2]
            keypoints: facial query keypoints [batch_size, num_pixels, num_kp, 2]
            expression_vector: latent repres. of the query expression [batch_size, expression_size]
        """
        bs = pixels.shape[0]
        nr = pixels.shape[1]
        nkp = keypoints.shape[-2]
        queries = self.positional_encoder(pixels, keypoints.view(bs,nr,nkp*2))
        
        if expression_vector is not None:
            queries = torch.cat([queries,expression_vector[:,None].repeat(1,queries.shape[1],1)],dim=-1)

        if self.input_mlp is not None:
            queries = self.input_mlp(queries)

        output = self.transformer(queries, z)
        
        if self.output_mlp is not None:
            output = self.output_mlp(output)
            
        return output
    

class ImprovedFSRTDecoder(nn.Module):
    """ Scene Representation Transformer Decoder with the improvements from Appendix A.4 in the OSRT paper."""
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, num_kp=10, kp_octaves=4, kp_start_octave=-1, expression_size=0):
        super().__init__()
        self.allocation_transformer = FSRTPixelPredictor(num_att_blocks=num_att_blocks,
                                                   pix_start_octave=pix_start_octave,
                                                   pix_octaves=pix_octaves,
                                                   z_dim=768,
                                                   input_mlp=True,
                                                   output_mlp=False,
                                                   expression_size=expression_size,
                                                   kp_octaves=kp_octaves,
                                                   kp_start_octave = kp_start_octave
                                                )
        self.expression_size = expression_size 
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size
        self.render_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 3),
        )

    def forward(self, z, x, pixels, expression_vector=None):
        x = self.allocation_transformer(z, x, pixels, expression_vector = expression_vector)
        pixels = self.render_mlp(x)
        return pixels, {}






import torch
import torch.nn as nn
import numpy as np
import math

from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves, start_octave):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class FSRTPosEncoder(nn.Module):
    def __init__(self, kp_octaves=4, kp_start_octave=-1, pix_start_octave=-1, pix_octaves=16):
        super().__init__()
        self.kp_encoding = PositionalEncoding(num_octaves=kp_octaves, start_octave=kp_start_octave)
        self.pix_encoding = PositionalEncoding(num_octaves=pix_octaves, start_octave=pix_start_octave)
    def forward(self, pixels, kps=None):
        if len(pixels.shape) == 4:
            batchsize, height, width, _ = pixels.shape
            pixels = pixels.flatten(1, 2)
            pix_enc = self.pix_encoding(pixels)
            pix_enc = pix_enc.view(batchsize, height, width, pix_enc.shape[-1])
            pix_enc = pix_enc.permute((0, 3, 1, 2))
            
            if kps is not None:
                kp_enc = self.kp_encoding(kps.unsqueeze(1))
                kp_enc = kp_enc.view(batchsize, kp_enc.shape[-1], 1, 1).repeat(1, 1, height, width)
                x = torch.cat((kp_enc, pix_enc), 1)
        else:
            pix_enc = self.pix_encoding(pixels)
            
            if kps is not None:
                kp_enc = self.kp_encoding(kps)
                x = torch.cat((kp_enc, pix_enc), -1)

        return x



# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, kv_dim=kv_dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x


import numpy as np
import torch
import torch.nn as nn
import math

from srt.layers import Transformer, FSRTPosEncoder


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)

    
class ImprovedFSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """
    def __init__(self, num_conv_blocks=3, num_att_blocks=5, pix_octaves=16, pix_start_octave=-1, num_kp = 10, expression_size=256, encode_with_expression=True, kp_octaves=4, kp_start_octave=-1):
        super().__init__()
        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)

        self.encode_with_expression = encode_with_expression
        if self.encode_with_expression:
            self.expression_size = expression_size
        else:
            self.expression_size=0
        conv_blocks = [SRTConvBlock(idim=3+pix_octaves*4+num_kp*kp_octaves*4+self.expression_size, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=True) 
        self.num_kp = num_kp

    def forward(self, images, keypoints, pixels, expression_vector=None):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]
            keypoints: [batch_size, num_images, num_kp, 2]
            pixels: [batch_size, num_images, height, width, 2]
            expression_vector: [batch_size, num_images, expression_size]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        keypoints = keypoints.flatten(-2,-1).flatten(0,1)
        pixels = pixels.flatten(0, 1)

        pos_enc = self.positional_encoder(pixels,keypoints)
        if expression_vector is not None and self.encode_with_expression:
            expression_vector = expression_vector.flatten(0,1)[:,:,None,None].repeat(1,1,images.shape[-2],images.shape[-1])
            x = torch.cat([x,pos_enc,expression_vector], 1)
        else:
            x = torch.cat([x,pos_enc], 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x
    



