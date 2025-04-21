# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# import numpy as np
# import torch.nn.functional as F

# import wandb
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import itertools
# import math



# def lambda_init_fn(depth):
#     return 0.8 - 0.6 * math.exp(-0.3 * depth)


# class PatchEmbedding(nn.Module):
#   def __init__(self, d_model, img_size, patch_size, n_channels):
#     super().__init__()

#     self.d_model = d_model # Dimensionality of Model
#     self.img_size = img_size # Image Size
#     self.patch_size = patch_size # Patch Size
#     self.n_channels = n_channels # Number of Channels

#     self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

#   # B: Batch Size
#   # C: Image Channels
#   # H: Image Height
#   # W: Image Width
#   # P_col: Patch Column
#   # P_row: Patch Row
#   def forward(self, x):
#     x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

#     x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

#     x = x.transpose(-2, -1) # (B, d_model, P) -> (B, P, d_model)

#     return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length, pos_type='sinusoidal', 
#                  img_size=None, patch_size=None):
#         super().__init__()
#         self.pos_type = pos_type
#         self.d_model = d_model
#         self.max_seq_length = max_seq_length
        
#         # CLS token (present in all configurations)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
#         if self.pos_type == '1d_learned':
#             self.pos_embed = nn.Parameter(torch.randn(1, max_seq_length, d_model))
#         elif self.pos_type == '2d_learned':
#             assert img_size and patch_size, "Need img_size and patch_size for 2D"
#             self.h_patches = img_size[0] // patch_size[0]
#             self.w_patches = img_size[1] // patch_size[1]
#             self.row_embed = nn.Embedding(self.h_patches, d_model)
#             self.col_embed = nn.Embedding(self.w_patches, d_model)
#             # Separate learnable positional embedding for CLS token
#             self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, d_model))
#         elif self.pos_type == 'sinusoidal':
#             pe = torch.zeros(max_seq_length, d_model)
#             position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#             div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#             pe[:, 0::2] = torch.sin(position * div_term)
#             pe[:, 1::2] = torch.cos(position * div_term)
#             self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         B = x.size(0)  # Batch size
        
#         # Add CLS token to all configurations
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)
        
#         if self.pos_type == '1d_learned':
#             x = x + self.pos_embed
#         elif self.pos_type == '2d_learned':
#             # Generate grid positions
#             rows = torch.arange(self.h_patches, device=x.device).view(-1,1).expand(-1, self.w_patches)
#             cols = torch.arange(self.w_patches, device=x.device).view(1,-1).expand(self.h_patches, -1)
            
#             # Convert to 1D sequences
#             rows = rows.reshape(-1)
#             cols = cols.reshape(-1)
            
#             # Get 2D positional embeddings (n_patches, d_model)
#             pos_emb = self.row_embed(rows) + self.col_embed(cols)
            
#             # Add CLS positional embedding and expand to batch size
#             cls_pos = self.cls_pos_embed.expand(B, -1, -1)  # (B, 1, d_model)
#             patch_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, n_patches, d_model)
            
#             # Combine CLS and patch positions
#             pos_emb = torch.cat([cls_pos, patch_pos], dim=1)
#             x = x + pos_emb
#         elif self.pos_type == 'sinusoidal':
#             x = x + self.pe

#         return x




# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.zeros(dim))

#     def forward(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

# class DifferentialAttention(nn.Module):
#     def __init__(self, dim_model: int, head_nums: int, depth: int):
#         super().__init__()
        
#         self.head_dim = dim_model // head_nums

#         self.Q = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
#         self.K = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
#         self.V = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
#         self.scale = self.head_dim ** -0.5
#         self.depth = depth
#         self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    
#     def forward(self, x):
#         lambda_init = lambda_init_fn(self.depth)
#         Q = self.Q(x)
#         K = self.K(x)

    
#         Q1, Q2 = Q.chunk(2, dim=-1)
#         K1, K2 = K.chunk(2, dim=-1)
#         V = self.V(x)
#         A1 = Q1 @ K1.transpose(-2, -1) * self.scale
#         A2 = Q2 @ K2.transpose(-2, -1) * self.scale
#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q1)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q2)
#         lambda_ = lambda_1 - lambda_2 + lambda_init
#         return (F.softmax(A1, dim=-1)  - lambda_ * F.softmax(A2, dim=-1)) @ V





# class MultiHeadDifferentialAttention(nn.Module):
#     def __init__(self, dim_model: int, head_nums: int, depth: int):
#         super().__init__()
#         self.heads = nn.ModuleList([DifferentialAttention(dim_model, head_nums, depth) for _ in range(head_nums)])
#         self.group_norm = RMSNorm(dim_model)
#         self.output = nn.Linear(2 * dim_model, dim_model, bias=False)
#         self.lambda_init = lambda_init_fn(depth)
    
#     def forward(self, x):
#         o = torch.cat([self.group_norm(h(x)) for h in self.heads], dim=-1)
#         o = o * (1 - self.lambda_init)
#         return self.output(o)


# class TransformerEncoder(nn.Module):
#   def __init__(self, d_model, n_heads,depths, r_mlp=4):
#     super().__init__()
#     self.d_model = d_model
#     self.n_heads = n_heads

#     # Sub-Layer 1 Normalization
#     self.ln1 = nn.LayerNorm(d_model)

#     # Multi-Head Attention
#     self.mha = MultiHeadDifferentialAttention(d_model, n_heads, depths)

#     # Sub-Layer 2 Normalization
#     self.ln2 = nn.LayerNorm(d_model)

#     # Multilayer Perception
#     self.mlp = nn.Sequential(
#         nn.Linear(d_model, d_model*r_mlp),
#         nn.GELU(),
#         nn.Linear(d_model*r_mlp, d_model)
#     )

#   def forward(self, x):
#     # Residual Connection After Sub-Layer 1
#     out = x + self.mha(self.ln1(x))

#     # Residual Connection After Sub-Layer 2
#     out = out + self.mlp(self.ln2(out))

#     return out


# class VisionTransformer(nn.Module):
#   def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, depths, pos_type = "sinusoidal"):
#     super().__init__()

#     assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
#     assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

#     self.d_model = d_model # Dimensionality of model
#     self.n_classes = n_classes # Number of classes
#     self.img_size = img_size # Image size
#     self.patch_size = patch_size # Patch size
#     self.n_channels = n_channels # Number of channels
#     self.n_heads = n_heads # Number of attention heads
#     self.depths = depths
#     self.pos_type = pos_type

#     self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
#     self.max_seq_length = self.n_patches + 1

#     self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
#     self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length, pos_type=self.pos_type, 
#                                                   img_size=self.img_size, patch_size= self.patch_size)
#     self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads, self.depths) for _ in range(n_layers)])

#     # Classification MLP
#     self.classifier = nn.Sequential(
#         nn.Linear(self.d_model, self.n_classes),
#         nn.Softmax(dim=-1)
#     )

#   def forward(self, images):
#     x = self.patch_embedding(images)

#     x = self.positional_encoding(x)

#     x = self.transformer_encoder(x)

#     x = self.classifier(x[:,0])

#     return x


import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, pos_type='sinusoidal', 
                 img_size=None, patch_size=None):
        super().__init__()
        self.pos_type = pos_type
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # CLS token (present in all configurations)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        if self.pos_type == '1d_learned':
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        elif self.pos_type == '2d_learned':
            assert img_size and patch_size, "Need img_size and patch_size for 2D"
            self.h_patches = img_size[0] // patch_size[0]
            self.w_patches = img_size[1] // patch_size[1]
            self.row_embed = nn.Embedding(self.h_patches, d_model)
            self.col_embed = nn.Embedding(self.w_patches, d_model)
            # Separate learnable positional embedding for CLS token
            self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, d_model))
        elif self.pos_type == 'sinusoidal':
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        B = x.size(0)  # Batch size
        
        # Add CLS token to all configurations
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        if self.pos_type == '1d_learned':
            x = x + self.pos_embed
        elif self.pos_type == '2d_learned':
            # Generate grid positions
            rows = torch.arange(self.h_patches, device=x.device).view(-1,1).expand(-1, self.w_patches)
            cols = torch.arange(self.w_patches, device=x.device).view(1,-1).expand(self.h_patches, -1)
            
            # Convert to 1D sequences
            rows = rows.reshape(-1)
            cols = cols.reshape(-1)
            
            # Get 2D positional embeddings (n_patches, d_model)
            pos_emb = self.row_embed(rows) + self.col_embed(cols)
            
            # Add CLS positional embedding and expand to batch size
            cls_pos = self.cls_pos_embed.expand(B, -1, -1)  # (B, 1, d_model)
            patch_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, n_patches, d_model)
            
            # Combine CLS and patch positions
            pos_emb = torch.cat([cls_pos, patch_pos], dim=1)
            x = x + pos_emb
        elif self.pos_type == 'sinusoidal':
            x = x + self.pe

        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(-2, -1) # (B, d_model, P) -> (B, P, d_model)

    return x

class DifferentialAttention(nn.Module):
    
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        
        self.head_dim = dim_model // head_nums

        self.Q = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.K = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.V = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.depth = depth
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))    # ... (existing code)
    
    def forward(self, x):
        lambda_init = lambda_init_fn(self.depth)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)
        
        A1 = Q1 @ K1.transpose(-2, -1) * self.scale
        A2 = Q2 @ K2.transpose(-2, -1) * self.scale
        
        attn1 = F.softmax(A1, dim=-1)
        attn2 = F.softmax(A2, dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(Q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(Q2)
        lambda_ = lambda_1 - lambda_2 + lambda_init
        
        combined_attn = attn1 - lambda_ * attn2
        output = combined_attn @ V
        return output, combined_attn

class MultiHeadDifferentialAttention(nn.Module):
    
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        self.heads = nn.ModuleList([DifferentialAttention(dim_model, head_nums, depth) for _ in range(head_nums)])
        self.group_norm = RMSNorm(dim_model)
        self.output = nn.Linear(2 * dim_model, dim_model, bias=False)
        self.lambda_init = lambda_init_fn(depth)
    
    def forward(self, x):
        head_outputs = []
        attn_maps = []
        for h in self.heads:
            h_out, attn = h(x)
            head_outputs.append(self.group_norm(h_out))
            attn_maps.append(attn)
        o = torch.cat(head_outputs, dim=-1)
        o = o * (1 - self.lambda_init)
        return self.output(o), attn_maps

class TransformerEncoder(nn.Module):
    
    def __init__(self, d_model, n_heads,depths, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadDifferentialAttention(d_model, n_heads, depths)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )
    
    def forward(self, x):
        attn_out, attn_maps = self.mha(self.ln1(x))
        out = x + attn_out
        out = out + self.mlp(self.ln2(out))
        return out, attn_maps

class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, depths, pos_type = "sinusoidal"):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model # Dimensionality of model
        self.n_classes = n_classes # Number of classes
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads
        self.depths = depths
        self.pos_type = pos_type

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length, pos_type=self.pos_type, 
                                                    img_size=self.img_size, patch_size= self.patch_size)
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, depths) for _ in range(n_layers)
        ])
        #Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )#
    
    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        all_attentions = []
        for encoder in self.transformer_encoder:
            x, attn_maps = encoder(x)
            all_attentions.append(attn_maps)
        x = self.classifier(x[:, 0])
        return x, all_attentions