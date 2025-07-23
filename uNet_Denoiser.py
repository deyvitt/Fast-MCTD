#___________________________________________________________________________________________________

# uNet_Denoiser.py (sample code)
import math
from typing import Optional, List

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class AttentionBlock(nn.Module):
    """Self-attention for capturing long-range dependencies"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        # Attention computation
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhdi,bhdj->bhij', q, k) * scale, dim=-1)
        h = torch.einsum('bhij,bhdj->bhdi', attn, v)
        h = h.reshape(B, C, H, W)
        
        return x + self.proj_out(h)

class UNetBlock(nn.Module):
    """UNet residual block with time conditioning"""
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, 
                 use_attention: bool = False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) \
                              if in_channels != out_channels else nn.Identity()
                              
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        
        # Add time conditioning
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj
        
        h = F.silu(self.norm2(self.conv2(h)))
        h = self.attention(h)
        
        return h + self.skip_connection(x)

class AdvancedUNet(nn.Module):
    """Enhanced UNet architecture for diffusion denoising"""
    def __init__(self, 
                 input_channels: int = 128,  # VAE latent dimension
                 model_channels: int = 256,
                 out_channels: int = 128,
                 num_res_blocks: int = 2,
                 attention_resolutions: List[int] = [16, 8],
                 channel_mult: List[int] = [1, 2, 4, 8],
                 time_embedding_dim: int = 512):
        super().__init__()
        
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Input projection (treat latent as spatial for processing)
        self.input_proj = nn.Conv2d(1, model_channels, 3, padding=1)
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                use_attn = (32 // (2 ** level)) in attention_resolutions
                self.down_blocks.append(
                    UNetBlock(input_ch, out_ch, time_embedding_dim, use_attn)
                )
                input_ch = out_ch
                
            if level < len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
        
        # Middle block
        self.middle_block = nn.Sequential(
            UNetBlock(input_ch, input_ch, time_embedding_dim, True),
            UNetBlock(input_ch, input_ch, time_embedding_dim, True)
        )
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                use_attn = (32 // (2 ** (len(channel_mult) - 1 - level))) in attention_resolutions
                self.up_blocks.append(
                    UNetBlock(input_ch + (out_ch if i == 0 else 0), out_ch, 
                             time_embedding_dim, use_attn)
                )
                input_ch = out_ch
                
            if level < len(channel_mult) - 1:
                self.up_blocks.append(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
        
        # Output projection
        self.out_norm = nn.GroupNorm(8, model_channels)
        self.out_conv = nn.Conv2d(model_channels, 1, 3, padding=1)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy latent vectors [B, latent_dim]
            timesteps: Diffusion timesteps [B]
        Returns:
            Predicted noise [B, latent_dim]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Reshape latent to spatial format for UNet processing
        B, D = x.shape
        H = W = int(math.sqrt(D))  # Assume square spatial arrangement
        if H * W != D:
            H = W = 16  # Default spatial size
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), H * W).squeeze(1)
        
        h = x.view(B, 1, H, W)  # [B, 1, H, W]
        h = self.input_proj(h)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for block in self.down_blocks:
            if isinstance(block, UNetBlock):
                h = block(h, time_emb)
                skip_connections.append(h)
            else:  # Downsampling
                h = block(h)
                
        # Middle
        h = self.middle_block[0](h, time_emb)
        h = self.middle_block[1](h, time_emb)
        
        # Decoder
        for block in self.up_blocks:
            if isinstance(block, UNetBlock):
                if skip_connections:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
            else:  # Upsampling
                h = block(h)
        
        # Output
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        
        # Reshape back to latent format
        h = h.view(B, -1)  # [B, H*W]
        if h.shape[1] != x.shape[1]:
            h = F.adaptive_avg_pool1d(h.unsqueeze(1), x.shape[1]).squeeze(1)
            
        return h

  #_______________________________________________________________________________________________________________
