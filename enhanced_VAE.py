#_______________________________________________________________________________________________
# enhanced_VAE.py (sample code)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)

class VAEEncoder(nn.Module):
    """Enhanced VAE encoder with residual connections"""
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoding pathway
        self.conv_in = nn.Conv2d(input_channels, 64, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),
                ResidualBlock(128),
                ResidualBlock(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1),
                ResidualBlock(256),
                ResidualBlock(256)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1),
                ResidualBlock(512),
                ResidualBlock(512)
            )
        ])
        
        # Latent projection
        self.conv_norm_out = nn.GroupNorm(8, 512)
        self.conv_out = nn.Conv2d(512, 2 * latent_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.conv_in(x))
        
        # Downsampling
        for block in self.down_blocks:
            h = block(h)
            
        # Final projection
        h = F.silu(self.conv_norm_out(h))
        h = self.conv_out(h)
        
        # Global average pooling and split
        h = h.mean(dim=[2, 3])  # [B, 2*latent_dim]
        mean, logvar = h.chunk(2, dim=1)
        
        return mean, logvar

class VAEDecoder(nn.Module):
    """Enhanced VAE decoder with attention mechanisms"""
    def __init__(self, latent_dim: int = 128, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        
        # Initial projection
        self.conv_in = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(512),
                ResidualBlock(512),
                nn.ConvTranspose2d(512, 256, 4, 2, 1)
            ),
            nn.Sequential(
                ResidualBlock(256),
                ResidualBlock(256),
                nn.ConvTranspose2d(256, 128, 4, 2, 1)
            ),
            nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                nn.ConvTranspose2d(128, 64, 4, 2, 1)
            )
        ])
        
        # Output projection
        self.conv_norm_out = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, output_channels, 3, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project and reshape
        h = self.conv_in(z)  # [B, 512*4*4]
        h = h.view(h.shape[0], 512, 4, 4)  # [B, 512, 4, 4]
        
        # Upsampling
        for block in self.up_blocks:
            h = block(h)
            
        # Final output
        h = F.silu(self.conv_norm_out(h))
        h = torch.sigmoid(self.conv_out(h))
        
        return h

class EnhancedVAE(nn.Module):
    """Complete VAE with improved architecture"""
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = VAEEncoder(input_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_channels)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
#_______________________________________________________________________________________________________
