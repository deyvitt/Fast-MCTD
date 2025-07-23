#______________________________________________________________________________________________________________________

# diffusion_trainer.py
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from tqdm import tqdm

class DiffusionLoss:
    """Combined loss function for training"""
    
    def __init__(self, vae_weight: float = 1.0, 
                 diffusion_weight: float = 1.0,
                 mctd_weight: float = 0.5):
        self.vae_weight = vae_weight
        self.diffusion_weight = diffusion_weight
        self.mctd_weight = mctd_weight
    
    def vae_loss(self, x_recon: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reconstruction + KL divergence loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def diffusion_loss(self, noise_pred: torch.Tensor, 
                      noise_true: torch.Tensor) -> torch.Tensor:
        """L2 loss for noise prediction"""
        return F.mse_loss(noise_pred, noise_true)
    
    def mctd_policy_loss(self, tree_values: List[float], 
                        target_rewards: List[float]) -> torch.Tensor:
        """Policy improvement loss for MCTD"""
        if not tree_values or not target_rewards:
            return torch.tensor(0.0)
        
        tree_tensor = torch.tensor(tree_values)
        target_tensor = torch.tensor(target_rewards)
        
        return F.mse_loss(tree_tensor, target_tensor)

class ImageDataset(Dataset):
    """Simple image dataset for training"""
    
    def __init__(self, image_dir: str, image_size: int = 64):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

class FastMCTDTrainer:
    """Complete training pipeline for Fast-MCTD"""
    
    def __init__(self,
                 vae: EnhancedVAE,
                 unet: AdvancedUNet,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4):
        
        self.vae = vae
        self.unet = unet
        self.device = device
        
        # Move models to device
        self.vae.to(device)
        self.unet.to(device)
        
        # Optimizers
        self.vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
        self.unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = DiffusionLoss()
        
        # Noise schedule
        self.noise_schedule = create_noise_schedule(1000, 'cosine').to(device)
        
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.vae.train()
        self.unet.train()
        
        total_vae_loss = 0.0
        total_diffusion_loss = 0.0
        total_mctd_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            batch_size = batch.shape[0]
            
            # Phase 1: Train VAE
            self.vae_optimizer.zero_grad()
            
            x_recon, mu, logvar = self.vae(batch)
            vae_loss = self.loss_fn.vae_loss(x_recon, batch, mu, logvar)
            
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            self.vae_optimizer.step()
            
            # Phase 2: Train UNet on diffusion
            self.unet_optimizer.zero_grad()
            
            # Sample random timesteps
            timesteps = torch.randint(0, len(self.noise_schedule), (batch_size,), device=self.device)
            
            # Get clean latents
            with torch.no_grad():
                _, mu, _ = self.vae(batch)
                clean_latents = mu  # Use mean as clean latent
            
            # Add noise
            noise = torch.randn_like(clean_latents)
            alpha_cumprod = self.noise_schedule[timesteps]
            noisy_latents = (alpha_cumprod.sqrt().view(-1, 1) * clean_latents + 
                           (1 - alpha_cumprod).sqrt().view(-1, 1) * noise)
            
            # Predict noise
            noise_pred = self.unet(noisy_latents, timesteps)
            diffusion_loss = self.loss_fn.diffusion_loss(noise_pred, noise)
            
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.unet_optimizer.step()
            
            # Phase 3: MCTD policy improvement (every few batches)
            mctd_loss = torch.tensor(0.0)
            if batch_idx % 10 == 0:  # Less frequent MCTD training
                mctd_loss = self._train_mctd_policy(clean_latents[:2])  # Small batch
            
            # Update running losses
            total_vae_loss += vae_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_mctd_loss += mctd_loss.item() if isinstance(mctd_loss, torch.Tensor) else mctd_loss
            
            # Update progress bar
            pbar.set_postfix({
                'VAE': f'{vae_loss.item():.4f}',
                'Diff': f'{diffusion_loss.item():.4f}',
                'MCTD': f'{mctd_loss.item() if isinstance(mctd_loss, torch.Tensor) else mctd_loss:.4f}'
            })
        
        return {
            'vae_loss': total_vae_loss / len(dataloader),
            'diffusion_loss': total_diffusion_loss / len(dataloader),
            'mctd_loss': total_mctd_loss / len(dataloader)
        }
    
    def _train_mctd_policy(self, clean_latents: torch.Tensor) -> torch.Tensor:
        """Train MCTD policy using reinforcement learning"""
        with torch.no_grad():
            # Add noise to create initial states
            initial_timestep = 100  # Shorter for training efficiency
            noise = torch.randn_like(clean_latents)
            alpha = self.noise_schedule[initial_timestep]
            noisy_latents = (alpha.sqrt() * clean_latents + 
                           (1 - alpha).sqrt() * noise)
        
        # Collect MCTD rollouts
        tree_values = []
        target_rewards = []
        
        for i in range(clean_latents.shape[0]):
            latent = noisy_latents[i]
            
            # Create mini MCTD tree
            reward_fn = RewardFunction(self.vae, clean_latents[i:i+1])
            mctd = ThreadSafeMCTD(
                root_state=latent,
                initial_timestep=initial_timestep,
                unet=self.unet,
                reward_function=reward_fn,
                noise_schedule=self.noise_schedule
            )
            
            # Run few MCTD iterations
            for _ in range(5):  # Limited iterations for training
                reward = mctd.single_iteration()
                tree_values.append(mctd.root.value / max(mctd.root.visits, 1))
                target_rewards.append(reward)
        
        # Compute policy loss
        if tree_values and target_rewards:
            return self.loss_fn.mctd_policy_loss(tree_values, target_rewards)
        else:
            return torch.tensor(0.0)
    
    def train(self, 
              train_dataset: Dataset,
              num_epochs: int = 100,
              batch_size: int = 16,
              save_interval: int = 10,
              log_wandb: bool = True):
        """Complete training loop"""
        
        # Initialize wandb
        if log_wandb:
            wandb.init(project="fast-mctd", config={
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': self.vae_optimizer.param_groups[0]['lr']
            })
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        # Training loop
        for epoch in range(num_epochs):
            losses = self.train_epoch(train_loader, epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  VAE Loss: {losses['vae_loss']:.4f}")
            print(f"  Diffusion Loss: {losses['diffusion_loss']:.4f}")
            print(f"  MCTD Loss: {losses['mctd_loss']:.4f}")
            
            # Log to wandb
            if log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'vae_loss': losses['vae_loss'],
                    'diffusion_loss': losses['diffusion_loss'],
                    'mctd_loss': losses['mctd_loss']
                })
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                
                # Generate samples for visualization
                if log_wandb:
                    self._log_samples_to_wandb(epoch)
        
        if log_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'vae_state_dict': self.vae.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'unet_optimizer_state_dict': self.unet_optimizer.state_dict(),
            'noise_schedule': self.noise_schedule
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.unet_optimizer.load_state_dict(checkpoint['unet_optimizer_state_dict'])
        self.noise_schedule = checkpoint['noise_schedule']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def _log_samples_to_wandb(self, epoch: int):
        """Generate and log samples to wandb"""
        self.vae.eval()
        self.unet.eval()
        
        try:
            # Create sampler
            sampler = FastMCTDSampler(
                vae=self.vae,
                unet=self.unet,
                noise_schedule=self.noise_schedule,
                num_workers=2,  # Reduced for memory
                device=self.device
            )
            
            # Generate samples
            with torch.no_grad():
                samples = sampler.sample(
                    batch_size=4,  # Small batch for logging
                    num_parallel_batches=3,  # Reduced iterations
                    num_sparse_iterations=2,
                    latent_dim=128
                )
            
            # Convert to wandb images
            samples_np = samples.cpu().numpy()
            samples_np = (samples_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
            samples_np = np.transpose(samples_np, (0, 2, 3, 1))  # BHWC
            
            wandb.log({
                f"samples_epoch_{epoch}": [wandb.Image(img) for img in samples_np]
            })
            
        except Exception as e:
            print(f"Failed to log samples: {e}")
        
        self.vae.train()
        self.unet.train()

#_____________________________________________________________________________________________________________________________
