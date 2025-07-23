#_______________________________________________________________________________________________________________

# inference.py
import lpips
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional

class QualityEvaluator:
    """Comprehensive quality evaluation for generated samples"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Initialize LPIPS (Learned Perceptual Image Patch Similarity)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # Initialize Inception for FID
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
    
    def compute_fid(self, real_images: torch.Tensor, 
                   generated_images: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance"""
        
        def get_activations(images):
            with torch.no_grad():
                # Resize to 299x299 for Inception
                images_resized = F.interpolate(images, size=(299, 299), 
                                             mode='bilinear', align_corners=False)
                
                # Get features before final layer
                features = self.inception(images_resized)
                return features.cpu().numpy()
        
        # Get features
        real_features = get_activations(real_images)
        gen_features = get_activations(generated_images)
        
        # Compute statistics
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real @ sigma_gen)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return float(fid)
    
    def compute_lpips(self, real_images: torch.Tensor, 
                     generated_images: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance"""
        with torch.no_grad():
            # Normalize to [-1, 1] for LPIPS
            real_norm = 2 * real_images - 1
            gen_norm = 2 * generated_images - 1
            
            lpips_scores = []
            for i in range(min(len(real_norm), len(gen_norm))):
                score = self.lpips_fn(real_norm[i:i+1], gen_norm[i:i+1])
                lpips_scores.append(score.item())
            
            return np.mean(lpips_scores)
    
    def compute_is(self, generated_images: torch.Tensor, 
                   splits: int = 10) -> Tuple[float, float]:
        """Compute Inception Score"""
        with torch.no_grad():
            # Resize for Inception
            images_resized = F.interpolate(generated_images, size=(299, 299), 
                                         mode='bilinear', align_corners=False)
            
            # Get predictions
            preds = F.softmax(self.inception(images_resized), dim=1)
            preds_np = preds.cpu().numpy()
            
            # Split and compute IS
            scores = []
            for i in range(splits):
                part = preds_np[i * len(preds_np) // splits:(i + 1) * len(preds_np) // splits]
                kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
                kl_div = np.mean(np.sum(kl_div, axis=1))
                scores.append(np.exp(kl_div))
            
            return np.mean(scores), np.std(scores)
    
    def compute_ssim(self, real_images: torch.Tensor, 
                    generated_images: torch.Tensor) -> float:
        """Compute Structural Similarity Index"""
        def ssim_single(img1, img2):
            # Convert to grayscale if RGB
            if img1.shape[1] == 3:
                img1 = torch.mean(img1, dim=1, keepdim=True)
                img2 = torch.mean(img2, dim=1, keepdim=True)
            
            # Constants for SSIM
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu1 = F.avg_pool2d(img1, 3, 1, 1)
            mu2 = F.avg_pool2d(img2, 3, 1, 1)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map.mean().item()
        
        ssim_scores = []
        for i in range(min(len(real_images), len(generated_images))):
            score = ssim_single(real_images[i:i+1], generated_images[i:i+1])
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    def compute_psnr(self, real_images: torch.Tensor, 
                    generated_images: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse_values = []
        for i in range(min(len(real_images), len(generated_images))):
            mse = F.mse_loss(real_images[i], generated_images[i])
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            mse_values.append(psnr.item())
        
        return np.mean(mse_values)
    
    def evaluate_all_metrics(self, real_images: torch.Tensor, 
                           generated_images: torch.Tensor) -> Dict[str, float]:
        """Compute all quality metrics"""
        metrics = {}
        
        print("Computing FID...")
        metrics['fid'] = self.compute_fid(real_images, generated_images)
        
        print("Computing LPIPS...")
        metrics['lpips'] = self.compute_lpips(real_images, generated_images)
        
        print("Computing IS...")
        is_mean, is_std = self.compute_is(generated_images)
        metrics['is_mean'] = is_mean
        metrics['is_std'] = is_std
        
        print("Computing SSIM...")
        metrics['ssim'] = self.compute_ssim(real_images, generated_images)
        
        print("Computing PSNR...")
        metrics['psnr'] = self.compute_psnr(real_images, generated_images)
        
        return metrics

#_______________________________________________________________________________________________________________________
