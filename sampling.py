#______________________________________________________________________________________________________

# sampling.py
class AdaptiveFastMCTDSampler(FastMCTDSampler):
    """Adaptive sampler with dynamic resource allocation"""
    
    def __init__(self, *args, resource_budget: int = 1000, 
                 adaptation_strategy: str = 'performance_based', **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_budget = resource_budget
        self.adaptation_strategy = adaptation_strategy
        self.performance_history = []
        self.current_allocation = {
            'parallel_workers': self.num_workers,
            'parallel_batches': 10,
            'sparse_iterations': 5
        }
    
    def adaptive_sample(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples with adaptive resource allocation"""
        samples = []
        
        for b in range(batch_size):
            print(f"Adaptive sampling {b+1}/{batch_size}")
            
            # Allocate resources based on strategy
            allocation = self._allocate_resources()
            
            # Sample with current allocation
            start_time = time.time()
            sample = self._sample_with_allocation(allocation, **kwargs)
            end_time = time.time()
            
            # Record performance
            performance = {
                'time': end_time - start_time,
                'allocation': allocation.copy(),
                'quality': self._estimate_quality(sample)
            }
            self.performance_history.append(performance)
            
            # Adapt for next iteration
            self._adapt_allocation(performance)
            
            samples.append(sample)
        
        return torch.stack(samples)
    
    def _allocate_resources(self) -> Dict[str, int]:
        """Allocate computational resources based on strategy"""
        if self.adaptation_strategy == 'performance_based':
            return self._performance_based_allocation()
        elif self.adaptation_strategy == 'quality_focused':
            return self._quality_focused_allocation()
        elif self.adaptation_strategy == 'speed_focused':
            return self._speed_focused_allocation()
        else:
            return self.current_allocation.copy()
    
    def _performance_based_allocation(self) -> Dict[str, int]:
        """Allocate based on historical performance"""
        if len(self.performance_history) < 3:
            return self.current_allocation.copy()
        
        # Analyze recent performance
        recent_performance = self.performance_history[-3:]
        avg_time = np.mean([p['time'] for p in recent_performance])
        avg_quality = np.mean([p['quality'] for p in recent_performance])
        
        allocation = self.current_allocation.copy()
        
        # Adjust based on performance trends
        if avg_time > 30.0 and avg_quality < 0.7:  # Too slow, poor quality
            allocation['parallel_batches'] = max(5, allocation['parallel_batches'] - 2)
            allocation['sparse_iterations'] = max(3, allocation['sparse_iterations'] - 1)
        elif avg_time < 10.0 and avg_quality > 0.8:  # Fast and good quality
            allocation['parallel_batches'] = min(20, allocation['parallel_batches'] + 2)
            allocation['sparse_iterations'] = min(10, allocation['sparse_iterations'] + 1)
        
        return allocation
    
    def _quality_focused_allocation(self) -> Dict[str, int]:
        """Allocate resources to maximize quality"""
        return {
            'parallel_workers': min(8, self.num_workers * 2),
            'parallel_batches': 15,
            'sparse_iterations': 8
        }
    
    def _speed_focused_allocation(self) -> Dict[str, int]:
        """Allocate resources to maximize speed"""
        return {
            'parallel_workers': max(2, self.num_workers // 2),
            'parallel_batches': 5,
            'sparse_iterations': 3
        }
    
    def _sample_with_allocation(self, allocation: Dict[str, int], **kwargs) -> torch.Tensor:
        """Sample using specific resource allocation"""
        # Override current settings temporarily
        original_workers = self.num_workers
        self.num_workers = allocation['parallel_workers']
        
        try:
            sample = self.sample(
                batch_size=1,
                num_parallel_batches=allocation['parallel_batches'],
                num_sparse_iterations=allocation['sparse_iterations'],
                **kwargs
            )
            return sample[0]  # Return single sample
        finally:
            # Restore original settings
            self.num_workers = original_workers
    
    def _estimate_quality(self, sample: torch.Tensor) -> float:
        """Estimate sample quality using lightweight metrics"""
        with torch.no_grad():
            # Convert to image if needed
            if sample.dim() == 1:  # Latent space
                sample_img = self.vae.decode(sample.unsqueeze(0))
            else:
                sample_img = sample.unsqueeze(0)
            
            # Simple quality metrics
            # 1. Sharpness (edge variance)
            gray = torch.mean(sample_img, dim=1, keepdim=True)
            sobel_x = F.conv2d(gray, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                               dtype=torch.float32, device=sample.device), padding=1)
            sobel_y = F.conv2d(gray, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                               dtype=torch.float32, device=sample.device), padding=1)
            edges = torch.sqrt(sobel_x**2 + sobel_y**2)
            sharpness = torch.var(edges).item()
            
            # 2. Contrast
            contrast = torch.std(sample_img).item()
            
            # 3. Color diversity
            color_std = torch.std(sample_img, dim=[2, 3]).mean().item()
            
            # Combine metrics (normalized to [0, 1])
            quality_score = (
                min(sharpness / 10.0, 1.0) * 0.4 +
                min(contrast / 2.0, 1.0) * 0.4 +
                min(color_std / 1.0, 1.0) * 0.2
            )
            
            return quality_score
    
    def _adapt_allocation(self, performance: Dict):
        """Adapt current allocation based on performance"""
        efficiency = performance['quality'] / max(performance['time'], 0.1)
        
        if efficiency > 0.05:  # Good efficiency
            # Slightly increase resources
            self.current_allocation['parallel_batches'] = min(
                20, self.current_allocation['parallel_batches'] + 1
            )
        elif efficiency < 0.01:  # Poor efficiency
            # Reduce resources
            self.current_allocation['parallel_batches'] = max(
                5, self.current_allocation['parallel_batches'] - 1
            )
            self.current_allocation['sparse_iterations'] = max(
                3, self.current_allocation['sparse_iterations'] - 1
            )

class HierarchicalFastMCTDSampler(FastMCTDSampler):
    """Multi-level hierarchical sampling for complex generation"""
    
    def __init__(self, *args, num_levels: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_levels = num_levels
        
    def hierarchical_sample(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples using hierarchical refinement"""
        samples = []
        
        for b in range(batch_size):
            print(f"Hierarchical sampling {b+1}/{batch_size}")
            
            # Start with coarse sampling
            current_sample = self._coarse_sample(**kwargs)
            
            # Refine through multiple levels
            for level in range(self.num_levels):
                print(f"  Refining level {level+1}/{self.num_levels}")
                current_sample = self._refine_sample(current_sample, level, **kwargs)
            
            samples.append(current_sample)
        
        return torch.stack(samples)
    
    def _coarse_sample(self, **kwargs) -> torch.Tensor:
        """Generate initial coarse sample"""
        # Use fewer iterations for coarse sampling
        return self.sample(
            batch_size=1,
            num_parallel_batches=3,
            num_sparse_iterations=2,
            initial_timestep=200,  # Start from less noisy state
            **kwargs
        )[0]
    
    def _refine_sample(self, sample: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """Refine sample at given level"""
        # Create a new MCTD tree starting from current sample
        latent_dim = sample.shape[0] if sample.dim() == 1 else sample.numel()
        
        if sample.dim() > 1:  # Convert image to latent
            with torch.no_grad():
                if hasattr(self.vae, 'encode'):
                    mu, _ = self.vae.encode(sample.unsqueeze(0))
                    sample = mu.squeeze(0)
                else:
                    sample = sample.view(-1)
        
        # Increase refinement quality with each level
        iterations = (level + 1) * 3
        sparse_its = (level + 1) * 2
        
        reward_fn = RewardFunction(self.vae)
        
        mctd = ThreadSafeMCTD(
            root_state=sample,
            initial_timestep=50 * (level + 1),  # More steps for finer levels
            unet=self.unet,
            reward_function=reward_fn,
            noise_schedule=self.noise_schedule
        )
        
        # Run refinement iterations
        for _ in range(iterations):
            mctd.single_iteration()
        
        # Extract refined sample
        refined_latent = self._extract_best_trajectory(mctd.root)
        
        return refined_latent

class ConditionalFastMCTDSampler(FastMCTDSampler):
    """Conditional sampling with guidance"""
    
    def __init__(self, *args, guidance_scale: float = 7.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.guidance_scale = guidance_scale
    
    def conditional_sample(self, 
                          condition: torch.Tensor,
                          batch_size: int = 1,
                          condition_type: str = 'text',
                          **kwargs) -> torch.Tensor:
        """Generate samples conditioned on input"""
        samples = []
        
        for b in range(batch_size):
            print(f"Conditional sampling {b+1}/{batch_size}")
            
            # Create conditional reward function
            reward_fn = ConditionalRewardFunction(
                self.vae, 
                condition=condition,
                condition_type=condition_type,
                guidance_scale=self.guidance_scale
            )
            
            # Initialize with random noise
            initial_latent = torch.randn(128, device=self.device)  # Assume 128-dim latent
            
            # Create MCTD with conditional rewards
            mctd = ThreadSafeMCTD(
                root_state=initial_latent,
                initial_timestep=1000,
                unet=self.unet,
                reward_function=reward_fn,
                noise_schedule=self.noise_schedule
            )
            
            # Run conditional search
            parallel_mctd = ParallelMCTD(mctd, self.num_workers, batch_size=16)
            
            for epoch in range(kwargs.get('num_parallel_batches', 10)):
                parallel_rewards = parallel_mctd.parallel_search_batch()
            
            # Extract best conditional sample
            best_latent = self._extract_best_trajectory(mctd.root)
            samples.append(best_latent)
        
        return torch.stack(samples)

class ConditionalRewardFunction(RewardFunction):
    """Reward function with conditional guidance"""
    
    def __init__(self, vae, condition: torch.Tensor, 
                 condition_type: str = 'text', 
                 guidance_scale: float = 7.5):
        super().__init__(vae)
        self.condition = condition
        self.condition_type = condition_type
        self.guidance_scale = guidance_scale
    
    def evaluate(self, latent: torch.Tensor) -> float:
        """Evaluate with conditional guidance"""
        # Base reward
        base_reward = super().evaluate(latent)
        
        # Conditional reward based on type
        if self.condition_type == 'text':
            conditional_reward = self._text_conditional_reward(latent)
        elif self.condition_type == 'image':
            conditional_reward = self._image_conditional_reward(latent)
        else:
            conditional_reward = 0.0
        
        # Combine with guidance scale
        total_reward = base_reward + self.guidance_scale * conditional_reward
        return total_reward
    
    def _text_conditional_reward(self, latent: torch.Tensor) -> float:
        """Text-conditioned reward (simplified)"""
        # In practice, this would use CLIP or similar text-image similarity
        # For now, return a simple conditional score
        return 0.1  # Placeholder
    
    def _image_conditional_reward(self, latent: torch.Tensor) -> float:
        """Image-conditioned reward"""
        with torch.no_grad():
            generated = self.vae.decode(latent.unsqueeze(0))
            
            # L2 similarity to condition
            similarity = -F.mse_loss(generated, self.condition.unsqueeze(0)).item()
            return similarity * 0.5

# Utility functions for sampling
def create_sampling_schedule(num_steps: int, schedule_type: str = 'ddpm') -> torch.Tensor:
    """Create sampling schedule for inference"""
    if schedule_type == 'ddpm':
        # Standard DDPM sampling schedule
        return torch.linspace(1000, 1, num_steps, dtype=torch.long)
    elif schedule_type == 'ddim':
        # DDIM uniform sampling
        step_size = 1000 // num_steps
        return torch.arange(1000, 0, -step_size, dtype=torch.long)
    elif schedule_type == 'adaptive':
        # Adaptive schedule (more steps early, fewer later)
        early_steps = torch.linspace(1000, 500, num_steps // 2, dtype=torch.long)
        late_steps = torch.linspace(499, 1, num_steps - num_steps // 2, dtype=torch.long)
        return torch.cat([early_steps, late_steps])
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def batch_sample_with_config(sampler: FastMCTDSampler,
                           config: Dict,
                           save_path: Optional[str] = None) -> torch.Tensor:
    """Batch sampling with configuration"""
    print(f"Starting batch sampling with config: {config}")
    
    samples = sampler.sample(
        batch_size=config.get('batch_size', 4),
        num_parallel_batches=config.get('parallel_batches', 10),
        num_sparse_iterations=config.get('sparse_iterations', 5),
        latent_dim=config.get('latent_dim', 128),
        initial_timestep=config.get('initial_timestep', 1000)
    )
    
    if save_path:
        torch.save(samples, save_path)
        print(f"Samples saved to {save_path}")
    
    return samples

#___________________________________________________________________________________________________________
