#_____________________________________________________________________________________________________

# fastMCTD_MoR-enhanced.py (sample codes):
"""This is also a sample code that I spent the whole weekend to figure out, plus some copying from other sifus (sorry ya?) 
   so you have to really check and see how these can be incorporated into your fastMCTD codes (I presume you too are either an AI Engineer
   or passionate coder who is obsessed with AI like me :-P"""
class MoRFastMCTDSampler(FastMCTDSampler):
    """Fast-MCTD Sampler with Mixture of Recursions integration"""
    
    def __init__(self,
                 vae: EnhancedVAE,
                 unet: Union[AdvancedUNet, MoREnhancedUNet],
                 noise_schedule: torch.Tensor,
                 num_workers: int = 4,
                 device: str = 'cuda',
                 mor_config: Optional[Dict] = None):
        
        super().__init__(vae, unet, noise_schedule, num_workers, device)
        
        # MoR configuration
        self.mor_config = mor_config or {
            'max_recursions': 4,
            'complexity_threshold': 0.6,
            'adaptive_depth': True,
            'cache_computations': True
        }
        
        # Initialize MoR-specific components
        self.global_complexity_tracker = ComplexityTracker()
        self.adaptive_scheduler = AdaptiveComputationScheduler()
    
    def mor_sample(self,
                   batch_size: int = 1,
                   num_parallel_batches: int = 10,
                   num_sparse_iterations: int = 5,
                   latent_dim: int = 128,
                   initial_timestep: int = 1000,
                   complexity_budget: float = 1.0) -> torch.Tensor:
        """Generate samples using MoR-enhanced Fast-MCTD"""
        
        samples = []
        
        for b in range(batch_size):
            print(f"MoR sampling {b+1}/{batch_size}")
            
            # Initialize with random noise
            initial_latent = torch.randn(latent_dim, device=self.device)
            
            # Create MoR-enhanced reward function
            reward_fn = MoRRewardFunction(
                self.vae, 
                complexity_tracker=self.global_complexity_tracker
            )
            
            # Initialize MoR-MCTD
            mor_mctd = MoRThreadSafeMCTD(
                root_state=initial_latent,
                initial_timestep=initial_timestep,
                unet=self.unet,
                reward_function=reward_fn,
                noise_schedule=self.noise_schedule,
                mor_config=self.mor_config
            )
            
            # Adaptive sampling based on complexity budget
            adaptive_batches = self.adaptive_scheduler.schedule_batches(
                num_parallel_batches, complexity_budget
            )
            
            adaptive_iterations = self.adaptive_scheduler.schedule_iterations(
                num_sparse_iterations, complexity_budget
            )
            
            # MoR-enhanced parallel search
            for epoch in range(adaptive_batches):
                print(f"  MoR parallel batch {epoch+1}/{adaptive_batches}")
                
                # Track complexity before iteration
                pre_complexity = mor_mctd.get_mor_statistics()['complexity_stats']['mean']
                
                # Execute MoR iterations
                batch_rewards = []
                for _ in range(16):  # Standard batch size
                    reward = mor_mctd.mor_single_iteration()
                    batch_rewards.append(reward)
                
                avg_reward = np.mean(batch_rewards)
                post_complexity = mor_mctd.get_mor_statistics()['complexity_stats']['mean']
                
                # Update complexity tracker
                self.global_complexity_tracker.update(pre_complexity, post_complexity, avg_reward)
                
                print(f"    Avg reward: {avg_reward:.4f}, Complexity: {post_complexity:.3f}")
                
                # Adaptive early stopping
                if self.adaptive_scheduler.should_early_stop(avg_reward, post_complexity):
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            # Sparse iterations for long-horizon planning
            sparse_mctd = MoRSparseMCTD(mor_mctd, skip_interval=3)
            for _ in range(adaptive_iterations):
                sparse_mctd.sparse_search_iteration()
            
            # Extract best trajectory
            best_latent = self._extract_best_trajectory(mor_mctd.root)
            
            # Decode to image space
            with torch.no_grad():
                generated_image = self.vae.decode(best_latent.unsqueeze(0))
                samples.append(generated_image.squeeze(0))
            
            # Log MoR statistics
            mor_stats = mor_mctd.get_mor_statistics()
            print(f"  MoR Stats - Recursion rate: {mor_stats['router_stats']['recursion_rate']:.3f}")
            print(f"  Cache hit rate: {mor_stats['cache_stats']['cache_hit_rate']:.3f}")
        
        return torch.stack(samples)
    
    def get_comprehensive_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics including MoR metrics"""
        stats = {
            'complexity_tracker': self.global_complexity_tracker.get_stats(),
            'adaptive_scheduler': self.adaptive_scheduler.get_stats()
        }
        
        if isinstance(self.unet, MoREnhancedUNet):
            stats['unet_mor_stats'] = self.unet.get_mor_layer_stats()
        
        return stats

class ComplexityTracker:
    """Tracks complexity patterns across sampling sessions"""
    
    def __init__(self):
        self.complexity_history = []
        self.reward_history = []
        self.efficiency_history = []
    
    def update(self, pre_complexity: float, post_complexity: float, reward: float):
        """Update complexity tracking"""
        complexity_reduction = pre_complexity - post_complexity
        efficiency = reward / max(pre_complexity, 0.1)  # Reward per unit complexity
        
        self.complexity_history.append({
            'pre': pre_complexity,
            'post': post_complexity,
            'reduction': complexity_reduction
        })
        self.reward_history.append(reward)
        self.efficiency_history.append(efficiency)
    
    def get_stats(self) -> Dict[str, float]:
        """Get complexity tracking statistics"""
        if not self.complexity_history:
            return {'mean_efficiency': 0.0, 'complexity_trend': 0.0}
        
        efficiencies = self.efficiency_history[-100:]  # Last 100 updates
        complexities = [c['reduction'] for c in self.complexity_history[-100:]]
        
        return {
            'mean_efficiency': np.mean(efficiencies),
            'efficiency_std': np.std(efficiencies),
            'complexity_trend': np.mean(complexities),
            'total_updates': len(self.complexity_history)
        }

class AdaptiveComputationScheduler:
    """Schedules computation based on complexity and performance"""
    
    def __init__(self):
        self.performance_history = []
        self.early_stop_threshold = 0.95  # Stop if reward plateau
        self.complexity_target = 0.4  # Target complexity level
    
    def schedule_batches(self, base_batches: int, complexity_budget: float) -> int:
        """Schedule number of parallel batches based on complexity budget"""
        budget_multiplier = min(2.0, complexity_budget * 1.5)
        scheduled_batches = int(base_batches * budget_multiplier)
        return max(3, min(scheduled_batches, 20))  # Clamp between 3-20
    
    def schedule_iterations(self, base_iterations: int, complexity_budget: float) -> int:
        """Schedule sparse iterations based on complexity budget"""
        budget_multiplier = complexity_budget
        scheduled_iterations = int(base_iterations * budget_multiplier)
        return max(2, min(scheduled_iterations, 10))  # Clamp between 2-10
    
    def should_early_stop(self, current_reward: float, current_complexity: float) -> bool:
        """Determine if early stopping should be triggered"""
        self.performance_history.append({
            'reward': current_reward,
            'complexity': current_complexity
        })
        
        # Need at least 3 data points
        if len(self.performance_history) < 3:
            return False
        
        # Check for reward plateau
        recent_rewards = [p['reward'] for p in self.performance_history[-3:]]
        reward_improvement = (recent_rewards[-1] - recent_rewards[0]) / max(recent_rewards[0], 0.1)
        
        # Check if complexity is very low (simple problem)
        if current_complexity < 0.2:
            return True
        
        # Check for reward plateau
        if reward_improvement < 0.05:  # Less than 5% improvement
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, float]:
        """Get scheduler statistics"""
        if not self.performance_history:
            return {'total_decisions': 0}
        
        rewards = [p['reward'] for p in self.performance_history]
        complexities = [p['complexity'] for p in self.performance_history]
        
        return {
            'total_decisions': len(self.performance_history),
            'mean_reward': np.mean(rewards),
            'mean_complexity': np.mean(complexities),
            'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0.0
        }

class MoRRewardFunction(RewardFunction):
    """Reward function enhanced with complexity-aware evaluation"""
    
    def __init__(self, vae: EnhancedVAE, 
                 target_distribution: Optional[torch.Tensor] = None,
                 complexity_tracker: Optional[ComplexityTracker] = None):
        super().__init__(vae, target_distribution)
        self.complexity_tracker = complexity_tracker
    
    def evaluate(self, latent: torch.Tensor) -> float:
        """Complexity-aware reward evaluation"""
        base_reward = super().evaluate(latent)
        
        # Analyze latent complexity
        complexity = self._compute_latent_complexity(latent)
        
        # Adaptive reward scaling based on complexity
        if complexity > 0.7:  # High complexity
            # Reward high-quality outputs more for complex states
            quality_bonus = base_reward * 0.3 if base_reward > 0.5 else 0.0
        else:  # Low complexity
            # Maintain reasonable rewards for simple states
            quality_bonus = base_reward * 0.1
        
        total_reward = base_reward + quality_bonus
        
        # Update complexity tracker if available
        if self.complexity_tracker:
            self.complexity_tracker.update(complexity, complexity, total_reward)
        
        return total_reward
    
    def _compute_latent_complexity(self, latent: torch.Tensor) -> float:
        """Compute complexity of latent representation"""
        with torch.no_grad():
            # Statistical measures
            variance = torch.var(latent).item()
            entropy = -torch.sum(F.softmax(latent, dim=0) * F.log_softmax(latent, dim=0)).item()
            
            # Spectral complexity (FFT-based)
            fft_coeffs = torch.fft.fft(latent.view(-1))
            spectral_energy = torch.sum(torch.abs(fft_coeffs)**2).item()
            
            # Normalize and combine
            normalized_variance = min(variance / 2.0, 1.0)
            normalized_entropy = min(entropy / 10.0, 1.0)
            normalized_spectral = min(spectral_energy / 1000.0, 1.0)
            
            complexity = (normalized_variance + normalized_entropy + normalized_spectral) / 3.0
            
        return complexity

class MoRSparseMCTD(SparseMCTD):
    """Sparse MCTD with MoR routing for long-horizon planning"""
    
    def __init__(self, base_mor_mctd: MoRThreadSafeMCTD, skip_interval: int = 2):
        self.base_mctd = base_mor_mctd  # MoR-enhanced base
        self.skip_interval = skip_interval
        self.adaptive_skip = True
    
    def adaptive_sparse_iteration(self) -> float:
        """Sparse iteration with adaptive skip intervals based on complexity"""
        with self.base_mctd.lock:
            selected_node = self.base_mctd.select_node_with_mor(self.base_mctd.root)
            
            # Adapt skip interval based on node complexity
            if hasattr(selected_node, 'complexity_score'):
                # More skipping for simple states, less for complex
                adaptive_skip = max(1, int(self.skip_interval * (1.2 - selected_node.complexity_score)))
            else:
                adaptive_skip = self.skip_interval
            
            # Sparse expansion with adaptive skipping
            if not selected_node.is_terminal and selected_node.visits > 0:
                children = self._adaptive_sparse_expand(selected_node, adaptive_skip)
                selected_node = children[0] if children else selected_node
        
        # Sparse simulation
        reward = self._adaptive_sparse_simulate(selected_node, adaptive_skip)
        
        # Backpropagation
        with self.base_mctd.lock:
            selected_node.backpropagate(reward)
        
        return reward
    
    def _adaptive_sparse_expand(self, node: MoREnhancedMCTDNode, skip_interval: int) -> List[MoREnhancedMCTDNode]:
        """Adaptive sparse expansion"""
        # Use MoR router to determine if expansion should be sparse or dense
        if node.router and node.complexity_score > 0.6:
            # High complexity: use denser expansion
            return self.base_mctd.expand_node_with_mor(node, num_actions=6)
        else:
            # Low complexity: use standard sparse expansion
            return self._sparse_expand(node)
    
    def _adaptive_sparse_simulate(self, node: MoREnhancedMCTDNode, skip_interval: int) -> float:
        """Adaptive sparse simulation"""
        current_state = node.state.clone()
        current_timestep = node.timestep
        
        while current_timestep > 0:
            # Determine next timestep based on complexity
            if hasattr(node, 'complexity_score') and node.complexity_score > 0.5:
                # High complexity: smaller steps
                next_timestep = max(0, current_timestep - max(1, skip_interval // 2))
            else:
                # Low complexity: larger steps
                next_timestep = max(0, current_timestep - skip_interval)
            
            # Multi-step denoising
            with torch.no_grad():
                for t in range(current_timestep, next_timestep, -1):
                    timestep_tensor = torch.tensor([t], device=current_state.device)
                    
                    # Use MoR-enhanced UNet if available
                    if isinstance(self.base_mctd.unet, MoREnhancedUNet):
                        predicted_noise = self.base_mctd.unet.forward_with_mor(
                            current_state.unsqueeze(0), timestep_tensor
                        ).squeeze(0)
                    else:
                        predicted_noise = self.base_mctd.unet(
                            current_state.unsqueeze(0), timestep_tensor
                        ).squeeze(0)
                    
                    alpha = self.base_mctd.noise_schedule[t]
                    beta = 1 - alpha
                    current_state = (current_state - beta.sqrt() * predicted_noise) / alpha.sqrt()
            
            current_timestep = next_timestep
        
        return self.base_mctd.reward_function.evaluate(current_state)

  #___________________________________________________________________________________________________________________
