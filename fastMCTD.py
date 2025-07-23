#_________________________________________________________________________________________________

# fastMCTD.py
import concurrent.futures
import copy
from threading import Barrier

class ParallelMCTD:
    """Parallel Monte Carlo Tree Diffusion with delayed updates"""
    
    def __init__(self, 
                 base_mctd: ThreadSafeMCTD,
                 num_workers: int = 4,
                 batch_size: int = 16):
        self.base_mctd = base_mctd
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def parallel_search_batch(self) -> List[float]:
        """Execute parallel MCTS iterations with delayed updates"""
        
        # Step 1: Create tree snapshot
        with self.base_mctd.lock:
            tree_snapshot = self._create_tree_snapshot()
        
        # Step 2: Parallel rollouts on snapshot
        rollout_results = []
        
        def worker_rollouts(worker_id: int) -> List[Tuple[List[MCTDNode], float]]:
            """Worker function for parallel rollouts"""
            local_results = []
            rollouts_per_worker = self.batch_size // self.num_workers
            
            for _ in range(rollouts_per_worker):
                # Selection on snapshot
                path = []
                current = tree_snapshot
                
                while current.children and not current.is_terminal:
                    current = current.select_best_child()
                    path.append(current)
                
                # Expansion and simulation
                if not current.is_terminal:
                    children = self._expand_node_snapshot(current)
                    if children:
                        current = children[0]
                        path.append(current)
                
                # Simulation
                reward = self._simulate_from_node(current)
                local_results.append((path, reward))
            
            return local_results
        
        # Execute parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(worker_rollouts, i) for i in range(self.num_workers)]
            
            for future in concurrent.futures.as_completed(futures):
                rollout_results.extend(future.result())
        
        # Step 3: Delayed batch updates to main tree
        rewards = []
        with self.base_mctd.lock:
            for path, reward in rollout_results:
                self._apply_delayed_update(path, reward)
                rewards.append(reward)
        
        return rewards
    
    def _create_tree_snapshot(self) -> MCTDNode:
        """Create a deep copy of the current tree for parallel processing"""
        return copy.deepcopy(self.base_mctd.root)
    
    def _expand_node_snapshot(self, node: MCTDNode) -> List[MCTDNode]:
        """Expand node in snapshot (simplified version)"""
        if node.is_terminal or node.children:
            return node.children
        
        # Generate actions using the shared UNet
        with torch.no_grad():
            timestep_tensor = torch.tensor([node.timestep], device=node.state.device)
            predicted_noise = self.base_mctd.unet(node.state.unsqueeze(0), timestep_tensor)
            
            # Create multiple action variations
            actions = []
            for i in range(3):  # Reduced for efficiency
                noise_var = predicted_noise + 0.1 * torch.randn_like(predicted_noise)
                alpha = self.base_mctd.noise_schedule[node.timestep]
                beta = 1 - alpha
                denoised = (node.state.unsqueeze(0) - beta.sqrt() * noise_var) / alpha.sqrt()
                actions.append(denoised.squeeze(0) - node.state)
        
        return node.expand(actions)
    
    def _simulate_from_node(self, node: MCTDNode) -> float:
        """Simulate trajectory from node"""
        return self.base_mctd.simulate(node)
    
    def _apply_delayed_update(self, path: List[MCTDNode], reward: float):
        """Apply delayed update to corresponding nodes in main tree"""
        current = self.base_mctd.root
        
        # Navigate to corresponding node in main tree
        for snapshot_node in path:
            # Find matching child in main tree
            matching_child = None
            for child in current.children:
                if torch.allclose(child.state, snapshot_node.state, atol=1e-6):
                    matching_child = child
                    break
            
            if matching_child is None:
                # Create missing node in main tree
                matching_child = MCTDNode(
                    state=snapshot_node.state.clone(),
                    timestep=snapshot_node.timestep,
                    parent=current
                )
                current.children.append(matching_child)
            
            current = matching_child
        
        # Backpropagate reward
        current.backpropagate(reward)

class SparseMCTD:
    """Sparse MCTD with coarsened trajectory planning"""
    
    def __init__(self, 
                 base_mctd: ThreadSafeMCTD,
                 skip_interval: int = 2):
        self.base_mctd = base_mctd
        self.skip_interval = skip_interval
    
    def sparse_search_iteration(self) -> float:
        """Perform one sparse MCTS iteration with larger steps"""
        with self.base_mctd.lock:
            # Selection
            selected_node = self.base_mctd.select_node(self.base_mctd.root)
            
            # Expansion with sparse actions
            if not selected_node.is_terminal and selected_node.visits > 0:
                children = self._sparse_expand(selected_node)
                selected_node = children[0] if children else selected_node
        
        # Sparse simulation
        reward = self._sparse_simulate(selected_node)
        
        # Backpropagation
        with self.base_mctd.lock:
            selected_node.backpropagate(reward)
        
        return reward
    
    def _sparse_expand(self, node: MCTDNode) -> List[MCTDNode]:
        """Expand with coarsened actions"""
        if node.is_terminal or node.children:
            return node.children
        
        # Generate sparse denoising actions
        actions = []
        current_timestep = node.timestep
        
        while current_timestep > 0 and len(actions) < 3:
            target_timestep = max(0, current_timestep - self.skip_interval)
            
            with torch.no_grad():
                # Multi-step denoising
                temp_state = node.state.clone()
                for t in range(current_timestep, target_timestep, -1):
                    timestep_tensor = torch.tensor([t], device=temp_state.device)
                    predicted_noise = self.base_mctd.unet(temp_state.unsqueeze(0), timestep_tensor)
                    
                    alpha = self.base_mctd.noise_schedule[t]
                    beta = 1 - alpha
                    temp_state = (temp_state - beta.sqrt() * predicted_noise.squeeze(0)) / alpha.sqrt()
                
                actions.append(temp_state - node.state)
                current_timestep = target_timestep
        
        return node.expand(actions) if actions else [node]
    
    def _sparse_simulate(self, node: MCTDNode) -> float:
        """Simulate with sparse timesteps"""
        current_state = node.state.clone()
        current_timestep = node.timestep
        
        while current_timestep > 0:
            next_timestep = max(0, current_timestep - self.skip_interval)
            
            with torch.no_grad():
                # Multi-step denoising
                for t in range(current_timestep, next_timestep, -1):
                    timestep_tensor = torch.tensor([t], device=current_state.device)
                    predicted_noise = self.base_mctd.unet(current_state.unsqueeze(0), timestep_tensor)
                    
                    alpha = self.base_mctd.noise_schedule[t]
                    beta = 1 - alpha
                    current_state = (current_state - beta.sqrt() * predicted_noise.squeeze(0)) / alpha.sqrt()
            
            current_timestep = next_timestep
        
        return self.base_mctd.reward_function.evaluate(current_state)

class FastMCTDSampler:
    """Complete Fast-MCTD sampling orchestrator"""
    
    def __init__(self,
                 vae: EnhancedVAE,
                 unet: AdvancedUNet,
                 noise_schedule: torch.Tensor,
                 num_workers: int = 4,
                 device: str = 'cuda'):
        self.vae = vae
        self.unet = unet
        self.noise_schedule = noise_schedule
        self.num_workers = num_workers
        self.device = device
        
        # Move models to device
        self.vae.to(device)
        self.unet.to(device)
        
    def sample(self,
               batch_size: int = 1,
               num_parallel_batches: int = 10,
               num_sparse_iterations: int = 5,
               latent_dim: int = 128,
               initial_timestep: int = 1000) -> torch.Tensor:
        """Generate samples using Fast-MCTD"""
        
        samples = []
        
        for b in range(batch_size):
            print(f"Generating sample {b+1}/{batch_size}")
            
            # Initialize with random noise in latent space
            initial_latent = torch.randn(latent_dim, device=self.device)
            
            # Create reward function
            reward_fn = RewardFunction(self.vae)
            
            # Initialize MCTD
            base_mctd = ThreadSafeMCTD(
                root_state=initial_latent,
                initial_timestep=initial_timestep,
                unet=self.unet,
                reward_function=reward_fn,
                noise_schedule=self.noise_schedule
            )
            
            # Initialize parallel and sparse components
            parallel_mctd = ParallelMCTD(base_mctd, self.num_workers, batch_size=16)
            sparse_mctd = SparseMCTD(base_mctd, skip_interval=3)
            
            # Alternating parallel and sparse search phases
            for epoch in range(num_parallel_batches):
                print(f"  Parallel batch {epoch+1}/{num_parallel_batches}")
                
                # Parallel phase
                parallel_rewards = parallel_mctd.parallel_search_batch()
                avg_reward = np.mean(parallel_rewards)
                print(f"    Avg parallel reward: {avg_reward:.4f}")
                
                # Sparse phase for long-horizon planning
                sparse_rewards = []
                for _ in range(num_sparse_iterations):
                    reward = sparse_mctd.sparse_search_iteration()
                    sparse_rewards.append(reward)
                
                avg_sparse_reward = np.mean(sparse_rewards)
                print(f"    Avg sparse reward: {avg_sparse_reward:.4f}")
            
            # Extract best trajectory
            best_latent = self._extract_best_trajectory(base_mctd.root)
            
            # Decode to image space
            with torch.no_grad():
                generated_image = self.vae.decode(best_latent.unsqueeze(0))
                samples.append(generated_image.squeeze(0))
        
        return torch.stack(samples)
    
    def _extract_best_trajectory(self, root: MCTDNode) -> torch.Tensor:
        """Extract the best trajectory from the MCTS tree"""
        current = root
        
        # Follow best path to terminal
        while current.children and not current.is_terminal:
            # Select child with highest value/visit ratio
            best_child = max(current.children, 
                           key=lambda c: c.value / max(c.visits, 1))
            current = best_child
        
        return current.state

# Noise schedule utilities
def create_noise_schedule(num_timesteps: int = 1000, 
                         schedule_type: str = 'cosine') -> torch.Tensor:
    """Create noise schedule for diffusion process"""
    
    if schedule_type == 'linear':
        beta_start, beta_end = 0.0001, 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
    elif schedule_type == 'cosine':
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return alphas_cumprod

#__________________________________________________________________________________________________________________________
