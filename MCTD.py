#_________________________________________________________________________________________________

# MCTD.py
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class MCTDNode:
    """Node in the Monte Carlo Tree Diffusion search tree"""
    state: torch.Tensor  # Current latent state
    timestep: int        # Diffusion timestep
    parent: Optional['MCTDNode'] = None
    children: List['MCTDNode'] = None
    visits: int = 0
    value: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def is_terminal(self) -> bool:
        return self.timestep <= 0
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        ) if self.parent else 0.0
        
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = 1.414) -> 'MCTDNode':
        """Select child with highest UCB score"""
        return max(self.children, key=lambda c: c.ucb_score(exploration_constant))
    
    def expand(self, possible_actions: List[torch.Tensor]) -> List['MCTDNode']:
        """Expand node by adding children for possible actions"""
        new_children = []
        for action in possible_actions:
            new_state = self.state + action  # Simplified action application
            child = MCTDNode(
                state=new_state,
                timestep=self.timestep - 1,
                parent=self
            )
            self.children.append(child)
            new_children.append(child)
        return new_children
    
    def backpropagate(self, reward: float):
        """Backpropagate reward up the tree"""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

class RewardFunction:
    """Evaluates the quality of generated samples"""
    def __init__(self, vae: EnhancedVAE, target_distribution: Optional[torch.Tensor] = None):
        self.vae = vae
        self.target_distribution = target_distribution
        
    def evaluate(self, latent: torch.Tensor) -> float:
        """Compute reward for a latent state"""
        with torch.no_grad():
            # Decode latent to image space
            generated = self.vae.decode(latent.unsqueeze(0))
            
            # Multiple reward components
            rewards = []
            
            # 1. Reconstruction quality (negative MSE from a clean reference)
            if self.target_distribution is not None:
                mse_reward = -F.mse_loss(generated, self.target_distribution).item()
                rewards.append(mse_reward)
            
            # 2. Diversity (entropy of generated pixels)
            entropy_reward = self._compute_entropy(generated)
            rewards.append(entropy_reward * 0.1)
            
            # 3. Perceptual quality (simplified - could use LPIPS)
            perceptual_reward = self._compute_perceptual_quality(generated)
            rewards.append(perceptual_reward * 0.5)
            
            return sum(rewards)
    
    def _compute_entropy(self, image: torch.Tensor) -> float:
        """Compute image entropy as diversity measure"""
        # Convert to numpy and compute histogram
        img_np = image.cpu().numpy().flatten()
        hist, _ = np.histogram(img_np, bins=256, range=(0, 1))
        hist = hist / hist.sum()  # Normalize
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return entropy
    
    def _compute_perceptual_quality(self, image: torch.Tensor) -> float:
        """Simplified perceptual quality measure"""
        # Edge sharpness
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        if image.shape[1] == 3:  # RGB
            image_gray = torch.mean(image, dim=1, keepdim=True)
        else:
            image_gray = image
            
        edges_x = F.conv2d(image_gray, sobel_x, padding=1)
        edges_y = F.conv2d(image_gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        sharpness = torch.mean(edges).item()
        return min(sharpness, 1.0)  # Cap at 1.0

class ThreadSafeMCTD:
    """Thread-safe Monte Carlo Tree Diffusion implementation"""
    def __init__(self, 
                 root_state: torch.Tensor,
                 initial_timestep: int,
                 unet: AdvancedUNet,
                 reward_function: RewardFunction,
                 noise_schedule: torch.Tensor):
        self.root = MCTDNode(root_state, initial_timestep)
        self.unet = unet
        self.reward_function = reward_function
        self.noise_schedule = noise_schedule
        self.lock = threading.RLock()
        
    def select_node(self, node: MCTDNode) -> MCTDNode:
        """Select leaf node using UCB policy"""
        current = node
        while current.children and not current.is_terminal:
            current = current.select_best_child()
        return current
    
    def expand_node(self, node: MCTDNode, num_actions: int = 4) -> List[MCTDNode]:
        """Expand node with possible denoising actions"""
        if node.is_terminal:
            return [node]
            
        # Generate possible denoising actions using UNet
        with torch.no_grad():
            # Predict noise
            timestep_tensor = torch.tensor([node.timestep], device=node.state.device)
            predicted_noise = self.unet(node.state.unsqueeze(0), timestep_tensor)
            
            # Generate multiple action variations
            actions = []
            for i in range(num_actions):
                # Add small random perturbations for exploration
                noise_variation = predicted_noise + 0.1 * torch.randn_like(predicted_noise)
                
                # Denoising step
                alpha = self.noise_schedule[node.timestep]
                beta = 1 - alpha
                denoised = (node.state.unsqueeze(0) - beta.sqrt() * noise_variation) / alpha.sqrt()
                
                actions.append(denoised.squeeze(0) - node.state)
        
        return node.expand(actions)
    
    def simulate(self, node: MCTDNode) -> float:
        """Simulate trajectory from node to terminal state"""
        current_state = node.state.clone()
        current_timestep = node.timestep
        
        # Rollout trajectory
        while current_timestep > 0:
            with torch.no_grad():
                timestep_tensor = torch.tensor([current_timestep], device=current_state.device)
                predicted_noise = self.unet(current_state.unsqueeze(0), timestep_tensor)
                
                # Denoising step
                alpha = self.noise_schedule[current_timestep]
                beta = 1 - alpha
                current_state = (current_state - beta.sqrt() * predicted_noise.squeeze(0)) / alpha.sqrt()
                current_timestep -= 1
        
        # Evaluate final state
        return self.reward_function.evaluate(current_state)
    
    def single_iteration(self) -> float:
        """Perform one MCTS iteration"""
        with self.lock:
            # Selection
            selected_node = self.select_node(self.root)
            
            # Expansion
            if not selected_node.is_terminal and selected_node.visits > 0:
                children = self.expand_node(selected_node)
                selected_node = children[0] if children else selected_node
        
        # Simulation (can be done without lock)
        reward = self.simulate(selected_node)
        
        # Backpropagation
        with self.lock:
            selected_node.backpropagate(reward)
        
        return reward

#________________________________________________________________________________________________________________
