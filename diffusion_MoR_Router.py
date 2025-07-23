#___________________________________________________________________________________________________________

# diffusion_MoR_Router.py - (Sample Code):
"""This is where the additional module we need to allow us to modularly add the MoR into the Fast-MCTD architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import math

class DiffusionMoRRouter(nn.Module):
    """MoR Router specialized for diffusion model states"""
    
    def __init__(self,
                 latent_dim: int,
                 timestep_embed_dim: int = 128,
                 hidden_dim: int = 256,
                 max_recursions: int = 4,
                 complexity_threshold: float = 0.6):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.hidden_dim = hidden_dim
        self.max_recursions = max_recursions
        self.complexity_threshold = complexity_threshold
        
        # Timestep embedding for diffusion context
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, timestep_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim // 2, timestep_embed_dim)
        )
        
        # State complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Routing decision network
        input_dim = latent_dim + timestep_embed_dim + 1  # +1 for complexity score
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),  # [recurse, forward, skip, terminate]
            nn.Softmax(dim=-1)
        )
        
        # Recursion depth embedding
        self.depth_embed = nn.Embedding(max_recursions + 1, latent_dim)
        
        # Metrics tracking
        self.register_buffer('total_decisions', torch.tensor(0))
        self.register_buffer('recurse_decisions', torch.tensor(0))
        self.register_buffer('complexity_scores', torch.zeros(1000))  # Rolling buffer
        self.register_buffer('buffer_idx', torch.tensor(0))
    
    def analyze_state_complexity(self, 
                                latent_state: torch.Tensor,
                                timestep: int) -> float:
        """Analyze complexity of current diffusion state"""
        with torch.no_grad():
            # Compute state entropy
            state_var = torch.var(latent_state).item()
            
            # Compute gradient magnitude (proxy for denoising difficulty)
            latent_state.requires_grad_(True)
            fake_loss = torch.sum(latent_state ** 2)
            grad = torch.autograd.grad(fake_loss, latent_state, create_graph=False)[0]
            grad_magnitude = torch.norm(grad).item()
            
            # Timestep-based complexity (early steps are more complex)
            timestep_complexity = timestep / 1000.0
            
            # Neural complexity prediction
            complexity_features = latent_state.detach()
            neural_complexity = self.complexity_analyzer(complexity_features).item()
            
            # Combined complexity score
            complexity = (
                0.3 * min(state_var, 1.0) +
                0.3 * min(grad_magnitude / 10.0, 1.0) +
                0.2 * timestep_complexity +
                0.2 * neural_complexity
            )
            
        return min(complexity, 1.0)
    
    def forward(self, 
                latent_state: torch.Tensor,
                timestep: int,
                current_depth: int = 0,
                parent_complexity: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Make routing decision for diffusion state
        
        Returns:
            Dictionary with routing decision and metadata
        """
        batch_size = latent_state.shape[0] if latent_state.dim() > 1 else 1
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        # Analyze state complexity
        complexity = self.analyze_state_complexity(latent_state[0], timestep)
        
        # Update complexity tracking
        self.complexity_scores[self.buffer_idx % 1000] = complexity
        self.buffer_idx += 1
        
        # Create timestep embedding
        timestep_tensor = torch.tensor([timestep], dtype=torch.float32, 
                                     device=latent_state.device)
        timestep_emb = self.timestep_embed(timestep_tensor.unsqueeze(-1))
        
        # Add depth information
        depth_emb = self.depth_embed(torch.tensor(current_depth, device=latent_state.device))
        augmented_state = latent_state + depth_emb.unsqueeze(0)
        
        # Prepare router input
        complexity_tensor = torch.tensor([complexity], device=latent_state.device)
        router_input = torch.cat([
            augmented_state.flatten(start_dim=1),
            timestep_emb.expand(batch_size, -1),
            complexity_tensor.expand(batch_size, 1)
        ], dim=1)
        
        # Get routing probabilities
        routing_probs = self.router(router_input)
        
        # Make routing decision based on complexity and depth
        decision_logits = routing_probs.clone()
        
        # Modify probabilities based on constraints
        if current_depth >= self.max_recursions:
            decision_logits[:, 0] = 0  # Disable recursion
        
        if complexity < self.complexity_threshold:
            decision_logits[:, 0] *= 0.5  # Reduce recursion probability for simple states
        
        if timestep < 10:  # Near final steps
            decision_logits[:, 3] += 0.3  # Encourage termination
        
        # Renormalize
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Update metrics
        self.total_decisions += 1
        if decision_probs[0, 0] > 0.5:  # Recurse decision
            self.recurse_decisions += 1
        
        return {
            'routing_probs': decision_probs,
            'complexity': complexity,
            'timestep_emb': timestep_emb,
            'decision_logits': decision_logits,
            'augmented_state': augmented_state
        }
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics"""
        if self.total_decisions > 0:
            recursion_rate = (self.recurse_decisions.float() / 
                            self.total_decisions.float()).item()
        else:
            recursion_rate = 0.0
        
        # Average complexity over recent decisions
        valid_scores = self.complexity_scores[self.complexity_scores > 0]
        avg_complexity = valid_scores.mean().item() if len(valid_scores) > 0 else 0.0
        
        return {
            'total_decisions': int(self.total_decisions.item()),
            'recurse_decisions': int(self.recurse_decisions.item()),
            'recursion_rate': recursion_rate,
            'avg_complexity': avg_complexity,
            'complexity_std': valid_scores.std().item() if len(valid_scores) > 0 else 0.0
        }

class MoREnhancedMCTDNode(MCTDNode):
    """MCTD Node enhanced with MoR routing capabilities"""
    
    def __init__(self, 
                 state: torch.Tensor,
                 timestep: int,
                 parent: Optional['MoREnhancedMCTDNode'] = None,
                 router: Optional[DiffusionMoRRouter] = None,
                 recursion_depth: int = 0):
        super().__init__(state, timestep, parent)
        self.router = router
        self.recursion_depth = recursion_depth
        self.complexity_score = 0.0
        self.routing_history = []
        self.cached_computations = {}
    
    def should_recurse(self) -> bool:
        """Determine if this node should recurse based on MoR routing"""
        if self.router is None:
            return False
        
        routing_result = self.router(
            self.state, 
            self.timestep, 
            self.recursion_depth
        )
        
        self.complexity_score = routing_result['complexity']
        routing_probs = routing_result['routing_probs']
        
        # Store routing decision
        decision = {
            'timestep': self.timestep,
            'depth': self.recursion_depth,
            'complexity': self.complexity_score,
            'recurse_prob': routing_probs[0, 0].item(),
            'forward_prob': routing_probs[0, 1].item()
        }
        self.routing_history.append(decision)
        
        # Decision based on highest probability
        action_idx = torch.argmax(routing_probs, dim=1)[0].item()
        
        return action_idx == 0  # 0 = recurse
    
    def get_exploration_priority(self) -> float:
        """Get exploration priority based on complexity and routing"""
        base_priority = super().ucb_score()
        
        # Boost priority for complex states
        complexity_boost = self.complexity_score * 0.5
        
        # Reduce priority for deeply recursed nodes
        depth_penalty = self.recursion_depth * 0.1
        
        return base_priority + complexity_boost - depth_penalty

class MoRThreadSafeMCTD(ThreadSafeMCTD):
    """Thread-safe MCTD with MoR routing integration"""
    
    def __init__(self,
                 root_state: torch.Tensor,
                 initial_timestep: int,
                 unet: AdvancedUNet,
                 reward_function: RewardFunction,
                 noise_schedule: torch.Tensor,
                 mor_config: Optional[Dict] = None):
        
        # Initialize MoR router
        mor_config = mor_config or {}
        self.mor_router = DiffusionMoRRouter(
            latent_dim=root_state.shape[0],
            **mor_config
        )
        
        # Create MoR-enhanced root node
        self.root = MoREnhancedMCTDNode(
            root_state, 
            initial_timestep,
            router=self.mor_router
        )
        
        self.unet = unet
        self.reward_function = reward_function
        self.noise_schedule = noise_schedule
        self.lock = threading.RLock()
        
        # MoR-specific tracking
        self.recursion_cache = {}
        self.complexity_history = []
    
    def select_node_with_mor(self, node: MoREnhancedMCTDNode) -> MoREnhancedMCTDNode:
        """Select node using MoR-enhanced selection policy"""
        current = node
        
        while current.children and not current.is_terminal:
            # Use MoR routing to influence selection
            if isinstance(current, MoREnhancedMCTDNode) and current.should_recurse():
                # Create recursive node
                recursive_node = MoREnhancedMCTDNode(
                    state=current.state,
                    timestep=current.timestep,
                    parent=current,
                    router=self.mor_router,
                    recursion_depth=current.recursion_depth + 1
                )
                current.children.append(recursive_node)
                return recursive_node
            else:
                # Select best child based on MoR-enhanced priority
                if hasattr(current.children[0], 'get_exploration_priority'):
                    current = max(current.children, 
                                key=lambda c: c.get_exploration_priority())
                else:
                    current = current.select_best_child()
        
        return current
    
    def expand_node_with_mor(self, node: MoREnhancedMCTDNode, 
                           num_actions: int = 4) -> List[MoREnhancedMCTDNode]:
        """Expand node with MoR-guided action generation"""
        if node.is_terminal:
            return [node]
        
        # Check cache first
        cache_key = f"{node.timestep}_{hash(node.state.cpu().numpy().tobytes())}"
        if cache_key in self.recursion_cache:
            cached_actions = self.recursion_cache[cache_key]
        else:
            # Generate actions using UNet
            with torch.no_grad():
                timestep_tensor = torch.tensor([node.timestep], device=node.state.device)
                predicted_noise = self.unet(node.state.unsqueeze(0), timestep_tensor)
                
                # Generate multiple action variations based on complexity
                complexity = node.complexity_score
                num_variations = max(2, int(num_actions * (1 + complexity)))
                
                actions = []
                for i in range(min(num_variations, num_actions)):
                    # Add complexity-based noise variation
                    noise_scale = 0.05 + 0.15 * complexity
                    noise_variation = predicted_noise + noise_scale * torch.randn_like(predicted_noise)
                    
                    # Denoising step
                    alpha = self.noise_schedule[node.timestep]
                    beta = 1 - alpha
                    denoised = (node.state.unsqueeze(0) - beta.sqrt() * noise_variation) / alpha.sqrt()
                    
                    actions.append(denoised.squeeze(0) - node.state)
                
                # Cache for future use
                self.recursion_cache[cache_key] = actions
                cached_actions = actions
        
        # Create MoR-enhanced child nodes
        new_children = []
        for action in cached_actions:
            new_state = node.state + action
            child = MoREnhancedMCTDNode(
                state=new_state,
                timestep=node.timestep - 1,
                parent=node,
                router=self.mor_router,
                recursion_depth=0  # Reset recursion depth for new node
            )
            node.children.append(child)
            new_children.append(child)
        
        return new_children
    
    def mor_single_iteration(self) -> float:
        """Perform one MCTS iteration with MoR routing"""
        with self.lock:
            # MoR-enhanced selection
            selected_node = self.select_node_with_mor(self.root)
            
            # MoR-enhanced expansion
            if not selected_node.is_terminal and selected_node.visits > 0:
                children = self.expand_node_with_mor(selected_node)
                selected_node = children[0] if children else selected_node
        
        # Simulation (can be done without lock)
        reward = self.simulate(selected_node)
        
        # Backpropagation with complexity weighting
        with self.lock:
            # Weight reward based on complexity
            complexity_weight = 1.0 + 0.5 * selected_node.complexity_score
            weighted_reward = reward * complexity_weight
            selected_node.backpropagate(weighted_reward)
        
        return reward
    
    def get_mor_statistics(self) -> Dict[str, any]:
        """Get comprehensive MoR statistics"""
        router_stats = self.mor_router.get_routing_stats()
        
        # Analyze tree structure
        total_nodes = self._count_nodes(self.root)
        recursive_nodes = self._count_recursive_nodes(self.root)
        
        # Complexity distribution
        complexities = self._collect_complexities(self.root)
        
        return {
            'router_stats': router_stats,
            'tree_stats': {
                'total_nodes': total_nodes,
                'recursive_nodes': recursive_nodes,
                'recursion_ratio': recursive_nodes / max(total_nodes, 1)
            },
            'complexity_stats': {
                'mean': np.mean(complexities) if complexities else 0.0,
                'std': np.std(complexities) if complexities else 0.0,
                'min': np.min(complexities) if complexities else 0.0,
                'max': np.max(complexities) if complexities else 0.0
            },
            'cache_stats': {
                'cache_size': len(self.recursion_cache),
                'cache_hit_rate': getattr(self, 'cache_hits', 0) / max(getattr(self, 'cache_attempts', 1), 1)
            }
        }
    
    def _count_nodes(self, node: MoREnhancedMCTDNode) -> int:
        """Recursively count nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _count_recursive_nodes(self, node: MoREnhancedMCTDNode) -> int:
        """Count nodes that used recursion"""
        count = 1 if node.recursion_depth > 0 else 0
        for child in node.children:
            count += self._count_recursive_nodes(child)
        return count
    
    def _collect_complexities(self, node: MoREnhancedMCTDNode) -> List[float]:
        """Collect complexity scores from all nodes"""
        complexities = [node.complexity_score] if node.complexity_score > 0 else []
        for child in node.children:
            complexities.extend(self._collect_complexities(child))
        return complexities

#____________________________________________________________________________________________________________________
