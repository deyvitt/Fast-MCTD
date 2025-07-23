#_______________________________________________________________________________________________________________

# uNet_MoR_enhanced.py (sample code):
"""This is just a sample code, so it might differ from your actual uNet codes. Therefore, you have to go through your uNet codes to see how you can add and enhance it with MoR codes"""
class MoREnhancedUNet(AdvancedUNet):
    """UNet with MoR routing for adaptive processing depth"""
    
    def __init__(self, *args, mor_layers: List[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Determine which layers get MoR routing
        if mor_layers is None:
            mor_layers = [0, 1, 2, 3]  # Early layers benefit most from MoR
        
        self.mor_layers = mor_layers
        self.layer_routers = nn.ModuleDict()
        
        # Create routers for specified layers
        for layer_idx in mor_layers:
            self.layer_routers[f'router_{layer_idx}'] = DiffusionMoRRouter(
                latent_dim=self.model_channels,
                hidden_dim=self.model_channels // 2,
                max_recursions=3 - layer_idx // 2,  # Fewer recursions in deeper layers
                complexity_threshold=0.4 + layer_idx * 0.1
            )
    
    def forward_with_mor(self, 
                        x: torch.Tensor, 
                        timesteps: torch.Tensor,
                        max_recursions_per_layer: int = 3) -> torch.Tensor:
        """Forward pass with MoR routing at specified layers"""
        
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Reshape latent to spatial format
        B, D = x.shape
        H = W = int(math.sqrt(D))
        if H * W != D:
            H = W = 16
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), H * W).squeeze(1)
        
        h = x.view(B, 1, H, W)
        h = self.input_proj(h)
        
        # Process through layers with MoR routing
        skip_connections = []
        
        for layer_idx, block in enumerate(self.down_blocks):
            if layer_idx in self.mor_layers:
                # Apply MoR routing
                router_key = f'router_{layer_idx}'
                router = self.layer_routers[router_key]
                
                # Flatten for router
                h_flat = h.view(B, -1)
                
                recursion_count = 0
                while recursion_count < max_recursions_per_layer:
                    # Get routing decision
                    routing_result = router(h_flat, timesteps[0].item(), recursion_count)
                    routing_probs = routing_result['routing_probs']
                    
                    # Make decision (0=recurse, 1=forward, 2=skip, 3=terminate)
                    decision = torch.argmax(routing_probs, dim=1)[0].item()
                    
                    if decision == 0 and recursion_count < max_recursions_per_layer - 1:
                        # Recurse: Apply same block again
                        if isinstance(block, UNetBlock):
                            h = block(h, time_emb)
                        else:
                            h = block(h)
                        recursion_count += 1
                    else:
                        # Forward or skip
                        if decision != 2:  # Not skip
                            if isinstance(block, UNetBlock):
                                h = block(h, time_emb)
                            else:
                                h = block(h)
                        break
            else:
                # Standard processing
                if isinstance(block, UNetBlock):
                    h = block(h, time_emb)
                else:
                    h = block(h)
            
            if isinstance(block, UNetBlock):
                skip_connections.append(h)
        
        # Middle processing
        h = self.middle_block[0](h, time_emb)
        h = self.middle_block[1](h, time_emb)
        
        # Decoder (simplified for this example)
        for block in self.up_blocks:
            if isinstance(block, UNetBlock) and skip_connections:
                h = torch.cat([h, skip_connections.pop()], dim=1)
                h = block(h, time_emb)
            else:
                h = block(h)
        
        # Output
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        
        # Reshape back to latent format
        h = h.view(B, -1)
        if h.shape[1] != x.shape[1]:
            h = F.adaptive_avg_pool1d(h.unsqueeze(1), x.shape[1]).squeeze(1)
        
        return h
    
    def get_mor_layer_stats(self) -> Dict[str, Dict]:
        """Get MoR statistics for each layer"""
        stats = {}
        for layer_idx in self.mor_layers:
            router_key = f'router_{layer_idx}'
            if router_key in self.layer_routers:
                stats[f'layer_{layer_idx}'] = self.layer_routers[router_key].get_routing_stats()
        return stats

#_______________________________________________________________________________________________________________________
