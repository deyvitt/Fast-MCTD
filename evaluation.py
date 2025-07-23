#______________________________________________________________________________________________________

# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ComprehensiveEvaluator:
    """Complete evaluation pipeline for Fast-MCTD"""
    
    def __init__(self, 
                 sampler: FastMCTDSampler,
                 quality_evaluator: QualityEvaluator,
                 reference_dataset: torch.Tensor,
                 device: str = 'cuda'):
        self.sampler = sampler
        self.quality_evaluator = quality_evaluator
        self.reference_dataset = reference_dataset
        self.device = device
        
    def full_evaluation(self, 
                       num_samples: int = 100,
                       sampling_configs: List[Dict] = None,
                       save_dir: str = 'evaluation_results') -> Dict:
        """Run comprehensive evaluation"""
        
        if sampling_configs is None:
            sampling_configs = [
                {'name': 'standard', 'parallel_batches': 10, 'sparse_iterations': 5},
                {'name': 'high_quality', 'parallel_batches': 20, 'sparse_iterations': 10},
                {'name': 'fast', 'parallel_batches': 5, 'sparse_iterations': 3}
            ]
        
        results = {}
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for config in sampling_configs:
            print(f"\nEvaluating configuration: {config['name']}")
            
            # Generate samples
            config_samples = self._generate_samples_for_config(config, num_samples)
            
            # Compute quality metrics
            quality_metrics = self._evaluate_quality(config_samples)
            
            # Compute speed metrics
            speed_metrics = self._evaluate_speed(config)
            
            # Combine results
            config_results = {
                'quality': quality_metrics,
                'speed': speed_metrics,
                'config': config
            }
            
            results[config['name']] = config_results
            
            # Save samples
            sample_path = save_path / f"samples_{config['name']}.pt"
            torch.save(config_samples, sample_path)
        
        # Generate comparison plots
        self._create_comparison_plots(results, save_path)
        
        # Save full results
        torch.save(results, save_path / 'full_results.pt')
        
        return results
    
    def _generate_samples_for_config(self, config: Dict, num_samples: int) -> torch.Tensor:
        """Generate samples for a specific configuration"""
        batch_size = min(16, num_samples)  # Process in batches
        all_samples = []
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            samples = self.sampler.sample(
                batch_size=current_batch_size,
                num_parallel_batches=config.get('parallel_batches', 10),
                num_sparse_iterations=config.get('sparse_iterations', 5)
            )
            
            all_samples.append(samples)
            print(f"  Generated {i + current_batch_size}/{num_samples} samples")
        
        return torch.cat(all_samples, dim=0)
    
    def _evaluate_quality(self, samples: torch.Tensor) -> Dict[str, float]:
        """Evaluate quality metrics"""
        print("  Computing quality metrics...")
        
        # Select subset of reference images for comparison
        ref_subset = self.reference_dataset[:len(samples)]
        
        return self.quality_evaluator.evaluate_all_metrics(ref_subset, samples)
    
    def _evaluate_speed(self, config: Dict) -> Dict[str, float]:
        """Evaluate speed metrics"""
        print("  Computing speed metrics...")
        
        import time
        
        # Time single sample generation
        start_time = time.time()
        self.sampler.sample(
            batch_size=1,
            num_parallel_batches=config.get('parallel_batches', 10),
            num_sparse_iterations=config.get('sparse_iterations', 5)
        )
        single_sample_time = time.time() - start_time
        
        # Time batch generation
        start_time = time.time()
        self.sampler.sample(
            batch_size=4,
            num_parallel_batches=config.get('parallel_batches', 10),
            num_sparse_iterations=config.get('sparse_iterations', 5)
        )
        batch_time = time.time() - start_time
        
        return {
            'single_sample_time': single_sample_time,
            'batch_time': batch_time,
            'samples_per_second': 4 / batch_time,
            'efficiency': 4 / (batch_time / single_sample_time)  # Batch efficiency
        }
    
    def _create_comparison_plots(self, results: Dict, save_path: Path):
        """Create comparison plots"""
        print("Creating comparison plots...")
        
        # Quality comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        configs = list(results.keys())
        metrics = ['fid', 'lpips', 'is_mean', 'ssim', 'psnr']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            values = [results[config]['quality'][metric] for config in configs]
            
            ax.bar(configs, values)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.upper())
            
            # Rotate x-axis labels if needed
            if len(max(configs, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        # Speed comparison
        ax = axes[1, 2]
        speed_values = [results[config]['speed']['samples_per_second'] for config in configs]
        ax.bar(configs, speed_values, color='orange')
        ax.set_title('Speed Comparison')
        ax.set_ylabel('Samples per Second')
        
        plt.tight_layout()
        plt.savefig(save_path / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Quality vs Speed scatter plot
        plt.figure(figsize=(10, 8))
        
        for config in configs:
            quality_score = 1 / results[config]['quality']['fid']  # Higher is better
            speed_score = results[config]['speed']['samples_per_second']
            
            plt.scatter(speed_score, quality_score, s=100, label=config)
            plt.annotate(config, (speed_score, quality_score), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Samples per Second')
        plt.ylabel('Quality Score (1/FID)')
        plt.title('Quality vs Speed Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path / 'quality_speed_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_path}")

#_________________________________________________________________________________________________________
