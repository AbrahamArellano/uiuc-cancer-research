#!/usr/bin/env python3
"""
H100 GPU Hyperparameter Optimization for TabNet Prostate Cancer Classification
Systematic grid search to achieve 80-85% accuracy target using 36 GPU hours
"""
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.append('/u/aa107/uiuc-cancer-research/src/model')
sys.path.append('/u/aa107/uiuc-cancer-research')

from tabnet_prostate_variant_classifier import ProstateVariantTabNet
from config.pipeline_config import config

class H100TabNetOptimizer:
    """
    H100 GPU hyperparameter optimization for TabNet prostate cancer classification
    Systematic grid search across 108 configurations to achieve 80-85% accuracy
    """
    
    def __init__(self, gpu_hours_budget=36, parallel_jobs=2):
        """
        Initialize optimizer
        
        Args:
            gpu_hours_budget: Total GPU hours available for optimization
            parallel_jobs: Number of parallel optimization jobs
        """
        self.gpu_hours_budget = gpu_hours_budget
        self.parallel_jobs = parallel_jobs
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.results_dir = config.RESULTS_DIR / "optimization" / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_results = []
        self.best_configs = []
        self.completed_configs = 0
        
        # Load data once
        self.X, self.y = self._load_optimization_data()
        
    def _load_optimization_data(self):
        """Load and prepare data for optimization"""
        print("📁 Loading optimization dataset...")
        
        model = ProstateVariantTabNet()
        X, y = model.load_data()
        
        # Use subset for faster optimization if dataset is very large
        max_optimization_size = 50000  # Reasonable size for H100 optimization
        if len(X) > max_optimization_size:
            print(f"  📊 Using subset of {max_optimization_size:,} for optimization")
            indices = np.random.choice(len(X), max_optimization_size, replace=False)
            X, y = X.iloc[indices], y[indices]
        
        print(f"✅ Optimization data ready: {len(X):,} variants, {X.shape[1]} features")
        return X, y
    
    def generate_hyperparameter_grid(self):
        """
        Generate systematic hyperparameter grid for TabNet optimization
        Total configurations: 3*3*3*3*3*4 = 108 configurations
        """
        param_grid = {
            'n_d': [32, 64, 128],
            'n_a': [32, 64, 128], 
            'n_steps': [6, 7, 8],
            'gamma': [1.0, 1.3, 1.5],
            'lambda_sparse': [1e-4, 1e-3, 1e-2],
            'learning_rate': [1e-3, 2e-3, 1e-2, 2e-2]
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        configurations = []
        
        for combination in itertools.product(*values):
            config_dict = dict(zip(keys, combination))
            configurations.append(config_dict)
        
        print(f"✅ Generated {len(configurations)} hyperparameter configurations")
        
        # Prioritize configurations likely to achieve target
        configurations = self._prioritize_configurations(configurations)
        
        return configurations
    
    def _prioritize_configurations(self, configurations):
        """
        Prioritize configurations based on TabNet best practices
        Put most promising configurations first
        """
        def config_priority_score(config):
            score = 0
            
            # Prefer balanced n_d and n_a
            if config['n_d'] == config['n_a']:
                score += 10
            
            # Prefer 6-7 steps for interpretability
            if config['n_steps'] in [6, 7]:
                score += 8
            
            # Prefer moderate gamma
            if config['gamma'] == 1.3:
                score += 5
            
            # Prefer moderate sparsity
            if config['lambda_sparse'] == 1e-3:
                score += 5
            
            # Prefer moderate learning rates
            if config['learning_rate'] in [2e-3, 1e-2]:
                score += 5
            
            # Prefer larger networks for complex data
            if config['n_d'] >= 64:
                score += 3
            
            return score
        
        # Sort by priority score (highest first)
        prioritized = sorted(configurations, key=config_priority_score, reverse=True)
        
        print("🎯 Configurations prioritized by expected performance")
        return prioritized
    
    def evaluate_configuration(self, config, cv_folds=3, max_epochs=100):
        """
        Evaluate a single hyperparameter configuration using cross-validation
        
        Args:
            config: Hyperparameter configuration dictionary
            cv_folds: Number of CV folds for evaluation
            max_epochs: Maximum training epochs per fold
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        print(f"⚙️  Evaluating config: n_d={config['n_d']}, n_a={config['n_a']}, "
              f"steps={config['n_steps']}, lr={config['learning_rate']}")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_f1s = []
        fold_aucs = []
        fold_times = []
        
        config_start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            fold_start = time.time()
            
            # Split data
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            try:
                # Create model with current configuration
                model = ProstateVariantTabNet(
                    n_d=config['n_d'],
                    n_a=config['n_a'],
                    n_steps=config['n_steps'],
                    gamma=config['gamma'],
                    lambda_sparse=config['lambda_sparse'],
                    learning_rate=config['learning_rate']
                )
                
                # Update learning rate in model
                model.model = None  # Reset model to apply new LR
                
                # Train with current config
                val_accuracy = model.train(
                    X_train, y_train, X_val, y_val,
                    max_epochs=max_epochs,
                    patience=10  # Reduced patience for optimization
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
                
                try:
                    auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
                except:
                    auc = 0.0
                
                fold_accuracies.append(accuracy)
                fold_f1s.append(f1)
                fold_aucs.append(auc)
                
                fold_time = time.time() - fold_start
                fold_times.append(fold_time)
                
                print(f"    Fold {fold + 1}: accuracy={accuracy:.3f}, time={fold_time:.1f}s")
                
            except Exception as e:
                print(f"    ❌ Fold {fold + 1} failed: {e}")
                fold_accuracies.append(0.0)
                fold_f1s.append(0.0)
                fold_aucs.append(0.0)
                fold_times.append(time.time() - fold_start)
        
        # Calculate summary metrics
        config_time = time.time() - config_start_time
        
        results = {
            'config': config,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_f1': np.mean(fold_f1s),
            'mean_auc': np.mean(fold_aucs),
            'fold_accuracies': fold_accuracies,
            'fold_f1s': fold_f1s,
            'fold_aucs': fold_aucs,
            'training_time': config_time,
            'avg_fold_time': np.mean(fold_times),
            'meets_target': np.mean(fold_accuracies) >= config.TARGET_ACCURACY,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✅ Config result: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        
        return results
    
    def run_optimization(self, save_frequency=10, target_configs=5):
        """
        Run complete hyperparameter optimization
        
        Args:
            save_frequency: Save results every N configurations
            target_configs: Stop early if we find this many target-meeting configs
        """
        global config
        
        print("🚀 Starting H100 TabNet Hyperparameter Optimization")
        print("=" * 70)
        print(f"GPU Budget: {self.gpu_hours_budget} hours")
        print(f"Target Accuracy: {config.TARGET_ACCURACY:.1%}")
        print(f"Results Directory: {self.results_dir}")
        print("=" * 70)
        
        # Generate configurations
        configurations = self.generate_hyperparameter_grid()
        total_configs = len(configurations)
        
        # Estimate time per configuration
        estimated_time_per_config = (self.gpu_hours_budget * 3600) / total_configs
        print(f"📊 Estimated time per config: {estimated_time_per_config:.1f} seconds")
        
        # Track progress
        target_meeting_configs = 0
        
        for i, config in enumerate(configurations):
            # Check time budget
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.gpu_hours_budget:
                print(f"⏰ GPU time budget exhausted ({elapsed_hours:.1f} hours)")
                break
            
            print(f"\n🔧 Configuration {i+1}/{total_configs} "
                  f"(Elapsed: {elapsed_hours:.1f}h/{self.gpu_hours_budget}h)")
            
            # Evaluate configuration
            try:
                results = self.evaluate_configuration(config)
                self.optimization_results.append(results)
                self.completed_configs += 1
                
                # Check if target is met
                if results['meets_target']:
                    target_meeting_configs += 1
                    print(f"🎯 Target achieved! ({target_meeting_configs}/{target_configs})")
                    
                    # Save this promising config
                    self.best_configs.append(results)
                
                # Save intermediate results
                if (i + 1) % save_frequency == 0:
                    self._save_intermediate_results()

                if (self.completed_configs % save_frequency == 0) and self.completed_configs > 0:
                    print(f"\n📊 Progress Update: {self.completed_configs} configurations completed")
                    self.save_optimization_heatmaps()  # Generate intermediate heat maps                    
                
                # Early stopping if we have enough good configs
                if target_meeting_configs >= target_configs:
                    print(f"✅ Found {target_configs} target-meeting configurations!")
                    print("🎉 Early stopping - optimization goals achieved")
                    break
                    
            except Exception as e:
                print(f"❌ Configuration {i+1} failed: {e}")
                continue
        
        # Final results processing
        self._finalize_optimization()

        print("\n📊 GENERATING FINAL HEAT MAPS")
        final_heatmaps = self.create_hyperparameter_heatmaps(self.optimization_results)        

        return self.optimization_results
    
    def _save_intermediate_results(self):
        """Save intermediate optimization results"""
        results_file = self.results_dir / "intermediate_results.json"
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'completed_configs': self.completed_configs,
            'target_meeting_configs': len(self.best_configs),
            'optimization_results': self.optimization_results,
            'best_configs': self.best_configs
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"💾 Intermediate results saved: {results_file}")
    
    def _finalize_optimization(self):
        """Process and save final optimization results"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETED - PROCESSING RESULTS")
        print("=" * 70)
        
        if not self.optimization_results:
            print("❌ No successful configurations found")
            return
        
        # Sort results by accuracy
        sorted_results = sorted(
            self.optimization_results, 
            key=lambda x: x['mean_accuracy'], 
            reverse=True
        )
        
        # Get top configurations
        top_configs = sorted_results[:10]
        target_meeting = [r for r in sorted_results if r['meets_target']]
        
        # Generate final report
        self._generate_optimization_report(sorted_results, top_configs, target_meeting)
        
        # Save all results
        final_results_file = self.results_dir / "final_optimization_results.json"
        with open(final_results_file, 'w') as f:
            json.dump({
                'optimization_summary': {
                    'total_configs_evaluated': len(self.optimization_results),
                    'target_meeting_configs': len(target_meeting),
                    'best_accuracy': sorted_results[0]['mean_accuracy'] if sorted_results else 0,
                    'optimization_time_hours': (time.time() - self.start_time) / 3600
                },
                'top_configurations': top_configs,
                'target_meeting_configurations': target_meeting,
                'all_results': sorted_results
            }, f, indent=2, default=str)
        
        print(f"✅ Final results saved: {final_results_file}")
        
        # Save best model configurations for easy loading
        if target_meeting:
            best_config = target_meeting[0]['config']
            best_config_file = self.results_dir / "best_configuration.json"
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"🏆 Best configuration saved: {best_config_file}")
        
    def _generate_optimization_report(self, all_results, top_configs, target_meeting):
        """Generate human-readable optimization report"""
        report_file = self.results_dir / "optimization_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("H100 TABNET HYPERPARAMETER OPTIMIZATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary
            f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Evaluated: {len(all_results)}\n")
            f.write(f"Target Accuracy: {config.TARGET_ACCURACY:.1%}\n")
            f.write(f"Configurations Meeting Target: {len(target_meeting)}\n")
            f.write(f"Optimization Time: {(time.time() - self.start_time) / 3600:.1f} hours\n\n")
            
            if all_results:
                best_result = all_results[0]
                f.write(f"Best Accuracy Achieved: {best_result['mean_accuracy']:.3f} ± {best_result['std_accuracy']:.3f}\n")
                f.write(f"Best Configuration:\n")
                for key, value in best_result['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Top 5 configurations
            f.write("TOP 5 CONFIGURATIONS:\n")
            f.write("-" * 50 + "\n")
            for i, result in enumerate(top_configs[:5]):
                f.write(f"{i+1}. Accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}\n")
                f.write(f"   Config: n_d={result['config']['n_d']}, n_a={result['config']['n_a']}, ")
                f.write(f"steps={result['config']['n_steps']}, lr={result['config']['learning_rate']}\n")
                f.write(f"   Target Met: {'✅' if result['meets_target'] else '❌'}\n\n")
            
            # Target meeting configurations
            if target_meeting:
                f.write("CONFIGURATIONS MEETING TARGET:\n")
                f.write("-" * 50 + "\n")
                for i, result in enumerate(target_meeting):
                    f.write(f"{i+1}. Accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}\n")
                    f.write(f"   Training Time: {result['training_time']:.1f} seconds\n")
                    config_str = ", ".join([f"{k}={v}" for k, v in result['config'].items()])
                    f.write(f"   Config: {config_str}\n\n")
            else:
                f.write("NO CONFIGURATIONS MET TARGET ACCURACY\n")
                f.write("Consider:\n")
                f.write("- Expanding hyperparameter ranges\n")
                f.write("- Increasing training time\n")
                f.write("- Adding more features\n")
                f.write("- Ensemble methods\n\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"📊 Optimization report saved: {report_file}")

    def create_hyperparameter_heatmaps(self, results):
        """Create comprehensive heat map visualizations for hyperparameter tracking"""
        print("\n📊 GENERATING HYPERPARAMETER HEAT MAPS")
        print("-" * 40)
        
        if len(results) < 4:
            print("⚠️ Insufficient results for heat map generation")
            return []
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                **result['config'],
                'accuracy': result['mean_accuracy'],
                'std_accuracy': result['std_accuracy'],
                'training_time': result['training_time'],
                'meets_target': result['meets_target']
            }
            for result in results
        ])
        
        heatmap_files = []
        
        # 1. Architecture Heat Map (n_d vs n_a)
        plt.figure(figsize=(10, 8))
        pivot1 = df.pivot_table(values='accuracy', index='n_d', columns='n_a', aggfunc='mean')
        sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Accuracy'})
        plt.title('TabNet Architecture: Decision Width (n_d) vs Attention Width (n_a)', fontsize=14)
        plt.xlabel('Attention Width (n_a)')
        plt.ylabel('Decision Width (n_d)')
        
        heatmap_file1 = self.results_dir / "heatmap_architecture.png"
        plt.savefig(heatmap_file1, dpi=300, bbox_inches='tight')
        plt.close()
        heatmap_files.append(heatmap_file1)
        
        # 2. Learning Dynamics Heat Map (learning_rate vs gamma)
        plt.figure(figsize=(10, 8))
        pivot2 = df.pivot_table(values='accuracy', index='learning_rate', columns='gamma', aggfunc='mean')
        sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='plasma', cbar_kws={'label': 'Accuracy'})
        plt.title('Learning Dynamics: Learning Rate vs Feature Reuse (Gamma)', fontsize=14)
        plt.xlabel('Feature Reuse Parameter (gamma)')
        plt.ylabel('Learning Rate')
        
        heatmap_file2 = self.results_dir / "heatmap_learning_dynamics.png"
        plt.savefig(heatmap_file2, dpi=300, bbox_inches='tight')
        plt.close()
        heatmap_files.append(heatmap_file2)
        
        # 3. Regularization Heat Map (n_steps vs lambda_sparse)
        plt.figure(figsize=(10, 8))
        pivot3 = df.pivot_table(values='accuracy', index='n_steps', columns='lambda_sparse', aggfunc='mean')
        sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'Accuracy'})
        plt.title('Regularization: Decision Steps vs Sparsity (Lambda)', fontsize=14)
        plt.xlabel('Sparsity Parameter (lambda_sparse)')
        plt.ylabel('Decision Steps (n_steps)')
        
        heatmap_file3 = self.results_dir / "heatmap_regularization.png"
        plt.savefig(heatmap_file3, dpi=300, bbox_inches='tight')
        plt.close()
        heatmap_files.append(heatmap_file3)
        
        # 4. Performance Summary Heat Map
        plt.figure(figsize=(12, 8))
        
        # Create a comprehensive view
        # Use n_d and learning_rate as primary axes
        pivot4 = df.pivot_table(values='accuracy', index='n_d', columns='learning_rate', aggfunc='mean')
        sns.heatmap(pivot4, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'}, center=0.82)
        plt.title('Performance Overview: Architecture vs Learning Rate\n(Target: 82% accuracy)', fontsize=14)
        plt.xlabel('Learning Rate')
        plt.ylabel('Decision Width (n_d)')
        
        # Add target line annotation
        plt.axhline(y=0.82, color='red', linestyle='--', alpha=0.7, label='Target: 82%')
        
        heatmap_file4 = self.results_dir / "heatmap_performance_overview.png"
        plt.savefig(heatmap_file4, dpi=300, bbox_inches='tight')
        plt.close()
        heatmap_files.append(heatmap_file4)
        
        # 5. Training Efficiency Heat Map (accuracy vs training_time)
        plt.figure(figsize=(10, 8))
        
        # Create efficiency score (accuracy / log(training_time))
        df['efficiency'] = df['accuracy'] / np.log(df['training_time'] + 1)
        pivot5 = df.pivot_table(values='efficiency', index='n_d', columns='n_a', aggfunc='mean')
        sns.heatmap(pivot5, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'Efficiency Score'})
        plt.title('Training Efficiency: Accuracy per Log(Training Time)', fontsize=14)
        plt.xlabel('Attention Width (n_a)')
        plt.ylabel('Decision Width (n_d)')
        
        heatmap_file5 = self.results_dir / "heatmap_efficiency.png"
        plt.savefig(heatmap_file5, dpi=300, bbox_inches='tight')
        plt.close()
        heatmap_files.append(heatmap_file5)
        
        print(f"✅ Generated {len(heatmap_files)} heat map visualizations")
        return heatmap_files
    
    def save_optimization_heatmaps(self):
        """Save heat maps during optimization progress"""
        if len(self.optimization_results) >= 10:  # Minimum results for meaningful visualization
            heatmap_files = self.create_hyperparameter_heatmaps(self.optimization_results)
            
            # Save current progress
            progress_file = self.results_dir / "optimization_progress.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    'completed_configs': len(self.optimization_results),
                    'best_accuracy': max(r['mean_accuracy'] for r in self.optimization_results),
                    'target_meeting_configs': len([r for r in self.optimization_results if r['meets_target']]),
                    'heatmap_files': [str(f) for f in heatmap_files]
                }, f, indent=2)
            
            return heatmap_files
        return []        

def main():
    """
    Main optimization workflow for Week 5
    """
    print("🧬 H100 TabNet Hyperparameter Optimization")
    print("Week 5: Systematic search for 80-85% accuracy")
    print("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot run H100 optimization")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detected: {gpu_name}")
    
    # Initialize optimizer
    optimizer = H100TabNetOptimizer(
        gpu_hours_budget=config.GPU_CONFIG['optimization_budget'],  # 36 hours
        parallel_jobs=config.GPU_CONFIG['parallel_jobs']  # 2 jobs
    )
    
    try:
        # Run optimization
        results = optimizer.run_optimization(
            save_frequency=5,  # Save every 5 configs
            target_configs=5   # Stop after finding 5 good configs
        )
        
        print("\n🎉 OPTIMIZATION COMPLETED!")
        
        if results:
            # Find best result
            best_result = max(results, key=lambda x: x['mean_accuracy'])
            print(f"🏆 Best Accuracy: {best_result['mean_accuracy']:.3f}")
            
            target_met = best_result['mean_accuracy'] >= config.TARGET_ACCURACY
            print(f"🎯 Target Status: {'✅ ACHIEVED' if target_met else '❌ MISSED'}")
            
            if target_met:
                print("\n✅ Ready for Week 6: Clinical validation and interpretability analysis")
            else:
                gap = config.TARGET_ACCURACY - best_result['mean_accuracy']
                print(f"\n⚠️  Gap to target: {gap:.3f} - consider ensemble methods or more features")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
        optimizer._save_intermediate_results()
        return optimizer.optimization_results
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()