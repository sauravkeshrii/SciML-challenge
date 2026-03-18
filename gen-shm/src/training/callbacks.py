"""
Training callbacks for monitoring and visualization.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_training_end(self, history: Dict[str, List[float]]) -> None:
        """Called at the end of training."""
        pass


class ModelCheckpointCallback(TrainingCallback):
    """Save model checkpoints during training."""
    
    def __init__(self, save_path: str, save_freq: int = 50, save_best_only: bool = True):
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        current_loss = metrics.get('total_loss', float('inf'))
        
        if self.save_best_only:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._save_model(epoch, is_best=True)
        else:
            if epoch % self.save_freq == 0:
                self._save_model(epoch)
    
    def _save_model(self, epoch: int, is_best: bool = False) -> None:
        # This would save the actual model - implementation depends on trainer
        pass


class LearningRateSchedulerCallback(TrainingCallback):
    """Dynamic learning rate scheduling based on training progress."""
    
    def __init__(self, scheduler_type: str = 'plateau'):
        self.scheduler_type = scheduler_type
        self.loss_history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.loss_history.append(metrics.get('total_loss', 0.0))
        
        # Could implement custom LR scheduling logic here
        if len(self.loss_history) > 10:
            recent_improvement = (self.loss_history[-10] - self.loss_history[-1]) / self.loss_history[-10]
            if recent_improvement < 0.01:  # Less than 1% improvement
                # Reduce learning rate logic would go here
                pass


class VisualizationCallback(TrainingCallback):
    """Visualize training progress and model predictions."""
    
    def __init__(self, plot_freq: int = 20, save_plots: bool = True, plots_dir: str = 'plots'):
        self.plot_freq = plot_freq
        self.save_plots = save_plots
        self.plots_dir = Path(plots_dir)
        if save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.figures = {}
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if epoch % self.plot_freq == 0:
            self._plot_training_curves(metrics)
    
    def on_training_end(self, history: Dict[str, List[float]]) -> None:
        self._plot_final_results(history)
    
    def _plot_training_curves(self, metrics: Dict[str, float]) -> None:
        """Plot current training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress')
        
        # Plot 1: Loss curves
        axes[0, 0].plot(list(metrics.keys()), list(metrics.values()), 'o-')
        axes[0, 0].set_title('Current Epoch Losses')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Physics vs Data loss ratio
        if 'physics_loss' in metrics and 'data_loss' in metrics:
            ratio = metrics['physics_loss'] / (metrics['data_loss'] + 1e-8)
            axes[0, 1].bar(['Physics/Data Ratio'], [ratio])
            axes[0, 1].set_title('Loss Balance')
            axes[0, 1].set_ylabel('Ratio')
        
        # Plot 3: Training time
        axes[1, 0].hist([], alpha=0.7)  # Placeholder
        axes[1, 0].set_title('Training Time Distribution')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Model prediction example (placeholder)
        axes[1, 1].plot([], [], 'b-', label='Prediction')
        axes[1, 1].plot([], [], 'r--', label='Ground Truth')
        axes[1, 1].set_title('Model Prediction Example')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Displacement')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plots_dir / f'training_epoch_{len(self.loss_history)}.png')
        plt.close()
    
    def _plot_final_results(self, history: Dict[str, List[float]]) -> None:
        """Plot final training results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Final Training Results')
        
        epochs = range(len(history['train_loss']))
        
        # Loss convergence
        axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].semilogy(epochs, history['physics_loss'], 'r-', label='Physics Loss')
        axes[0, 0].semilogy(epochs, history['data_loss'], 'g-', label='Data Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate schedule
        axes[0, 1].semilogy(epochs, history['learning_rate'], 'purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        
        # Training time per epoch
        axes[1, 0].plot(epochs, history['epoch_time'], 'orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True)
        
        # Loss components comparison
        if len(history['boundary_loss']) > 0:
            axes[1, 1].stackplot(epochs,
                               history['data_loss'],
                               history['physics_loss'],
                               history['boundary_loss'],
                               labels=['Data', 'Physics', 'Boundary'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Components')
            axes[1, 1].set_title('Loss Component Contributions')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plots_dir / 'final_training_results.png')
        plt.close()


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation performance."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait_count = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        current_loss = metrics.get('total_loss', float('inf'))
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            self.should_stop = True
            
        return self.should_stop


class PhysicsValidationCallback(TrainingCallback):
    """Validate physics compliance during training."""
    
    def __init__(self, model, validation_points: int = 1000):
        self.model = model
        self.validation_points = validation_points
        self.physics_errors = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if epoch % 20 == 0:  # Validate every 20 epochs
            physics_error = self._compute_physics_error()
            self.physics_errors.append(physics_error)
    
    def _compute_physics_error(self) -> float:
        """Compute average physics residual error."""
        self.model.eval()
        
        # Generate random collocation points
        x = torch.rand(self.validation_points) * 1.0  # Normalized length
        t = torch.rand(self.validation_points) * 1.0  # Time
        damage_loc = torch.rand(self.validation_points) * 0.8 + 0.1  # Avoid extremes
        damage_sev = torch.rand(self.validation_points) * 0.3  # Up to 30% damage
        
        with torch.no_grad():
            residual = self.model.compute_physics_loss(x, t, damage_loc, damage_sev)
            
        return residual.item()


# Composite callback manager
class CallbackManager:
    """Manage multiple callbacks during training."""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Execute all callbacks and check for early stopping."""
        should_stop = False
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                if callback.on_epoch_end(epoch, metrics):
                    should_stop = True
            else:
                callback.on_epoch_end(epoch, metrics)
        return should_stop
    
    def on_training_end(self, history: Dict[str, List[float]]) -> None:
        for callback in self.callbacks:
            callback.on_training_end(history)