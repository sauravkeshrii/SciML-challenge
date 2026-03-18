"""
Main training framework for Gen-SHM PINN model.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import time

from ..models.pinn_generator import PINNGenerator
from ..utils.config import config
from ..utils.logger import get_experiment_logger
from ..utils.helpers import get_device, set_seed
from .loss_functions import PhysicsRegularizedLoss, AdaptiveWeightScheduler
from ..data.data_generation import create_data_loaders


class TrainingMonitor:
    """
    Monitor training progress and performance metrics.
    """
    
    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.epoch_times = []
        
    def update(self, current_loss: float, epoch: int) -> bool:
        """
        Update monitor and check for early stopping.
        
        Args:
            current_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop early
        """
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience


class PINNTrainer:
    """
    Main trainer for Physics-Informed Neural Networks.
    
    Features:
    - Adaptive loss weighting
    - Learning rate scheduling
    - Progress monitoring
    - Checkpoint management
    - Early stopping
    """
    
    def __init__(self, model: PINNGenerator, config_dict: dict = None):
        self.model = model
        self.config = config_dict or config.config
        self.device = get_device()
        
        # Training components
        self.loss_function = PhysicsRegularizedLoss(self.config)
        self.weight_scheduler = AdaptiveWeightScheduler(self.config)
        self.monitor = TrainingMonitor(patience=100)
        
        # Optimizer setup
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Logging
        self.logger = get_experiment_logger('pinn_training')
        self.history = {
            'train_loss': [],
            'physics_loss': [],
            'data_loss': [],
            'boundary_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration."""
        opt_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_name = self.config['training']['lr_scheduler'].lower()
        
        if scheduler_name == 'cosine_annealing':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=30,
                min_lr=1e-6
            )
        else:
            return None
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'physics': 0.0,
            'data': 0.0,
            'boundary': 0.0,
            'initial': 0.0
        }
        
        num_batches = 0
        
        for batch_data in train_loader:
            # Move data to device
            batch_data = {k: v.squeeze(0).to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.loss_function.compute_regularized_loss(self.model, batch_data)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                loss_key = key if key == 'total' else f'{key}_loss'
                if loss_key in losses:
                    epoch_losses[key] += losses[loss_key].item()
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: v.squeeze(0).to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                losses = self.loss_function.compute_regularized_loss(self.model, batch_data)
                val_loss += losses['total_loss'].item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, training_data: Dict[str, torch.Tensor], 
              validation_data: Dict[str, torch.Tensor] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            training_data: Dictionary of training tensors
            validation_data: Dictionary of validation tensors (optional)
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create data loaders
        train_loader = create_data_loaders(
            training_data,
            batch_size=self.config['training']['batch_size']
        )
        
        val_loader = None
        if validation_data:
            val_loader = create_data_loaders(
                validation_data,
                batch_size=self.config['training']['batch_size'],
                shuffle=False
            )
        
        # Training loop
        num_epochs = self.config['training']['epochs']
        
        progress_bar = tqdm(range(num_epochs), disable=not verbose)
        
        for epoch in progress_bar:
            epoch_start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch(train_loader)
            
            # Validation (if available)
            val_loss = 0.0
            if val_loader:
                val_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update loss weights adaptively
            current_losses = {
                'data': train_losses['data'],
                'physics': train_losses['physics'],
                'boundary': train_losses['boundary']
            }
            updated_weights = self.weight_scheduler.update_weights(current_losses)
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self._record_history(train_losses, val_loss, epoch_time)
            
            # Update progress bar
            if verbose:
                progress_bar.set_description(
                    f"Loss: {train_losses['total']:.4f} | "
                    f"Phys: {train_losses['physics']:.4f} | "
                    f"Data: {train_losses['data']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Log periodically
            if epoch % 50 == 0:
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_losses['total']:.6f}, "
                    f"Physics Loss = {train_losses['physics']:.6f}, "
                    f"Data Loss = {train_losses['data']:.6f}, "
                    f"LRate = {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Early stopping check
            if self.monitor.update(train_losses['total'], epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return self.history
    
    def _record_history(self, train_losses: Dict[str, float], 
                       val_loss: float, epoch_time: float) -> None:
        """Record training metrics to history."""
        self.history['train_loss'].append(train_losses['total'])
        self.history['physics_loss'].append(train_losses['physics'])
        self.history['data_loss'].append(train_losses['data'])
        self.history['boundary_loss'].append(train_losses['boundary'])
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.history['epoch_time'].append(epoch_time)
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        
        return checkpoint['epoch']


class TransferLearningTrainer(PINNTrainer):
    """
    Trainer with transfer learning capabilities for different damage scenarios.
    """
    
    def __init__(self, base_model: PINNGenerator, config_dict: dict = None):
        super().__init__(base_model, config_dict)
        self.base_model = base_model
    
    def fine_tune_damage_scenario(self, target_damage_level: float,
                                training_data: Dict[str, torch.Tensor],
                                epochs: int = 200) -> None:
        """
        Fine-tune model for a specific damage scenario.
        
        Args:
            target_damage_level: Target damage severity to specialize for
            training_data: Training data for the target scenario
            epochs: Number of fine-tuning epochs
        """
        # Freeze early layers for stability
        self._freeze_early_layers()
        
        # Modify final layers for specialization
        self._adapt_final_layers()
        
        # Train with reduced learning rate
        original_lr = self.config['training']['learning_rate']
        self.config['training']['learning_rate'] = original_lr * 0.1
        
        # Reinitialize optimizer with new learning rate
        self.optimizer = self._get_optimizer()
        
        # Train for specified epochs
        self.config['training']['epochs'] = epochs
        self.train(training_data, verbose=True)
        
        # Restore original learning rate
        self.config['training']['learning_rate'] = original_lr
    
    def _freeze_early_layers(self) -> None:
        """Freeze early network layers to preserve general knowledge."""
        # Freeze first few layers
        layers_to_freeze = list(self.model.network.children())[:3]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _adapt_final_layers(self) -> None:
        """Modify final layers for damage scenario specialization."""
        # This would add scenario-specific adaptation layers
        pass