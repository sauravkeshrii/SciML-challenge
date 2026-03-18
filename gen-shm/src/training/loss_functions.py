"""
Advanced loss functions for physics-informed training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from ..utils.config import config


class AdaptiveWeightScheduler:
    """
    Adaptive loss weight scheduler for balancing physics and data terms.
    
    Adjusts loss weights dynamically during training based on loss magnitudes.
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.weights = self.config['training']['loss_weights'].copy()
        self.adaptation_rate = 0.01  # How quickly to adapt weights
        
    def update_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss weights based on current loss values.
        
        Args:
            current_losses: Dictionary of current loss values
            
        Returns:
            Updated weight dictionary
        """
        if len(current_losses) < 2:
            return self.weights
            
        # Get loss values
        loss_values = list(current_losses.values())
        loss_keys = list(current_losses.keys())
        
        # Compute relative magnitudes
        total_loss = sum(loss_values)
        if total_loss == 0:
            return self.weights
            
        relative_magnitudes = [loss / total_loss for loss in loss_values]
        
        # Desired equal contribution (uniform distribution)
        target_magnitude = 1.0 / len(loss_values)
        
        # Update weights to balance contributions
        for i, key in enumerate(loss_keys):
            # Adjust weight proportionally to error from target
            error = relative_magnitudes[i] - target_magnitude
            adjustment = -self.adaptation_rate * error
            self.weights[key] *= (1.0 + adjustment)
            
            # Ensure weights stay positive and reasonable
            self.weights[key] = max(0.1, min(100.0, self.weights[key]))
        
        return self.weights.copy()


class PhysicsRegularizedLoss:
    """
    Enhanced physics-informed loss with regularization terms.
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.base_loss = PhysicsInformedLoss(config_dict)
        self.regularization_strength = 0.01
        
    def compute_regularized_loss(self, model, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute regularized loss with additional stability terms.
        
        Args:
            model: PINN model
            batch_data: Batch of training data
            
        Returns:
            Dictionary of loss components
        """
        # Compute base physics-informed loss
        losses = self.base_loss.compute_total_loss(
            model=model,
            x_data=batch_data['x_data'],
            t_data=batch_data['t_data'],
            u_data=batch_data['u_data'],
            damage_location=batch_data['damage_locations'],
            damage_severity=batch_data['damage_severities'],
            x_physics=batch_data['x_physics'],
            t_physics=batch_data['t_physics'],
            x_boundary=batch_data.get('x_boundary_left'),  # Use one boundary for simplicity
            t_boundary=batch_data.get('t_boundary'),
            x_initial=batch_data.get('x_initial'),
            t_initial=batch_data.get('t_initial')
        )
        
        # Add regularization terms
        regularization_loss = self._compute_regularization(model)
        losses['regularization'] = regularization_loss
        losses['total_loss'] += self.regularization_strength * regularization_loss
        
        return losses
    
    def _compute_regularization(self, model) -> torch.Tensor:
        """Compute model regularization loss."""
        reg_loss = 0.0
        
        # L2 regularization on weights
        for param in model.parameters():
            reg_loss += torch.sum(param**2)
            
        return reg_loss


class MultiScaleTraining:
    """
    Multi-scale training strategy for improved convergence.
    
    Starts with coarse resolution and gradually increases detail.
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.current_scale = 1
        self.max_scale = 4
        self.scale_epochs = 50  # Epochs per scale
        
    def get_current_resolution(self) -> Dict[str, int]:
        """Get current training resolution parameters."""
        base_points = self.config['training']
        
        scale_factor = 2**(self.current_scale - 1)
        
        return {
            'physics_points': base_points['physics_points'] // scale_factor,
            'boundary_points': base_points['boundary_points'] // scale_factor,
            'initial_points': base_points['initial_condition_points'] // scale_factor,
            'spatial_resolution': self.config['data']['spatial_points'] // scale_factor,
            'temporal_resolution': self.config['data']['temporal_points'] // scale_factor
        }
    
    def should_increase_scale(self, epoch: int) -> bool:
        """Check if we should move to the next scale."""
        return (epoch > 0 and epoch % self.scale_epochs == 0 and 
                self.current_scale < self.max_scale)
    
    def increase_scale(self) -> None:
        """Move to the next training scale."""
        if self.current_scale < self.max_scale:
            self.current_scale += 1


# Import the base PhysicsInformedLoss from pinn_generator
from ..models.pinn_generator import PhysicsInformedLoss

class AdvancedTrainerComponents:
    """
    Collection of advanced training components.
    """
    
    def __init__(self):
        self.adaptive_scheduler = AdaptiveWeightScheduler()
        self.multi_scale = MultiScaleTraining()
        self.physics_regularizer = PhysicsRegularizedLoss()