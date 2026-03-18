"""
Physics-Informed Neural Network (PINN) generator for drone wing vibration modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

# Handle imports for both package and direct usage
try:
    from .beam_physics import BeamPhysics
    from ..utils.config import Config
    from ..utils.helpers import get_device, count_parameters
except ImportError:
    # Fallback for direct file execution
    import sys
    from pathlib import Path
    models_path = Path(__file__).parent
    utils_path = Path(__file__).parent.parent / 'utils'
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from beam_physics import BeamPhysics
    from config import Config
    from helpers import get_device, count_parameters


class Swish(nn.Module):
    """Swish activation function: f(x) = x * sigmoid(x)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connections for improved gradient flow."""
    
    def __init__(self, dim: int, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.linear1(x)))
        out = self.norm2(self.linear2(out))
        return self.activation(out + residual)


class PINNGenerator(nn.Module):
    """
    Physics-Informed Neural Network generator for drone wing vibration.
    
    Takes inputs [x, t, damage_location, damage_severity] and outputs displacement u(x,t).
    Embeds physics constraints through loss function during training.
    """
    
    def __init__(self, config_dict: dict = None):
        """
        Initialize PINN generator.
        
        Args:
            config_dict: Configuration dictionary (uses global config if None)
        """
        super().__init__()
        self.config = config_dict or config.config
        self.model_config = self.config['model']
        self.physics_engine = BeamPhysics(config_dict)
        
        # Network architecture parameters
        self.input_dim = self.model_config['input_dim']
        self.output_dim = self.model_config['output_dim']
        self.hidden_dim = self.model_config['hidden_dim']
        self.num_layers = self.model_config['hidden_layers']
        self.dropout_rate = self.model_config['dropout_rate']
        
        # Select activation function
        activation_name = self.model_config['activation'].lower()
        if activation_name == 'swish':
            self.activation = Swish()
        elif activation_name == 'silu':
            self.activation = nn.SiLU()
        elif activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")
        
        # Build network
        self.network = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(get_device())
    
    def _build_network(self) -> nn.Sequential:
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(self.activation)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden residual blocks
        for _ in range(self.num_layers):
            layers.append(ResidualBlock(self.hidden_dim, self.activation))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PINN.
        
        Args:
            x: Spatial coordinates (batch_size,)
            t: Temporal coordinates (batch_size,)  
            damage_location: Damage location parameter (batch_size,)
            damage_severity: Damage severity parameter (batch_size,)
            
        Returns:
            Predicted displacement field u(x,t) with shape (batch_size, 1)
        """
        # Concatenate inputs
        inputs = torch.stack([x, t, damage_location, damage_severity], dim=1)
        
        # Forward through network
        output = self.network(inputs)
        
        return output
    
    def predict_displacement(self, x: torch.Tensor, t: torch.Tensor,
                           damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for displacement prediction.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            damage_location: Damage location [0,1]
            damage_severity: Damage severity [0,1]
            
        Returns:
            Displacement predictions
        """
        return self.forward(x, t, damage_location, damage_severity)
    
    def compute_physics_loss(self, x: torch.Tensor, t: torch.Tensor,
                           damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss using automatic differentiation.
        
        Args:
            x: Spatial coordinates with gradients enabled
            t: Temporal coordinates with gradients enabled
            damage_location: Damage location parameters
            damage_severity: Damage severity parameters
            
        Returns:
            Physics residual loss
        """
        # Require gradients for physics computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Forward pass
        u_pred = self.forward(x, t, damage_location, damage_severity)
        
        # Stack damage parameters
        damage_param = torch.stack([damage_location, damage_severity], dim=1)
        
        # Compute physics residual
        residual = self.physics_engine.physics_residual(x, t, u_pred, damage_param)
        
        # Physics loss is mean squared residual
        physics_loss = torch.mean(residual**2)
        
        return physics_loss
    
    def compute_boundary_loss(self, x_bound: torch.Tensor, t_bound: torch.Tensor,
                            damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            x_bound: Boundary spatial coordinates
            t_bound: Boundary temporal coordinates
            damage_location: Damage parameters
            damage_severity: Damage parameters
            
        Returns:
            Boundary condition loss
        """
        x_bound.requires_grad_(True)
        t_bound.requires_grad_(True)
        
        u_bound = self.forward(x_bound, t_bound, damage_location, damage_severity)
        damage_param = torch.stack([damage_location, damage_severity], dim=1)
        
        left_bc, right_bc = self.physics_engine.boundary_conditions(x_bound, t_bound, u_bound)
        
        # Mean squared boundary condition residuals
        boundary_loss = torch.mean(left_bc**2) + torch.mean(right_bc**2)
        
        return boundary_loss
    
    def compute_initial_loss(self, x_init: torch.Tensor, t_init: torch.Tensor,
                           damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            x_init: Initial spatial coordinates
            t_init: Initial temporal coordinates (should be zeros)
            damage_location: Damage parameters
            damage_severity: Damage parameters
            
        Returns:
            Initial condition loss
        """
        x_init.requires_grad_(True)
        t_init.requires_grad_(True)
        
        u_init = self.forward(x_init, t_init, damage_location, damage_severity)
        damage_param = torch.stack([damage_location, damage_severity], dim=1)
        
        init_disp, init_vel = self.physics_engine.initial_conditions(x_init, t_init, u_init)
        
        # Mean squared initial condition residuals
        initial_loss = torch.mean(init_disp**2) + torch.mean(init_vel**2)
        
        return initial_loss
    
    def generate_acceleration(self, x: torch.Tensor, t: torch.Tensor,
                            damage_location: torch.Tensor, damage_severity: torch.Tensor) -> torch.Tensor:
        """
        Generate acceleration time history from displacement predictions.
        
        Args:
            x: Spatial coordinates (sensor locations)
            t: Temporal coordinates
            damage_location: Damage parameters
            damage_severity: Damage parameters
            
        Returns:
            Acceleration time history
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Predict displacement
        u = self.forward(x, t, damage_location, damage_severity)
        
        # Compute acceleration (second time derivative)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0]
        
        return u_tt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = count_parameters(self)
        
        return {
            'model_type': 'PINN Generator',
            'input_dimension': self.input_dim,
            'output_dimension': self.output_dim,
            'hidden_layers': self.num_layers,
            'hidden_dimension': self.hidden_dim,
            'total_parameters': total_params,
            'activation_function': self.model_config['activation'],
            'device': next(self.parameters()).device
        }


class PhysicsInformedLoss:
    """
    Composite loss function combining data fidelity and physics constraints.
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.loss_weights = self.config['training']['loss_weights']
    
    def compute_total_loss(self, model: PINNGenerator,
                          x_data: torch.Tensor, t_data: torch.Tensor,
                          u_data: torch.Tensor,
                          damage_location: torch.Tensor, damage_severity: torch.Tensor,
                          x_physics: torch.Tensor, t_physics: torch.Tensor,
                          x_boundary: torch.Tensor = None, t_boundary: torch.Tensor = None,
                          x_initial: torch.Tensor = None, t_initial: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute total composite loss.
        
        Args:
            model: PINN generator model
            x_data, t_data, u_data: Calibration data
            damage_location, damage_severity: Damage parameters
            x_physics, t_physics: Collocation points for physics loss
            x_boundary, t_boundary: Boundary points (optional)
            x_initial, t_initial: Initial condition points (optional)
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Data fidelity loss
        u_pred_data = model.predict_displacement(x_data, t_data, damage_location, damage_severity)
        data_loss = torch.mean((u_pred_data - u_data)**2)
        losses['data_loss'] = data_loss
        
        # Physics loss
        physics_loss = model.compute_physics_loss(x_physics, t_physics, damage_location, damage_severity)
        losses['physics_loss'] = physics_loss
        
        # Boundary condition loss (if provided)
        if x_boundary is not None and t_boundary is not None:
            boundary_loss = model.compute_boundary_loss(x_boundary, t_boundary, damage_location, damage_severity)
            losses['boundary_loss'] = boundary_loss
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=x_data.device)
        
        # Initial condition loss (if provided)
        if x_initial is not None and t_initial is not None:
            initial_loss = model.compute_initial_loss(x_initial, t_initial, damage_location, damage_severity)
            losses['initial_loss'] = initial_loss
        else:
            losses['initial_loss'] = torch.tensor(0.0, device=x_data.device)
        
        # Weighted total loss
        total_loss = (self.loss_weights['data'] * data_loss +
                     self.loss_weights['physics'] * physics_loss +
                     self.loss_weights['boundary'] * losses['boundary_loss'])
        
        losses['total_loss'] = total_loss
        
        return losses