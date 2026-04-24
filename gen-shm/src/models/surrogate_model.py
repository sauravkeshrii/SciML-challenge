"""
High-level surrogate model interface for Gen-SHM system.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from .pinn_generator import PINNGenerator
from .beam_physics import BeamPhysics
from ..utils.config import config
from ..utils.helpers import get_device, set_seed
from ..data.data_generation import SyntheticDataGenerator


class DroneWingSurrogate:
    """
    High-level interface for the drone wing structural health monitoring surrogate.
    
    This class provides a user-friendly interface for:
    1. Generating synthetic vibration data for arbitrary damage scenarios
    2. Training the underlying PINN model
    3. Performing damage detection inference
    4. Validating physics compliance
    """
    
    def __init__(self, model_path: str = None, config_dict: dict = None):
        """
        Initialize the drone wing surrogate.
        
        Args:
            model_path: Path to pretrained model weights (optional)
            config_dict: Configuration dictionary (uses global config if None)
        """
        self.config = config_dict or config.config
        self.device = get_device()
        
        # Initialize components
        self.pinn_model = PINNGenerator(self.config)
        self.physics_engine = BeamPhysics(self.config)
        self.data_generator = SyntheticDataGenerator(self.config)
        
        # Load pretrained model if specified
        if model_path:
            self.load_model(model_path)
        
        self.is_trained = model_path is not None
    
    def generate_samples(self, 
                        damage_level: float,
                        damage_location: float = 0.5,
                        num_samples: int = 100,
                        duration: float = 1.0,
                        sampling_rate: int = 1000) -> Dict[str, np.ndarray]:
      
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating samples. "
                             "Call train() method first.")
        
        # Validate inputs
        if not 0.0 <= damage_level <= 1.0:
            raise ValueError("Damage level must be between 0.0 and 1.0")
        if not 0.0 <= damage_location <= 1.0:
            raise ValueError("Damage location must be between 0.0 and 1.0")
        
        # Get sensor locations from config
        sensor_locations = torch.tensor(
            self.config['data']['sensor_locations'], 
            device=self.device, dtype=torch.float32
        )
        num_sensors = len(sensor_locations)
        
        # Create time vector
        time_steps = int(duration * sampling_rate)
        time_vector = torch.linspace(0, duration, time_steps, device=self.device)
        
        # Initialize output arrays
        acceleration_data = np.zeros((num_samples, num_sensors, time_steps))
        
        # Generate samples
        self.pinn_model.eval()
        with torch.no_grad():
            for sample_idx in range(num_samples):
                # Sample random initial conditions for diversity
                torch.manual_seed(sample_idx)  # For reproducible randomness
                
                # Generate acceleration for each sensor
                for sensor_idx, sensor_pos in enumerate(sensor_locations):
                    # Repeat sensor position for all time steps
                    x_sensor = sensor_pos.expand(time_steps)
                    t_sensor = time_vector
                    
                    # Repeat damage parameters
                    damage_loc_tensor = torch.full((time_steps,), damage_location, device=self.device)
                    damage_sev_tensor = torch.full((time_steps,), damage_level, device=self.device)
                    
                    # Generate acceleration
                    acceleration = self.pinn_model.generate_acceleration(
                        x_sensor, t_sensor, damage_loc_tensor, damage_sev_tensor
                    )
                    
                    # Convert to numpy and store
                    acceleration_data[sample_idx, sensor_idx, :] = acceleration.cpu().numpy().flatten()
        
        return {
            'acceleration': acceleration_data,
            'time': time_vector.cpu().numpy(),
            'sensor_positions': sensor_locations.cpu().numpy(),
            'damage_info': {
                'level': damage_level,
                'location': damage_location,
                'num_samples': num_samples
            }
        }
    
    def train(self, training_config: dict = None, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the PINN model on synthetic data.
        
        Args:
            training_config: Training configuration (overrides default if provided)
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary with loss values
        """
        from ..training.trainer import PINNTrainer
        
        # Use provided config or default
        if training_config:
            # Merge with existing config
            for key, value in training_config.items():
                self.config['training'][key] = value
        
        # Initialize trainer
        trainer = PINNTrainer(self.pinn_model, self.config)
        
        # Generate synthetic training data
        if verbose:
            print("Generating synthetic training data...")
        
        train_data = self.data_generator.generate_training_data()
        
        # Train model
        if verbose:
            print("Starting training...")
            
        history = trainer.train(train_data, verbose=verbose)
        
        self.is_trained = True
        return history
    
    def detect_damage(self, acceleration_data: np.ndarray) -> Dict[str, float]:
        """
        Perform damage detection on measured acceleration data.
        
        Args:
            acceleration_data: Measured acceleration (num_sensors, time_steps)
            
        Returns:
            Damage assessment dictionary with likelihood scores
        """
        # TODO: Implement damage detection using trained classifier
        # This would typically involve:
        # 1. Feature extraction from acceleration data
        # 2. Classification using trained edge model
        # 3. Uncertainty quantification
        
        # Placeholder implementation
        return {
            'damage_detected': False,
            'confidence': 0.0,
            'estimated_severity': 0.0,
            'estimated_location': 0.5
        }
    
    def validate_physics_compliance(self, num_test_points: int = 1000) -> Dict[str, float]:
        """
        Validate that the trained model satisfies physics constraints.
        
        Args:
            num_test_points: Number of collocation points for validation
            
        Returns:
            Physics compliance metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before validation.")
        
        self.pinn_model.eval()
        
        # Generate test points
        x_test = torch.rand(num_test_points, device=self.device) * self.config['physics']['beam_length']
        t_test = torch.rand(num_test_points, device=self.device) * 1.0  # 1 second duration
        
        # Test various damage scenarios
        damage_scenarios = [
            (0.0, 0.0),  # Healthy
            (0.3, 0.5),  # Moderate damage at center
            (0.7, 0.2),  # Severe damage near root
        ]
        
        compliance_metrics = {}
        
        with torch.no_grad():
            for damage_loc, damage_sev in damage_scenarios:
                damage_loc_tensor = torch.full((num_test_points,), damage_loc, device=self.device)
                damage_sev_tensor = torch.full((num_test_points,), damage_sev, device=self.device)
                
                # Compute physics residual
                residual = self.pinn_model.compute_physics_loss(
                    x_test, t_test, damage_loc_tensor, damage_sev_tensor
                )
                
                scenario_name = f"damage_{damage_loc:.1f}_{damage_sev:.1f}"
                compliance_metrics[f'{scenario_name}_residual_mean'] = residual.item()
                compliance_metrics[f'{scenario_name}_residual_max'] = residual.max().item()
        
        return compliance_metrics
    
    def save_model(self, path: str) -> None:
        """Save model weights and configuration."""
        checkpoint = {
            'model_state_dict': self.pinn_model.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: str) -> None:
        """Load model weights and configuration."""
        checkpoint = torch.load(path, map_location=self.device)
        self.pinn_model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        
        # Update config if stored
        if 'config' in checkpoint:
            self.config = checkpoint['config']
    
    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model information."""
        return {
            'pinn_info': self.pinn_model.get_model_info(),
            'physics_info': {
                'beam_length': self.config['physics']['beam_length'],
                'material_properties': {
                    'young_modulus': self.config['physics']['young_modulus'],
                    'density': self.config['physics']['density']
                }
            },
            'data_info': {
                'sensor_locations': self.config['data']['sensor_locations'],
                'sampling_rate': 1000  # Hz
            },
            'trained': self.is_trained
        }


# Convenience functions for common use cases
def quick_train_and_generate(damage_level: float = 0.2, 
                           damage_location: float = 0.0,
                           num_samples: int = 50) -> Dict[str, np.ndarray]:
    """
    Quick function to train model and generate samples in one call.
    
    Args:
        damage_level: Damage severity (0.0 to 1.0)
        damage_location: Damage location (0.0 to 1.0)  
        num_samples: Number of samples to generate
        
    Returns:
        Generated acceleration data dictionary
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize surrogate
    surrogate = DroneWingSurrogate()
    
    # Train model
    print("Training model...")
    surrogate.train(verbose=True)
    
    # Generate samples
    print(f"Generating {num_samples} samples with {damage_level*100:.0f}% damage at location {damage_location:.1f}")
    samples = surrogate.generate_samples(
        damage_level=damage_level,
        damage_location=damage_location,
        num_samples=num_samples
    )
    
    return samples


def demo_damage_scenarios() -> List[Dict[str, np.ndarray]]:
    """
    Generate samples for multiple damage scenarios for demonstration.
    
    Returns:
        List of sample dictionaries for different damage scenarios
    """
    scenarios = [
        {'level': 0.0, 'location': 0.5, 'name': 'Healthy'},
        {'level': 0.1, 'location': 0.2, 'name': 'Light_root_damage'},
        {'level': 0.2, 'location': 0.5, 'name': 'Moderate_center_damage'},
        {'level': 0.3, 'location': 0.8, 'name': 'Severe_tip_damage'}
    ]
    
    results = []
    surrogate = DroneWingSurrogate()
    surrogate.train(verbose=False)  # Silent training
    
    for scenario in scenarios:
        samples = surrogate.generate_samples(
            damage_level=scenario['level'],
            damage_location=scenario['location'],
            num_samples=25
        )
        samples['scenario_name'] = scenario['name']
        results.append(samples)
    
    return results
