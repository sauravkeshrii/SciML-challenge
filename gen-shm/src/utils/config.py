"""
Configuration management for Gen-SHM project.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for the Gen-SHM project."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            # Physics parameters
            'physics': {
                'beam_length': 1.0,  # meters
                'beam_width': 0.1,   # meters
                'beam_height': 0.02, # meters
                'young_modulus': 70e9,  # Pa (aluminum)
                'density': 2700,        # kg/m³
                'damping_coefficient': 10.0,
                'boundary_conditions': {
                    'left': 'clamped',   # x=0
                    'right': 'free'      # x=L
                }
            },
            
            # Damage parameters
            'damage': {
                'min_severity': 0.0,
                'max_severity': 0.5,  # 50% stiffness reduction
                'location_range': [0.1, 0.9],  # normalized position
                'damage_function': 'gaussian'  # gaussian or step
            },
            
            # Model architecture
            'model': {
                'input_dim': 4,      # x, t, damage_location, damage_severity
                'output_dim': 1,     # displacement
                'hidden_layers': 6,
                'hidden_dim': 128,
                'activation': 'swish',
                'dropout_rate': 0.0
            },
            
            # Training parameters
            'training': {
                'epochs': 1000,
                'batch_size': 1024,
                'learning_rate': 1e-3,
                'optimizer': 'adam',
                'lr_scheduler': 'cosine_annealing',
                'loss_weights': {
                    'data': 1.0,
                    'physics': 10.0,
                    'boundary': 100.0
                },
                'physics_points': 10000,
                'boundary_points': 1000,
                'initial_condition_points': 500
            },
            
            # Data generation
            'data': {
                'spatial_points': 100,
                'temporal_points': 200,
                'sensor_locations': [0.25, 0.5, 0.75],  # normalized positions
                'noise_level': 0.01,  # 1% noise
                'frequency_range': [10, 1000]  # Hz
            },
            
            # Paths
            'paths': {
                'data_dir': 'data/',
                'checkpoints_dir': 'checkpoints/',
                'logs_dir': 'logs/',
                'results_dir': 'results/'
            }
        }
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Global configuration instance
config = Config()