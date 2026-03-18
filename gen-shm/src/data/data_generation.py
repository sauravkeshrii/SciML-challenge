"""
Synthetic data generation for Gen-SHM training and validation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.signal import chirp, welch

# Handle imports for both package and direct usage
try:
    from ..utils.config import Config
    from ..utils.helpers import get_device, sample_collocation_points
    from .beam_physics import AnalyticalBeamSolution
except ImportError:
    # Fallback for direct file execution
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent.parent / 'utils'
    models_path = Path(__file__).parent.parent / 'models'
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))
    from config import Config
    from helpers import get_device, sample_collocation_points
    from beam_physics import AnalyticalBeamSolution


class SyntheticDataGenerator:
    """
    Generate synthetic training and validation data for Gen-SHM.
    
    Creates:
    1. Healthy state calibration data (sparse sensor measurements)
    2. Collocation points for physics loss computation
    3. Boundary and initial condition points
    4. Validation datasets with known damage scenarios
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.device = get_device()
        self.analytical_solver = AnalyticalBeamSolution()
    
    def generate_healthy_calibration_data(self, 
                                        num_sensors: int = 3,
                                        duration: float = 2.0,
                                        sampling_rate: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Generate sparse calibration data from healthy wing response.
        
        Args:
            num_sensors: Number of virtual sensors along wing
            duration: Measurement duration (seconds)
            sampling_rate: Sampling frequency (Hz)
            
        Returns:
            Dictionary with calibration data tensors
        """
        # Get sensor locations
        sensor_locations = torch.tensor(
            self.config['data']['sensor_locations'][:num_sensors],
            device=self.device, dtype=torch.float32
        ) * self.config['physics']['beam_length']
        
        # Time vector
        time_steps = int(duration * sampling_rate)
        t = torch.linspace(0, duration, time_steps, device=self.device)
        
        # Material properties
        L = self.config['physics']['beam_length']
        E = self.config['physics']['young_modulus']
        I = (self.config['physics']['beam_width'] * 
             self.config['physics']['beam_height']**3) / 12
        rho = self.config['physics']['density']
        A = self.config['physics']['beam_width'] * self.config['physics']['beam_height']
        
        # Generate excitation signal (chirp sweep)
        frequencies = self.config['data']['frequency_range']
        excitation = chirp(t.cpu().numpy(), 
                          f0=frequencies[0], 
                          f1=frequencies[1], 
                          t1=duration, 
                          method='linear')
        excitation = torch.from_numpy(excitation).float().to(self.device)
        
        # Compute first mode response (dominant for practical cases)
        omega_n, mode_shape = self.analytical_solver.undamaged_beam_modes(L, E, I, rho, A, mode=1)
        
        # Modal response (assuming linear system)
        modal_mass = rho * A * L  # Approximate
        modal_stiffness = (omega_n**2) * modal_mass
        
        # Simple harmonic oscillator response
        omega_d = omega_n * torch.sqrt(torch.tensor(1 - 0.02**2))  # Damped frequency
        response_amplitude = 1.0 / torch.sqrt(torch.tensor((modal_stiffness - (omega_n**2 * modal_mass))**2 + 
                                            (2 * 0.02 * omega_n * modal_mass * omega_n)**2))
        
        # Generate response for each sensor
        displacements = []
        velocities = []
        accelerations = []
        
        for sensor_pos in sensor_locations:
            # Mode shape at sensor location
            phi_x = mode_shape(sensor_pos)
            
            # Response components
            cos_term = torch.cos(omega_d * t)
            sin_term = torch.sin(omega_d * t)
            
            # Displacement
            disp = response_amplitude * phi_x * excitation * torch.exp(-0.02 * omega_n * t) * cos_term
            displacements.append(disp)
            
            # Velocity (first derivative)
            vel = (response_amplitude * phi_x * excitation * 
                  torch.exp(-0.02 * omega_n * t) * 
                  (-0.02 * omega_n * cos_term - omega_d * sin_term))
            velocities.append(vel)
            
            # Acceleration (second derivative) - what sensors actually measure
            acc = (response_amplitude * phi_x * excitation * 
                  torch.exp(-0.02 * omega_n * t) *
                  ((0.02 * omega_n)**2 * cos_term + 
                   2 * 0.02 * omega_n * omega_d * sin_term -
                   omega_d**2 * cos_term))
            accelerations.append(acc)
        
        # Stack results
        displacement_data = torch.stack(displacements, dim=0)  # (sensors, time)
        velocity_data = torch.stack(velocities, dim=0)
        acceleration_data = torch.stack(accelerations, dim=0)
        
        # Add measurement noise
        noise_level = self.config['data']['noise_level']
        noise = torch.randn_like(acceleration_data) * noise_level * torch.abs(acceleration_data).mean()
        acceleration_data += noise
        
        return {
            'sensor_positions': sensor_locations,
            'time': t,
            'displacement': displacement_data,
            'velocity': velocity_data,
            'acceleration': acceleration_data,
            'excitation': excitation
        }
    
    def generate_collocation_points(self, 
                                  num_physics: int = 10000,
                                  num_boundary: int = 1000,
                                  num_initial: int = 500) -> Dict[str, torch.Tensor]:
        """
        Generate collocation points for physics-informed training.
        
        Args:
            num_physics: Number of interior collocation points
            num_boundary: Number of boundary condition points
            num_initial: Number of initial condition points
            
        Returns:
            Dictionary with collocation point tensors
        """
        L = self.config['physics']['beam_length']
        T = 1.0  # Training time horizon
        
        # Interior physics points
        physics_points = sample_collocation_points(
            x_range=(0, L),
            t_range=(0, T),
            num_points=num_physics,
            device=self.device
        )
        x_physics = physics_points[:, 0]
        t_physics = physics_points[:, 1]
        
        # Boundary points
        # Left boundary (x=0) and right boundary (x=L) at various times
        t_boundary = torch.rand(num_boundary, device=self.device) * T
        x_left_boundary = torch.zeros(num_boundary, device=self.device)
        x_right_boundary = torch.full((num_boundary,), L, device=self.device)
        
        boundary_points = {
            'x_left': x_left_boundary,
            'x_right': x_right_boundary,
            't': t_boundary
        }
        
        # Initial condition points (t=0)
        x_initial = torch.rand(num_initial, device=self.device) * L
        t_initial = torch.zeros(num_initial, device=self.device)
        
        return {
            'physics': {'x': x_physics, 't': t_physics},
            'boundary': boundary_points,
            'initial': {'x': x_initial, 't': t_initial}
        }
    
    def generate_damage_scenarios(self, 
                                num_scenarios: int = 50,
                                severity_range: Tuple[float, float] = (0.05, 0.4)) -> Dict[str, torch.Tensor]:
        """
        Generate random damage scenarios for training diversity.
        
        Args:
            num_scenarios: Number of damage scenarios to generate
            severity_range: Range of damage severity values
            
        Returns:
            Dictionary with damage parameter tensors
        """
        # Sample damage locations and severities
        location_min, location_max = self.config['damage']['location_range']
        severity_min, severity_max = severity_range
        
        damage_locations = (torch.rand(num_scenarios, device=self.device) * 
                           (location_max - location_min) + location_min)
        damage_severities = (torch.rand(num_scenarios, device=self.device) * 
                            (severity_max - severity_min) + severity_min)
        
        return {
            'locations': damage_locations,
            'severities': damage_severities
        }
    
    def generate_training_data(self) -> Dict[str, torch.Tensor]:
        """
        Generate complete training dataset.
        
        Returns:
            Dictionary containing all training data components
        """
        # Generate healthy calibration data
        healthy_data = self.generate_healthy_calibration_data(
            num_sensors=3,
            duration=1.0,
            sampling_rate=500  # Lower rate for training efficiency
        )
        
        # Generate collocation points
        collocation_points = self.generate_collocation_points(
            num_physics=self.config['training']['physics_points'],
            num_boundary=self.config['training']['boundary_points'],
            num_initial=self.config['training']['initial_condition_points']
        )
        
        # Generate damage scenarios
        damage_scenarios = self.generate_damage_scenarios(
            num_scenarios=100,
            severity_range=(0.0, self.config['damage']['max_severity'])
        )
        
        # Package training data
        training_data = {
            # Calibration data
            'x_data': healthy_data['sensor_positions'].repeat(len(healthy_data['time'])),
            't_data': healthy_data['time'].repeat_interleave(len(healthy_data['sensor_positions'])),
            'u_data': healthy_data['displacement'].transpose(0, 1).flatten(),  # (time*sensors,)
            
            # Collocation points
            'x_physics': collocation_points['physics']['x'],
            't_physics': collocation_points['physics']['t'],
            'x_boundary_left': collocation_points['boundary']['x_left'],
            'x_boundary_right': collocation_points['boundary']['x_right'],
            't_boundary': collocation_points['boundary']['t'],
            'x_initial': collocation_points['initial']['x'],
            't_initial': collocation_points['initial']['t'],
            
            # Damage scenarios
            'damage_locations': damage_scenarios['locations'],
            'damage_severities': damage_scenarios['severities'],
            
            # Additional metadata
            'healthy_data': healthy_data,
            'collocation_points': collocation_points
        }
        
        return training_data
    
    def generate_validation_dataset(self, 
                                  num_scenarios: int = 10,
                                  samples_per_scenario: int = 5) -> List[Dict]:
        """
        Generate validation dataset with known damage scenarios.
        
        Args:
            num_scenarios: Number of different damage scenarios
            samples_per_scenario: Samples per damage scenario
            
        Returns:
            List of validation sample dictionaries
        """
        validation_data = []
        
        # Define specific damage scenarios for validation
        scenarios = [
            {'location': 0.0, 'severity': 0.1, 'name': 'Root_light'},      # Light root damage
            {'location': 0.0, 'severity': 0.3, 'name': 'Root_severe'},     # Severe root damage
            {'location': 0.5, 'severity': 0.15, 'name': 'Center_moderate'}, # Moderate center damage
            {'location': 0.5, 'severity': 0.25, 'name': 'Center_severe'},   # Severe center damage
            {'location': 0.9, 'severity': 0.2, 'name': 'Tip_moderate'},     # Moderate tip damage
        ]
        
        # Add random scenarios to complete the requested number
        while len(scenarios) < num_scenarios:
            loc = torch.rand(1).item() * 0.8 + 0.1  # Avoid extremes
            sev = torch.rand(1).item() * 0.3 + 0.05  # 5-35% damage
            scenarios.append({
                'location': loc,
                'severity': sev,
                'name': f'Random_{len(scenarios)}'
            })
        
        # Generate samples for each scenario
        for scenario in scenarios[:num_scenarios]:
            for sample_idx in range(samples_per_scenario):
                # Generate synthetic measurement data
                # In practice, this would use the trained model
                # For now, we'll create placeholder data
                sample_data = {
                    'scenario_name': scenario['name'],
                    'damage_location': scenario['location'],
                    'damage_severity': scenario['severity'],
                    'sample_index': sample_idx,
                    'sensor_data': None,  # Would be filled with actual generated data
                    'ground_truth': {
                        'location': scenario['location'],
                        'severity': scenario['severity']
                    }
                }
                validation_data.append(sample_data)
        
        return validation_data


class SHMDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for SHM training data.
    """
    
    def __init__(self, training_data: Dict[str, torch.Tensor], batch_size: int = 1024):
        self.training_data = training_data
        self.batch_size = batch_size
        
        # Determine dataset size from smallest component
        sizes = [
            len(training_data['x_data']),
            len(training_data['x_physics']),
            len(training_data['damage_locations'])
        ]
        self.dataset_size = min(sizes)
    
    def __len__(self) -> int:
        return self.dataset_size // self.batch_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get batch of training data."""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.dataset_size)
        
        batch_data = {}
        for key, tensor in self.training_data.items():
            if isinstance(tensor, torch.Tensor) and len(tensor) >= self.dataset_size:
                batch_data[key] = tensor[start_idx:end_idx]
            else:
                # For smaller tensors (damage parameters), repeat to match batch size
                if isinstance(tensor, torch.Tensor):
                    repeats = (self.batch_size + len(tensor) - 1) // len(tensor)
                    repeated = tensor.repeat(repeats)[:self.batch_size]
                    batch_data[key] = repeated
                else:
                    batch_data[key] = tensor
        
        return batch_data


def create_data_loaders(training_data: Dict[str, torch.Tensor], 
                       batch_size: int = 1024,
                       shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader for training data.
    
    Args:
        training_data: Dictionary of training tensors
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Returns:
        Configured DataLoader
    """
    dataset = SHMDataset(training_data, batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Dataset handles batching internally
        shuffle=shuffle,
        num_workers=0  # Set to 0 for reproducibility
    )
    
    return dataloader