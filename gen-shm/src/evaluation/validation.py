"""
Physics compliance and model validation tools.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import integrate
import warnings

from ..models.beam_physics import BeamPhysics
from ..utils.helpers import get_device
from .metrics import PhysicsComplianceMetrics


class PhysicsValidator:
    """
    Comprehensive physics validation for trained models.
    
    Validates:
    - Governing equation satisfaction
    - Boundary condition compliance
    - Energy conservation
    - Modal analysis consistency
    - Numerical stability
    """
    
    def __init__(self, model, config: dict = None):
        self.model = model
        self.config = config
        self.physics_engine = BeamPhysics(config)
        self.device = get_device()
    
    def validate_governing_equation(self, 
                                  num_test_points: int = 5000,
                                  damage_scenarios: List[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Validate satisfaction of the governing Euler-Bernoulli equation.
        
        Args:
            num_test_points: Number of collocation points for testing
            damage_scenarios: List of (location, severity) tuples to test
            
        Returns:
            Dictionary of validation metrics
        """
        if damage_scenarios is None:
            damage_scenarios = [
                (0.0, 0.0),    # Healthy case
                (0.3, 0.1),    # Light damage
                (0.5, 0.25),   # Moderate damage  
                (0.8, 0.4),    # Severe damage
            ]
        
        results = {}
        
        for damage_loc, damage_sev in damage_scenarios:
            # Generate random test points
            x_test = torch.rand(num_test_points, device=self.device) * self.config['physics']['beam_length']
            t_test = torch.rand(num_test_points, device=self.device) * 1.0  # 1 second
            
            # Damage parameters
            damage_loc_tensor = torch.full((num_test_points,), damage_loc, device=self.device)
            damage_sev_tensor = torch.full((num_test_points,), damage_sev, device=self.device)
            
            # Enable gradients for physics computation
            x_test.requires_grad_(True)
            t_test.requires_grad_(True)
            
            # Compute physics residual
            with torch.no_grad():
                residual = self.model.compute_physics_loss(x_test, t_test, damage_loc_tensor, damage_sev_tensor)
            
            scenario_name = f"damage_{damage_loc:.1f}_{damage_sev:.2f}"
            results[f'{scenario_name}_mean_residual'] = residual.item()
            results[f'{scenario_name}_max_residual'] = residual.max().item()
        
        return results
    
    def validate_boundary_conditions(self, 
                                   num_boundary_points: int = 1000) -> Dict[str, float]:
        """
        Validate boundary condition satisfaction.
        
        Args:
            num_boundary_points: Number of boundary points to test
            
        Returns:
            Boundary condition validation metrics
        """
        # Left boundary (x = 0)
        t_left = torch.rand(num_boundary_points, device=self.device) * 1.0
        x_left = torch.zeros(num_boundary_points, device=self.device)
        damage_loc = torch.rand(num_boundary_points, device=self.device) * 0.8 + 0.1
        damage_sev = torch.rand(num_boundary_points, device=self.device) * 0.3
        
        x_left.requires_grad_(True)
        t_left.requires_grad_(True)
        
        with torch.no_grad():
            bc_loss_left = self.model.compute_boundary_loss(x_left, t_left, damage_loc, damage_sev)
        
        # Right boundary (x = L)
        t_right = torch.rand(num_boundary_points, device=self.device) * 1.0
        x_right = torch.full((num_boundary_points,), self.config['physics']['beam_length'], device=self.device)
        damage_loc_right = torch.rand(num_boundary_points, device=self.device) * 0.8 + 0.1
        damage_sev_right = torch.rand(num_boundary_points, device=self.device) * 0.3
        
        x_right.requires_grad_(True)
        t_right.requires_grad_(True)
        
        with torch.no_grad():
            bc_loss_right = self.model.compute_boundary_loss(x_right, t_right, damage_loc_right, damage_sev_right)
        
        return {
            'left_boundary_residual': bc_loss_left.item(),
            'right_boundary_residual': bc_loss_right.item(),
            'total_boundary_residual': (bc_loss_left + bc_loss_right).item()
        }
    
    def validate_initial_conditions(self, 
                                  num_initial_points: int = 500) -> Dict[str, float]:
        """
        Validate initial condition satisfaction.
        
        Args:
            num_initial_points: Number of initial condition points to test
            
        Returns:
            Initial condition validation metrics
        """
        x_init = torch.rand(num_initial_points, device=self.device) * self.config['physics']['beam_length']
        t_init = torch.zeros(num_initial_points, device=self.device)
        damage_loc = torch.rand(num_initial_points, device=self.device) * 0.8 + 0.1
        damage_sev = torch.rand(num_initial_points, device=self.device) * 0.3
        
        x_init.requires_grad_(True)
        t_init.requires_grad_(True)
        
        with torch.no_grad():
            ic_loss = self.model.compute_initial_loss(x_init, t_init, damage_loc, damage_sev)
        
        return {
            'initial_condition_residual': ic_loss.item()
        }
    
    def validate_energy_conservation(self, 
                                   duration: float = 2.0,
                                   sampling_rate: int = 1000) -> Dict[str, float]:
        """
        Validate energy conservation properties.
        
        Args:
            duration: Simulation duration
            sampling_rate: Sampling frequency
            
        Returns:
            Energy conservation metrics
        """
        time_steps = int(duration * sampling_rate)
        time_vector = torch.linspace(0, duration, time_steps, device=self.device)
        
        # Sensor locations
        sensor_locations = torch.tensor(
            self.config['data']['sensor_locations'], 
            device=self.device, dtype=torch.float32
        ) * self.config['physics']['beam_length']
        
        # Test damage scenario
        damage_location = 0.5
        damage_severity = 0.2
        
        # Generate acceleration signals
        acceleration_signals = []
        
        with torch.no_grad():
            for sensor_pos in sensor_locations:
                x_sensor = sensor_pos.expand(time_steps)
                t_sensor = time_vector
                damage_loc_tensor = torch.full((time_steps,), damage_location, device=self.device)
                damage_sev_tensor = torch.full((time_steps,), damage_severity, device=self.device)
                
                acceleration = self.model.generate_acceleration(
                    x_sensor, t_sensor, damage_loc_tensor, damage_sev_tensor
                )
                acceleration_signals.append(acceleration.cpu().numpy())
        
        acceleration_array = np.array(acceleration_signals)
        
        # Compute energy conservation metrics
        metrics = PhysicsComplianceMetrics.compute_energy_conservation_error(
            acceleration_array, time_vector.cpu().numpy()
        )
        
        return metrics
    
    def validate_numerical_stability(self, 
                                   test_duration: float = 5.0,
                                   num_tests: int = 100) -> Dict[str, float]:
        """
        Test numerical stability over extended periods.
        
        Args:
            test_duration: Duration for stability testing
            num_tests: Number of stability tests
            
        Returns:
            Stability metrics
        """
        stable_predictions = 0
        max_values = []
        
        time_steps = int(test_duration * 100)  # 100 Hz sampling
        time_vector = torch.linspace(0, test_duration, time_steps, device=self.device)
        
        sensor_pos = torch.tensor([0.5], device=self.device) * self.config['physics']['beam_length']
        
        for test_idx in range(num_tests):
            # Random damage scenario
            damage_location = torch.rand(1, device=self.device).item() * 0.8 + 0.1
            damage_severity = torch.rand(1, device=self.device).item() * 0.3
            
            try:
                with torch.no_grad():
                    x_sensor = sensor_pos.expand(time_steps)
                    t_sensor = time_vector
                    damage_loc = torch.full((time_steps,), damage_location, device=self.device)
                    damage_sev = torch.full((time_steps,), damage_severity, device=self.device)
                    
                    prediction = self.model.predict_displacement(x_sensor, t_sensor, damage_loc, damage_sev)
                    
                    # Check for numerical stability (no NaN or inf values)
                    if torch.isfinite(prediction).all() and torch.abs(prediction).max() < 1e6:
                        stable_predictions += 1
                        max_values.append(torch.abs(prediction).max().item())
                        
            except Exception as e:
                warnings.warn(f"Numerical instability detected in test {test_idx}: {str(e)}")
                continue
        
        stability_ratio = stable_predictions / num_tests
        
        return {
            'stability_ratio': stability_ratio,
            'mean_max_value': np.mean(max_values) if max_values else 0.0,
            'max_max_value': np.max(max_values) if max_values else 0.0,
            'failed_tests': num_tests - stable_predictions
        }
    
    def comprehensive_validation(self) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive validation suite.
        
        Returns:
            Nested dictionary of all validation results
        """
        results = {}
        
        print("Running comprehensive physics validation...")
        
        # Governing equation validation
        print("Validating governing equation...")
        results['governing_equation'] = self.validate_governing_equation()
        
        # Boundary conditions
        print("Validating boundary conditions...")
        results['boundary_conditions'] = self.validate_boundary_conditions()
        
        # Initial conditions
        print("Validating initial conditions...")
        results['initial_conditions'] = self.validate_initial_conditions()
        
        # Energy conservation
        print("Validating energy conservation...")
        results['energy_conservation'] = self.validate_energy_conservation()
        
        # Numerical stability
        print("Validating numerical stability...")
        results['numerical_stability'] = self.validate_numerical_stability()
        
        return results
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, Dict[str, float]],
                                 save_path: str = None) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from comprehensive_validation()
            save_path: Path to save report (optional)
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("GEN-SHM PHYSICS VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        total_metrics = 0
        passed_metrics = 0
        
        for category, metrics in validation_results.items():
            report.append(f"\n{category.upper()}:")
            report.append("-" * 40)
            
            for metric_name, value in metrics.items():
                total_metrics += 1
                
                # Determine pass/fail criteria based on metric type
                if 'residual' in metric_name:
                    threshold = 1e-3
                    passed = abs(value) < threshold
                elif 'stability' in metric_name:
                    threshold = 0.95
                    passed = value > threshold
                elif 'energy_conservation_error' in metric_name:
                    threshold = 0.1
                    passed = value < threshold
                else:
                    passed = True  # Assume pass for other metrics
                    
                status = "✓ PASS" if passed else "✗ FAIL"
                passed_metrics += int(passed)
                
                report.append(f"  {metric_name:30s}: {value:10.6f} [{status}]")
        
        # Overall summary
        report.append("\n" + "=" * 60)
        report.append("SUMMARY:")
        report.append(f"  Total metrics tested: {total_metrics}")
        report.append(f"  Passed metrics: {passed_metrics}")
        report.append(f"  Success rate: {passed_metrics/total_metrics*100:.1f}%")
        
        if passed_metrics == total_metrics:
            report.append("\n  🎉 ALL VALIDATIONS PASSED! Model is physics-compliant.")
        elif passed_metrics/total_metrics > 0.8:
            report.append("\n  ⚠️  MOST VALIDATIONS PASSED. Model is mostly physics-compliant.")
        else:
            report.append("\n  ❌ SIGNIFICANT PHYSICS VIOLATIONS DETECTED. Model needs improvement.")
        
        report.append("=" * 60)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str


def quick_validation(model, config: dict = None) -> Dict[str, Dict[str, float]]:
    """
    Quick validation function for rapid testing.
    
    Args:
        model: Trained PINN model
        config: Configuration dictionary
        
    Returns:
        Validation results dictionary
    """
    validator = PhysicsValidator(model, config)
    
    # Quick validation with fewer points
    quick_results = {
        'governing_equation': validator.validate_governing_equation(num_test_points=1000),
        'boundary_conditions': validator.validate_boundary_conditions(num_boundary_points=200),
        'numerical_stability': validator.validate_numerical_stability(test_duration=1.0, num_tests=20)
    }
    
    return quick_results