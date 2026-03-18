"""
Unit tests for Gen-SHM physics implementation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path properly
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
os.chdir(project_root)

from src.models.beam_physics import BeamPhysics, AnalyticalBeamSolution
from src.utils.config import Config


class TestBeamPhysics:
    """Test physics implementation."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = Config()
        self.physics = BeamPhysics(self.config.config)
    
    def test_stiffness_field_healthy(self):
        """Test stiffness field computation for healthy beam."""
        x = torch.tensor([0.0, 0.5, 1.0])  # Normalized positions
        damage_location = 0.5
        damage_severity = 0.0  # No damage
        
        stiffness = self.physics.stiffness_field(x, damage_location, damage_severity)
        
        # Should be constant (undamaged) stiffness
        assert torch.allclose(stiffness, torch.full_like(stiffness, self.physics.EI_0))
    
    def test_stiffness_field_damaged(self):
        """Test stiffness field with damage."""
        x = torch.tensor([0.5])  # Damage location
        damage_location = 0.5
        damage_severity = 0.2   # 20% damage
        
        stiffness = self.physics.stiffness_field(x, damage_location, damage_severity)
        
        # Should have reduced stiffness at damage location
        expected_stiffness = self.physics.EI_0 * (1.0 - damage_severity)
        assert torch.allclose(stiffness, torch.tensor(expected_stiffness))
    
    def test_physics_residual_zero_for_analytical_solution(self):
        """Test that analytical solution gives zero residual."""
        # For simple static case, zero displacement should satisfy physics
        x = torch.tensor([0.5], requires_grad=True)
        t = torch.tensor([0.0], requires_grad=True)
        u = torch.zeros_like(x)
        damage_param = torch.tensor([[0.5, 0.0]])  # No damage
        
        residual = self.physics.physics_residual(x, t, u, damage_param)
        
        # Residual should be small (not exactly zero due to numerical precision)
        assert torch.abs(residual).max() < 1e-10
    
    def test_boundary_conditions(self):
        """Test boundary condition computation."""
        x = torch.tensor([0.0, 1.0], requires_grad=True)  # Boundary points
        t = torch.tensor([0.5, 0.5], requires_grad=True)
        u = torch.tensor([0.0, 0.1])  # Simple displacement
        
        left_bc, right_bc = self.physics.boundary_conditions(x, t, u)
        
        # Should return finite residuals
        assert torch.isfinite(left_bc).all()
        assert torch.isfinite(right_bc).all()


class TestAnalyticalSolutions:
    """Test analytical beam solutions."""
    
    def test_undamaged_beam_modes(self):
        """Test analytical mode computation."""
        L = 1.0
        E = 70e9
        I = 1e-6
        rho = 2700
        A = 0.01
        
        omega_n, mode_shape = AnalyticalBeamSolution.undamaged_beam_modes(L, E, I, rho, A, mode=1)
        
        # Natural frequency should be positive and finite
        assert omega_n > 0
        assert torch.isfinite(omega_n)
        
        # Mode shape should return callable function
        x_test = torch.tensor([0.5])
        phi_value = mode_shape(x_test)
        assert torch.isfinite(phi_value).all()


if __name__ == '__main__':
    pytest.main([__file__])