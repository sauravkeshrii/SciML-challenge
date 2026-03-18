"""
Physics implementation for Gen-SHM: Euler-Bernoulli beam theory with damage modeling.
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional

# Handle imports for both package and direct usage
try:
    from ..utils.config import Config
    from ..utils.helpers import compute_derivative
except ImportError:
    # Fallback for direct file execution
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent.parent / 'utils'
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from config import Config
    from helpers import compute_derivative


class BeamPhysics:
    """
    Physics engine for Euler-Bernoulli beam with spatially varying stiffness.
    
    Governing equation:
    ρA ∂²u/∂t² + c ∂u/∂t + ∂²/∂x²[EI(x;d) ∂²u/∂x²] = 0
    
    Where:
    - u(x,t): vertical displacement
    - ρA: mass per unit length
    - c: damping coefficient
    - EI(x;d): spatially varying flexural rigidity with damage parameter d
    """
    
    def __init__(self, config_dict: dict = None):
        """
        Initialize beam physics engine.
        
        Args:
            config_dict: Configuration dictionary (uses global config if None)
        """
        self.config = config_dict or config.config
        self.physics_params = self.config['physics']
        self.damage_params = self.config['damage']
        
        # Physical constants
        self.L = self.physics_params['beam_length']  # beam length
        self.rho = self.physics_params['density']    # density
        self.E = self.physics_params['young_modulus']  # Young's modulus
        
        # Geometric properties
        self.A = (self.physics_params['beam_width'] * 
                 self.physics_params['beam_height'])  # cross-sectional area
        self.I = (self.physics_params['beam_width'] * 
                 self.physics_params['beam_height']**3) / 12  # second moment of area
        
        # Mass per unit length and damping
        self.rho_A = self.rho * self.A
        self.c = self.physics_params['damping_coefficient']
        
        # Initial undamaged stiffness
        self.EI_0 = self.E * self.I
        
        # Damage function
        self.damage_function = self._get_damage_function()
    
    def _get_damage_function(self) -> Callable:
        """Get damage influence function based on configuration."""
        damage_type = self.damage_params['damage_function']
        
        if damage_type == 'gaussian':
            def gaussian_damage(x_norm: torch.Tensor, location: float, severity: float) -> torch.Tensor:
                """Gaussian damage influence function."""
                sigma = 0.1  # width parameter
                return severity * torch.exp(-((x_norm - location)**2) / (2 * sigma**2))
            return gaussian_damage
            
        elif damage_type == 'step':
            def step_damage(x_norm: torch.Tensor, location: float, severity: float) -> torch.Tensor:
                """Step function damage influence."""
                # Simple step damage around location
                width = 0.1
                mask = torch.abs(x_norm - location) <= width/2
                return severity * mask.float()
            return step_damage
            
        else:
            raise ValueError(f"Unknown damage function: {damage_type}")
    
    def stiffness_field(self, x: torch.Tensor, damage_location: float, 
                       damage_severity: float) -> torch.Tensor:
        """
        Compute spatially varying stiffness field EI(x;d).
        
        Args:
            x: Spatial coordinates (normalized to [0,1])
            damage_location: Damage location (normalized to [0,1])
            damage_severity: Damage severity [0,1] representing stiffness reduction
            
        Returns:
            Stiffness field EI(x) at each spatial location
        """
        # Normalize x to [0,1] if not already
        if x.max() > 1.0:
            x_norm = x / self.L
        else:
            x_norm = x
            
        # Compute damage influence
        damage_influence = self.damage_function(x_norm, damage_location, damage_severity)
        
        # Stiffness reduction: EI(x;d) = EI_0 * (1 - d * φ(x))
        stiffness_reduction = 1.0 - damage_influence
        return self.EI_0 * stiffness_reduction
    
    def physics_residual(self, x: torch.Tensor, t: torch.Tensor, 
                        u: torch.Tensor, damage_param: torch.Tensor) -> torch.Tensor:
        """
        Compute physics residual for Euler-Bernoulli beam equation.
        
        Residual = ρA ∂²u/∂t² + c ∂u/∂t + ∂²/∂x²[EI(x;d) ∂²u/∂x²]
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates  
            u: Predicted displacement field
            damage_param: Damage parameters [location, severity]
            
        Returns:
            Physics residual at each point
        """
        # Extract damage parameters
        damage_location = damage_param[:, 0]  # First column
        damage_severity = damage_param[:, 1]  # Second column
        
        # Compute first derivatives
        u_x = compute_derivative(u, x, order=1)
        u_t = compute_derivative(u, t, order=1)
        
        # Compute second derivatives
        u_xx = compute_derivative(u, x, order=2)
        u_tt = compute_derivative(u, t, order=2)
        
        # Compute spatially varying stiffness field
        # Expand dimensions to match batch size
        x_expanded = x.expand_as(u)
        stiffness = self.stiffness_field(x_expanded, damage_location.mean(), damage_severity.mean())
        
        # Compute ∂²/∂x²[EI(x;d) ∂²u/∂x²]
        EI_u_xx = stiffness * u_xx
        EI_u_xx_x = compute_derivative(EI_u_xx, x, order=1)
        EI_u_xx_xx = compute_derivative(EI_u_xx_x, x, order=1)
        
        # Assemble physics residual
        residual = (self.rho_A * u_tt + 
                   self.c * u_t + 
                   EI_u_xx_xx)
        
        return residual
    
    def boundary_conditions(self, x: torch.Tensor, t: torch.Tensor, 
                           u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute boundary condition residuals.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Displacement field
            
        Returns:
            Tuple of (left_bc_residual, right_bc_residual)
        """
        bc_type_left = self.physics_params['boundary_conditions']['left']
        bc_type_right = self.physics_params['boundary_conditions']['right']
        
        # Left boundary (x = 0)
        if bc_type_left == 'clamped':
            # u(0,t) = 0 and ∂u/∂x(0,t) = 0
            u_left = u[x == 0]
            u_x_left = compute_derivative(u[x == 0], x[x == 0], order=1)
            left_residual = torch.cat([u_left, u_x_left])
        elif bc_type_left == 'simply_supported':
            # u(0,t) = 0 and ∂²u/∂x²(0,t) = 0
            u_left = u[x == 0]
            u_xx_left = compute_derivative(u[x == 0], x[x == 0], order=2)
            left_residual = torch.cat([u_left, u_xx_left])
        else:
            # Free boundary
            left_residual = torch.tensor([0.0], device=u.device)
        
        # Right boundary (x = L)
        if bc_type_right == 'free':
            # ∂²u/∂x²(L,t) = 0 and ∂³u/∂x³(L,t) = 0
            u_xx_right = compute_derivative(u[x == self.L], x[x == self.L], order=2)
            u_xxx_right = compute_derivative(u_xx_right, x[x == self.L], order=1)
            right_residual = torch.cat([u_xx_right, u_xxx_right])
        elif bc_type_right == 'clamped':
            # u(L,t) = 0 and ∂u/∂x(L,t) = 0
            u_right = u[x == self.L]
            u_x_right = compute_derivative(u[x == self.L], x[x == self.L], order=1)
            right_residual = torch.cat([u_right, u_x_right])
        else:
            # Simply supported
            u_right = u[x == self.L]
            u_xx_right = compute_derivative(u[x == self.L], x[x == self.L], order=2)
            right_residual = torch.cat([u_right, u_xx_right])
        
        return left_residual, right_residual
    
    def initial_conditions(self, x: torch.Tensor, t: torch.Tensor, 
                          u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute initial condition residuals.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Displacement field
            
        Returns:
            Tuple of (initial_displacement_residual, initial_velocity_residual)
        """
        # Initial displacement: u(x,0) = 0
        u_initial = u[t == 0]
        initial_disp_residual = u_initial
        
        # Initial velocity: ∂u/∂t(x,0) = 0
        u_t_initial = compute_derivative(u[t == 0], t[t == 0], order=1)
        initial_vel_residual = u_t_initial
        
        return initial_disp_residual, initial_vel_residual
    
    def energy_conservation_check(self, x: torch.Tensor, t: torch.Tensor, 
                                 u: torch.Tensor, damage_param: torch.Tensor) -> dict:
        """
        Check energy conservation properties.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Displacement field
            damage_param: Damage parameters
            
        Returns:
            Dictionary with energy-related metrics
        """
        # Kinetic energy: T = 1/2 ∫ ρA (∂u/∂t)² dx
        u_t = compute_derivative(u, t, order=1)
        kinetic_energy_density = 0.5 * self.rho_A * u_t**2
        
        # Strain energy: V = 1/2 ∫ EI(x;d) (∂²u/∂x²)² dx
        u_xx = compute_derivative(u, x, order=2)
        stiffness = self.stiffness_field(x, damage_param[:, 0].mean(), damage_param[:, 1].mean())
        strain_energy_density = 0.5 * stiffness * u_xx**2
        
        # Total energies
        kinetic_energy = torch.trapz(kinetic_energy_density.flatten(), x.flatten())
        strain_energy = torch.trapz(strain_energy_density.flatten(), x.flatten())
        total_energy = kinetic_energy + strain_energy
        
        return {
            'kinetic_energy': kinetic_energy.item(),
            'strain_energy': strain_energy.item(),
            'total_energy': total_energy.item(),
            'energy_conservation_error': 0.0  # Would need time derivatives for full check
        }


class AnalyticalBeamSolution:
    """
    Analytical solutions for validation purposes.
    """
    
    @staticmethod
    def undamaged_beam_modes(L: float, E: float, I: float, rho: float, A: float,
                            mode: int = 1) -> Tuple[float, Callable]:
        """
        Compute natural frequency and mode shape for undamaged beam.
        
        Args:
            L: Beam length
            E: Young's modulus
            I: Second moment of area
            rho: Density
            A: Cross-sectional area
            mode: Mode number
            
        Returns:
            Tuple of (natural_frequency, mode_shape_function)
        """
        # For clamped-free beam, eigenvalues
        if mode == 1:
            beta_L = 1.875  # First mode
        elif mode == 2:
            beta_L = 4.694  # Second mode
        else:
            beta_L = (mode - 0.5) * torch.pi  # Approximation for higher modes
            
        # Natural frequency
        omega_n = (beta_L/L)**2 * torch.sqrt(torch.tensor(E*I/(rho*A)))
        
        # Mode shape function
        def mode_shape(x):
            beta = beta_L / L
            return (torch.cos(beta*x) - torch.cosh(beta*x) + 
                   0.734 * (torch.sin(beta*x) - torch.sinh(beta*x)))
        
        return omega_n, mode_shape