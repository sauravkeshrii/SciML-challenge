#!/usr/bin/env python3
"""
🎯 GEN-SHM FUNCTIONAL TESTS
Simple tests that verify core functionality is working
"""

import sys
from pathlib import Path
import os

# Setup paths properly
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
os.chdir(str(project_root))

def test_physics_basics():
    """Test basic physics engine functionality."""
    print("🧪 Testing Physics Engine Basics...")
    
    try:
        from src.models.beam_physics import BeamPhysics
        from src.utils.config import Config
        
        # Initialize physics engine
        config = Config()
        physics = BeamPhysics(config.config)
        
        # Test basic properties
        assert hasattr(physics, 'EI_0'), "Missing EI_0 property"
        assert physics.EI_0 > 0, "Invalid stiffness value"
        
        # Test stiffness field computation
        import torch
        x = torch.tensor([0.5])
        stiffness = physics.stiffness_field(x, 0.5, 0.0)  # Healthy case
        assert torch.allclose(stiffness, torch.tensor(physics.EI_0)), "Stiffness field computation failed"
        
        print("✅ Physics engine basics working")
        return True
        
    except Exception as e:
        print(f"❌ Physics test failed: {e}")
        return False

def test_pinn_structure():
    """Test PINN model structure."""
    print("🤖 Testing PINN Structure...")
    
    try:
        from src.models.pinn_generator import PINNGenerator
        from src.utils.config import Config
        
        # Initialize model
        config = Config()
        model = PINNGenerator(config.config)
        
        # Test basic model properties
        assert hasattr(model, 'forward'), "Missing forward method"
        assert hasattr(model, 'compute_physics_loss'), "Missing physics loss method"
        assert hasattr(model, 'network'), "Missing network attribute"
        
        # Test that it has reasonable parameter count
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000, f"Model too small: {total_params} parameters"
        assert total_params < 1000000, f"Model too large: {total_params} parameters"
        
        print(f"✅ PINN structure valid ({total_params:,} parameters)")
        return True
        
    except Exception as e:
        print(f"❌ PINN test failed: {e}")
        return False

def test_data_generation():
    """Test data generation capabilities."""
    print("📊 Testing Data Generation...")
    
    try:
        from src.data.data_generation import SyntheticDataGenerator
        from src.utils.config import Config
        
        # Initialize data generator
        config = Config()
        generator = SyntheticDataGenerator(config.config)
        
        # Test healthy data generation
        healthy_data = generator.generate_healthy_calibration_data(num_sensors=2)
        assert 'sensor_positions' in healthy_data, "Missing sensor positions"
        assert 'time' in healthy_data, "Missing time data"
        assert 'acceleration' in healthy_data, "Missing acceleration data"
        
        # Test that data has reasonable shapes
        assert len(healthy_data['sensor_positions']) == 2, "Wrong sensor count"
        assert len(healthy_data['time']) > 0, "Empty time vector"
        assert healthy_data['acceleration'].shape[0] == 2, "Wrong acceleration shape"
        
        print("✅ Data generation working")
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_config_system():
    """Test configuration system."""
    print("⚙️  Testing Configuration System...")
    
    try:
        from src.utils.config import Config
        
        # Test default config
        config = Config()
        assert config.get('physics.beam_length') == 1.0, "Wrong beam length"
        assert config.get('model.input_dim') == 4, "Wrong input dimension"
        assert config.get('training.epochs') == 1000, "Wrong epoch count"
        
        # Test config updating
        config.update('test.value', 42)
        assert config.get('test.value') == 42, "Config update failed"
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def run_functional_tests():
    """Run all functional tests."""
    print("=" * 60)
    print("🚀 GEN-SHM FUNCTIONAL TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_config_system,
        test_physics_basics,
        test_pinn_structure,
        test_data_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🏆 ALL TESTS PASSED!")
        print("✨ Your Gen-SHM implementation is working correctly!")
        return True
    else:
        print("⚠️  Some tests failed - check the errors above")
        return False

if __name__ == '__main__':
    success = run_functional_tests()
    exit(0 if success else 1)