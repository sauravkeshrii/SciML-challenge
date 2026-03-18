#!/usr/bin/env python3
"""
🎯 DIRECT IMPORT TESTS FOR GEN-SHM
Bypassing package import issues to test core functionality
"""

import sys
from pathlib import Path
import os

# Setup paths - add both project root and src to Python path
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / 'src'
models_path = src_path / 'models'
utils_path = src_path / 'utils'
data_path = src_path / 'data'

# Add all paths to sys.path
paths_to_add = [str(project_root), str(src_path), str(models_path), str(utils_path), str(data_path)]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(str(project_root))

def test_direct_imports():
    """Test direct file imports."""
    print("🧪 Testing Direct Imports...")
    
    try:
        # Test importing individual files directly
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        # Import config directly
        sys.path.insert(0, str(utils_path))
        from config import Config
        config = Config()
        print("✅ Config imported successfully")
        
        # Import physics directly
        sys.path.insert(0, str(models_path))
        from beam_physics import BeamPhysics
        physics = BeamPhysics(config.config)
        print("✅ Physics engine imported successfully")
        
        # Import PINN directly
        from pinn_generator import PINNGenerator
        model = PINNGenerator(config.config)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ PINN imported successfully ({params:,} parameters)")
        
        # Import data generator directly
        sys.path.insert(0, str(data_path))
        from data_generation import SyntheticDataGenerator
        generator = SyntheticDataGenerator(config.config)
        print("✅ Data generator imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_functionality():
    """Test core system functionality."""
    print("⚡ Testing Core Functionality...")
    
    try:
        import torch
        
        # Import everything we need
        from config import Config
        from beam_physics import BeamPhysics
        from pinn_generator import PINNGenerator
        from data_generation import SyntheticDataGenerator
        
        # Initialize components
        config = Config()
        physics = BeamPhysics(config.config)
        model = PINNGenerator(config.config)
        generator = SyntheticDataGenerator(config.config)
        
        # Test physics computation
        x = torch.tensor([0.5])
        stiffness = physics.stiffness_field(x, 0.5, 0.0)
        assert torch.allclose(stiffness, torch.tensor(physics.EI_0))
        print("✅ Physics computation working")
        
        # Test data generation
        data = generator.generate_healthy_calibration_data(num_sensors=2)
        assert 'acceleration' in data
        assert data['acceleration'].shape[0] == 2
        print("✅ Data generation working")
        
        # Test model forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 5
        x_input = torch.rand(batch_size, 1).to(device)
        t_input = torch.rand(batch_size, 1).to(device)
        loc_input = torch.rand(batch_size, 1).to(device)
        sev_input = torch.rand(batch_size, 1).to(device)
        
        # Test the forward method directly
        output = model(x_input.squeeze(), t_input.squeeze(), loc_input.squeeze(), sev_input.squeeze())
        assert output.shape == (batch_size, 1)
        print("✅ Model forward pass working")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test training-related components."""
    print("🏋️  Testing Training Components...")
    
    try:
        from config import Config
        from pinn_generator import PINNGenerator
        
        config = Config()
        model = PINNGenerator(config.config)
        
        # Test that model has required training methods
        assert hasattr(model, 'compute_physics_loss')
        assert hasattr(model, 'compute_boundary_loss')
        assert hasattr(model, 'compute_initial_loss')  # Changed from compute_data_loss
        
        print("✅ Training components present")
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def run_simple_tests():
    """Run simplified tests that bypass complex import issues."""
    print("=" * 60)
    print("🚀 GEN-SHM SIMPLE VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_direct_imports,
        test_core_functionality,
        test_training_components
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
        print("✨ Core Gen-SHM functionality is working correctly!")
        print("\n🚀 Ready for:")
        print("   • Full training: python experiments/train_model.py")
        print("   • Sample generation: python experiments/generate_samples.py")
        print("   • Performance evaluation: python experiments/evaluate_shm.py")
        return True
    else:
        print("⚠️  Some tests failed - system may need debugging")
        return False

if __name__ == '__main__':
    success = run_simple_tests()
    exit(0 if success else 1)