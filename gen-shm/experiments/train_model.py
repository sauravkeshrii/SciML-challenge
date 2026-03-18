#!/usr/bin/env python3
"""
Main training script for Gen-SHM model.

Usage:
    python train_model.py [--config CONFIG_PATH] [--epochs EPOCHS] [--gpu GPU_ID]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from datetime import datetime

from models.surrogate_model import DroneWingSurrogate
from utils.config import Config
from utils.logger import get_experiment_logger
from utils.helpers import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Gen-SHM PINN model')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, 
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str,
                       help='Custom experiment name')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        exp_name = f"{args.experiment_name}_{timestamp}"
    else:
        exp_name = f"gen_shm_training_{timestamp}"
    
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return device, exp_dir


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    device, exp_dir = setup_environment(args)
    
    # Setup logging
    logger = get_experiment_logger('training', str(exp_dir))
    logger.info(f"Starting Gen-SHM training experiment: {exp_dir.name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
        
        # Override epochs if specified
        if args.epochs:
            config.update('training.epochs', args.epochs)
            logger.info(f"Overriding epochs to: {args.epochs}")
        
        # Save configuration
        config_path = exp_dir / 'config.yaml'
        config.save(str(config_path))
        logger.info(f"Configuration saved to: {config_path}")
        
        # Initialize surrogate model
        logger.info("Initializing surrogate model...")
        surrogate = DroneWingSurrogate(config_dict=config.config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            surrogate.load_model(args.resume)
        
        # Train model
        logger.info("Starting training...")
        history = surrogate.train(verbose=args.verbose)
        
        # Save trained model
        model_path = exp_dir / 'trained_model.pt'
        surrogate.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        # Save training history
        import pickle
        history_path = exp_dir / 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        logger.info(f"Training history saved to: {history_path}")
        
        # Generate validation samples
        logger.info("Generating validation samples...")
        validation_samples = surrogate.generate_samples(
            damage_level=0.2,
            damage_location=0.5,
            num_samples=10
        )
        
        validation_path = exp_dir / 'validation_samples.pkl'
        with open(validation_path, 'wb') as f:
            pickle.dump(validation_samples, f)
        logger.info(f"Validation samples saved to: {validation_path}")
        
        # Physics validation
        logger.info("Running physics validation...")
        from evaluation.validation import quick_validation
        validation_results = quick_validation(surrogate.pinn_model, config.config)
        
        validation_report_path = exp_dir / 'validation_report.txt'
        from evaluation.validation import PhysicsValidator
        validator = PhysicsValidator(surrogate.pinn_model, config.config)
        report = validator.generate_validation_report(validation_results, str(validation_report_path))
        logger.info("Physics validation completed")
        print(report)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved in: {exp_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()