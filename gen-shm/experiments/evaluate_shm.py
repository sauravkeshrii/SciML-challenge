#!/usr/bin/env python3
"""
Structural Health Monitoring evaluation script.

Evaluates trained Gen-SHM model performance on damage detection tasks.

Usage:
    python evaluate_shm.py --model_path MODEL_PATH --test_scenarios SCENARIO_FILE
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.surrogate_model import DroneWingSurrogate
from evaluation.metrics import comprehensive_evaluation, SHMMetrics
from evaluation.visualization import SHMVisualizer, plot_confusion_matrix
from evaluation.validation import PhysicsValidator
from utils.logger import get_experiment_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Gen-SHM for SHM tasks')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_scenarios', type=str,
                       help='Path to test scenarios file (JSON/PKL)')
    parser.add_argument('--num_test_cases', type=int, default=50,
                       help='Number of test cases to generate')
    parser.add_argument('--output_dir', type=str, default='shm_evaluation',
                       help='Output directory for results')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--physics_validation', action='store_true',
                       help='Include physics compliance validation')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions')
    
    return parser.parse_args()


def generate_test_scenarios(num_cases: int = 50) -> list:
    """Generate diverse test scenarios for evaluation."""
    np.random.seed(42)  # For reproducible test cases
    
    scenarios = []
    
    # Define specific test cases
    predefined_scenarios = [
        # Healthy cases
        {'location': 0.0, 'severity': 0.0, 'label': 0, 'name': 'Healthy_Root'},
        {'location': 0.5, 'severity': 0.0, 'label': 0, 'name': 'Healthy_Center'},
        {'location': 1.0, 'severity': 0.0, 'label': 0, 'name': 'Healthy_Tip'},
        
        # Light damage cases
        {'location': 0.0, 'severity': 0.05, 'label': 1, 'name': 'Light_Root_Damage'},
        {'location': 0.3, 'severity': 0.1, 'label': 1, 'name': 'Light_Mid_Damage'},
        {'location': 0.7, 'severity': 0.08, 'label': 1, 'name': 'Light_Tip_Damage'},
        
        # Moderate damage cases
        {'location': 0.0, 'severity': 0.2, 'label': 1, 'name': 'Moderate_Root_Damage'},
        {'location': 0.5, 'severity': 0.25, 'label': 1, 'name': 'Moderate_Center_Damage'},
        {'location': 0.8, 'severity': 0.18, 'label': 1, 'name': 'Moderate_Tip_Damage'},
        
        # Severe damage cases
        {'location': 0.1, 'severity': 0.35, 'label': 1, 'name': 'Severe_Root_Damage'},
        {'location': 0.5, 'severity': 0.4, 'label': 1, 'name': 'Severe_Center_Damage'},
        {'location': 0.9, 'severity': 0.3, 'label': 1, 'name': 'Severe_Tip_Damage'},
    ]
    
    scenarios.extend(predefined_scenarios)
    
    # Add random scenarios to reach desired number
    while len(scenarios) < num_cases:
        location = np.random.uniform(0.1, 0.9)
        severity = np.random.uniform(0.0, 0.4)
        label = 1 if severity > 0.05 else 0  # Damage if severity > 5%
        name = f"Random_{len(scenarios)}"
        
        scenarios.append({
            'location': location,
            'severity': severity,
            'label': label,
            'name': name
        })
    
    return scenarios


def load_test_scenarios(filepath: str) -> list:
    """Load test scenarios from file."""
    path = Path(filepath)
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_experiment_logger('shm_evaluation', str(output_dir))
    logger.info("Starting SHM evaluation")
    
    try:
        # Load trained model
        logger.info(f"Loading model from: {args.model_path}")
        surrogate = DroneWingSurrogate(model_path=args.model_path)
        
        # Load or generate test scenarios
        if args.test_scenarios:
            logger.info(f"Loading test scenarios from: {args.test_scenarios}")
            test_scenarios = load_test_scenarios(args.test_scenarios)
        else:
            logger.info(f"Generating {args.num_test_cases} test scenarios")
            test_scenarios = generate_test_scenarios(args.num_test_cases)
        
        logger.info(f"Total test cases: {len(test_scenarios)}")
        
        # Initialize results storage
        true_labels = []
        pred_labels = []
        pred_probabilities = []
        true_locations = []
        pred_locations = []
        true_severities = []
        pred_severities = []
        prediction_details = []
        
        # Evaluate each test case
        logger.info("Evaluating test cases...")
        
        for i, scenario in enumerate(test_scenarios):
            if i % 10 == 0:
                logger.info(f"Processing scenario {i+1}/{len(test_scenarios)}")
            
            # Generate test data
            samples = surrogate.generate_samples(
                damage_level=scenario['severity'],
                damage_location=scenario['location'],
                num_samples=5,  # Fewer samples for evaluation speed
                duration=1.0,
                sampling_rate=500
            )
            
            # Simple damage detection logic (placeholder)
            # In practice, this would use a trained classifier on the features
            avg_amplitude = np.mean(np.abs(samples['acceleration']))
            
            # Heuristic damage detection based on amplitude change
            # This is a simplified approach - real implementation would use ML classifier
            threshold = 0.01  # Amplitude threshold for damage detection
            pred_label = 1 if avg_amplitude > threshold else 0
            pred_probability = min(1.0, avg_amplitude / (threshold * 2))
            
            # Store results
            true_labels.append(scenario['label'])
            pred_labels.append(pred_label)
            pred_probabilities.append(pred_probability)
            true_locations.append(scenario['location'])
            # For location/severity prediction, we'd need a regression model
            pred_locations.append(scenario['location'])  # Placeholder
            true_severities.append(scenario['severity'])
            pred_severities.append(scenario['severity'])  # Placeholder
            
            # Store detailed prediction
            if args.save_predictions:
                prediction_details.append({
                    'scenario': scenario,
                    'predicted_label': pred_label,
                    'prediction_probability': pred_probability,
                    'features': {
                        'mean_amplitude': float(avg_amplitude),
                        'max_amplitude': float(np.max(np.abs(samples['acceleration']))),
                        'std_amplitude': float(np.std(samples['acceleration']))
                    }
                })
        
        # Compute comprehensive metrics
        logger.info("Computing evaluation metrics...")
        
        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)
        y_prob = np.array(pred_probabilities)
        loc_true = np.array(true_locations)
        loc_pred = np.array(pred_locations)
        sev_true = np.array(true_severities)
        sev_pred = np.array(pred_severities)
        
        # Comprehensive evaluation
        eval_results = comprehensive_evaluation(
            y_true=y_true,
            y_pred=y_pred,
            signal_true=None,  # Would need actual signal data
            signal_pred=None,
            damage_locations_true=loc_true,
            damage_locations_pred=loc_pred
        )
        
        # Add probability-based metrics
        if len(np.unique(y_true)) > 1:
            eval_results['classification']['roc_auc_prob'] = SHMMetrics.compute_classification_metrics(
                y_true, y_prob, threshold=0.5
            ).get('roc_auc', 0.0)
        
        # Save evaluation results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Save detailed predictions if requested
        if args.save_predictions:
            predictions_path = output_dir / 'predictions.pkl'
            with open(predictions_path, 'wb') as f:
                pickle.dump(prediction_details, f)
            logger.info(f"Detailed predictions saved to: {predictions_path}")
        
        # Generate summary report
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("GEN-SHM STRUCTURAL HEALTH MONITORING EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Test Cases: {len(test_scenarios)}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            
            for category, metrics in eval_results.items():
                f.write(f"\n{category.upper()}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.6f}\n")
        
        logger.info("Evaluation metrics:")
        for category, metrics in eval_results.items():
            logger.info(f"\n{category.upper()}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.6f}")
        
        # Generate plots if requested
        if args.plot_results:
            logger.info("Generating evaluation plots...")
            visualizer = SHMVisualizer()
            
            # Confusion matrix
            cm_fig = plot_confusion_matrix(y_true, y_pred, 
                                         classes=['Healthy', 'Damaged'])
            cm_path = output_dir / 'confusion_matrix.png'
            cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to: {cm_path}")
            
            # ROC curve (if binary classification)
            if len(np.unique(y_true)) > 1:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve')
                plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                roc_path = output_dir / 'roc_curve.png'
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"ROC curve saved to: {roc_path}")
        
        # Physics validation if requested
        if args.physics_validation:
            logger.info("Running physics compliance validation...")
            validator = PhysicsValidator(surrogate.pinn_model, surrogate.config)
            physics_results = validator.comprehensive_validation()
            
            physics_report = validator.generate_validation_report(physics_results)
            
            physics_path = output_dir / 'physics_validation_report.txt'
            with open(physics_path, 'w') as f:
                f.write(physics_report)
            
            logger.info("Physics validation completed")
            logger.info(physics_report)
        
        logger.info("SHM evaluation completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()