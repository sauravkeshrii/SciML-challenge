#!/usr/bin/env python3
"""
Sample generation script for Gen-SHM model.

Generates synthetic vibration data for specified damage scenarios.

Usage:
    python generate_samples.py --model_path MODEL_PATH --damage_level 0.2 --damage_location 0.5 --num_samples 100
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.surrogate_model import DroneWingSurrogate
from evaluation.visualization import SHMVisualizer
from utils.logger import get_experiment_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic samples with Gen-SHM')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--damage_level', type=float, default=0.2,
                       help='Damage severity (0.0 to 1.0)')
    parser.add_argument('--damage_location', type=float, default=0.5,
                       help='Damage location along wing (0.0 to 1.0)')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to generate')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration of each sample in seconds')
    parser.add_argument('--sampling_rate', type=int, default=1000,
                       help='Sampling rate in Hz')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Output directory for results')
    parser.add_argument('--save_format', type=str, choices=['pkl', 'npz', 'csv'], 
                       default='pkl', help='Output file format')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of generated samples')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation of wave propagation')
    
    return parser.parse_args()


def save_samples(samples: dict, output_path: Path, format: str):
    """Save generated samples in specified format."""
    if format == 'pkl':
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(samples, f)
    elif format == 'npz':
        np.savez_compressed(output_path.with_suffix('.npz'),
                          acceleration=samples['acceleration'],
                          time=samples['time'],
                          sensor_positions=samples['sensor_positions'],
                          damage_info=samples['damage_info'])
    elif format == 'csv':
        # Save each sensor's data as separate CSV
        for sensor_idx in range(samples['acceleration'].shape[1]):
            csv_path = output_path.with_name(f"{output_path.stem}_sensor_{sensor_idx}.csv")
            data = samples['acceleration'][:, sensor_idx, :].T  # (time, samples)
            np.savetxt(csv_path, data, delimiter=',', header='time,' + ','.join([f'sample_{i}' for i in range(data.shape[1])]))


def main():
    """Main sample generation function."""
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_experiment_logger('sample_generation', str(output_dir))
    logger.info("Starting sample generation")
    
    try:
        # Load trained model
        logger.info(f"Loading model from: {args.model_path}")
        surrogate = DroneWingSurrogate(model_path=args.model_path)
        
        # Validate model is trained
        if not surrogate.is_trained:
            raise RuntimeError("Model is not trained. Please train the model first.")
        
        # Generate samples
        logger.info(f"Generating {args.num_samples} samples with "
                   f"damage level {args.damage_level} at location {args.damage_location}")
        
        samples = surrogate.generate_samples(
            damage_level=args.damage_level,
            damage_location=args.damage_location,
            num_samples=args.num_samples,
            duration=args.duration,
            sampling_rate=args.sampling_rate
        )
        
        # Save samples
        timestamp = Path(args.model_path).stem
        output_filename = f"samples_dmg{args.damage_level:.2f}_loc{args.damage_location:.2f}_{timestamp}"
        output_path = output_dir / output_filename
        
        save_samples(samples, output_path, args.save_format)
        logger.info(f"Samples saved to: {output_path}.{args.save_format}")
        
        # Generate summary statistics
        acceleration_data = samples['acceleration']
        stats = {
            'mean_amplitude': np.mean(np.abs(acceleration_data)),
            'max_amplitude': np.max(np.abs(acceleration_data)),
            'std_amplitude': np.std(acceleration_data),
            'snr_db': 10 * np.log10(np.mean(acceleration_data**2) / np.mean(np.std(acceleration_data, axis=0)**2))
        }
        
        # Save statistics
        stats_path = output_path.with_suffix('.stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Sample Generation Statistics\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Damage Level: {args.damage_level}\n")
            f.write(f"Damage Location: {args.damage_location}\n")
            f.write(f"Number of Samples: {args.num_samples}\n")
            f.write(f"Duration: {args.duration} seconds\n")
            f.write(f"Sampling Rate: {args.sampling_rate} Hz\n\n")
            f.write("Signal Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value:.6f}\n")
        
        logger.info("Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # Generate plots if requested
        if args.plot:
            logger.info("Generating plots...")
            visualizer = SHMVisualizer()
            
            # Plot sample signals
            fig1 = plt.figure(figsize=(15, 10))
            
            # Plot first few samples from each sensor
            num_plot_samples = min(5, args.num_samples)
            sensor_positions = samples['sensor_positions']
            
            for sensor_idx, sensor_pos in enumerate(sensor_positions):
                ax = plt.subplot(len(sensor_positions), 1, sensor_idx + 1)
                time = samples['time']
                
                for sample_idx in range(num_plot_samples):
                    ax.plot(time, acceleration_data[sample_idx, sensor_idx, :], 
                           alpha=0.7, linewidth=1)
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Acceleration')
                ax.set_title(f'Sensor at Position {sensor_pos:.2f}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_path.with_suffix('.signals.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Signal plots saved to: {plot_path}")
            
            # Plot frequency analysis
            fig2, axes = plt.subplots(len(sensor_positions), 1, figsize=(12, 4*len(sensor_positions)))
            if len(sensor_positions) == 1:
                axes = [axes]
            
            for sensor_idx, sensor_pos in enumerate(sensor_positions):
                # Compute average spectrum
                sample_spectra = []
                for sample_idx in range(args.num_samples):
                    signal = acceleration_data[sample_idx, sensor_idx, :]
                    fft_vals = np.fft.fft(signal)
                    freqs = np.fft.fftfreq(len(signal), 1/args.sampling_rate)
                    positive_freq_idx = freqs > 0
                    sample_spectra.append(np.abs(fft_vals[positive_freq_idx]))
                
                avg_spectrum = np.mean(sample_spectra, axis=0)
                freqs_positive = freqs[positive_freq_idx]
                
                axes[sensor_idx].semilogy(freqs_positive, avg_spectrum)
                axes[sensor_idx].set_xlabel('Frequency (Hz)')
                axes[sensor_idx].set_ylabel('Magnitude')
                axes[sensor_idx].set_title(f'Average Spectrum - Sensor {sensor_pos:.2f}')
                axes[sensor_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            spectrum_path = output_path.with_suffix('.spectrum.png')
            plt.savefig(spectrum_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Spectrum plots saved to: {spectrum_path}")
        
        # Create animation if requested
        if args.animate:
            logger.info("Creating animation...")
            # This would require 2D field data - simplified version
            logger.warning("Animation feature requires 2D field data implementation")
        
        logger.info("Sample generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Sample generation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()