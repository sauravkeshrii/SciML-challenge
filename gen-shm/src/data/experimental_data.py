"""
Experimental data handling for real sensor measurements.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from ..utils.config import config


class ExperimentalDataHandler:
    """
    Handle real experimental data from drone wing sensors.
    
    Supports:
    - Loading data from various formats (CSV, NumPy, etc.)
    - Preprocessing and normalization
    - Sensor calibration
    - Data validation and quality checks
    """
    
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config.config
        self.calibration_data = None
        self.sensor_mapping = None
    
    def load_csv_data(self, file_path: str, 
                     time_column: str = 'time',
                     sensor_columns: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load experimental data from CSV file.
        
        Args:
            file_path: Path to CSV file
            time_column: Name of time column
            sensor_columns: List of sensor column names
            
        Returns:
            Dictionary with loaded data
        """
        df = pd.read_csv(file_path)
        
        if sensor_columns is None:
            # Assume all columns except time are sensor data
            sensor_columns = [col for col in df.columns if col != time_column]
        
        data = {
            'time': df[time_column].values,
            'sensor_data': df[sensor_columns].values.T,  # (sensors, time)
            'sensor_names': sensor_columns,
            'sampling_rate': self._estimate_sampling_rate(df[time_column].values)
        }
        
        return data
    
    def load_numpy_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load experimental data from NumPy file.
        
        Expected format: dict with 'time' and 'sensor_data' keys
        """
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            # If just an array, assume it's sensor data vs time
            return {
                'time': np.arange(data.shape[1]),
                'sensor_data': data,
                'sampling_rate': 1000  # Default assumption
            }
        else:
            return dict(data)
    
    def calibrate_sensors(self, healthy_data: Dict[str, np.ndarray]) -> None:
        """
        Calibrate sensors using healthy baseline data.
        
        Args:
            healthy_data: Baseline healthy state data
        """
        self.calibration_data = {
            'mean': np.mean(healthy_data['sensor_data'], axis=1),
            'std': np.std(healthy_data['sensor_data'], axis=1),
            'sensor_names': healthy_data.get('sensor_names', 
                                           [f'sensor_{i}' for i in range(healthy_data['sensor_data'].shape[0])])
        }
    
    def preprocess_data(self, raw_data: Dict[str, np.ndarray],
                       apply_calibration: bool = True,
                       normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Preprocess experimental data for model consumption.
        
        Args:
            raw_data: Raw experimental data
            apply_calibration: Whether to apply sensor calibration
            normalize: Whether to normalize data to [0,1]
            
        Returns:
            Preprocessed data as PyTorch tensors
        """
        processed_data = {}
        
        # Convert to tensors
        time_tensor = torch.from_numpy(raw_data['time']).float()
        sensor_tensor = torch.from_numpy(raw_data['sensor_data']).float()
        
        # Apply calibration if available and requested
        if apply_calibration and self.calibration_data is not None:
            # Remove mean and normalize by standard deviation
            mean_expanded = torch.from_numpy(self.calibration_data['mean']).float().unsqueeze(1)
            std_expanded = torch.from_numpy(self.calibration_data['std']).float().unsqueeze(1)
            
            sensor_tensor = (sensor_tensor - mean_expanded) / (std_expanded + 1e-8)
        
        # Normalize to [0,1] if requested
        if normalize:
            sensor_tensor = (sensor_tensor - sensor_tensor.min()) / (sensor_tensor.max() - sensor_tensor.min() + 1e-8)
        
        processed_data.update({
            'time': time_tensor,
            'sensor_data': sensor_tensor,
            'sampling_rate': raw_data.get('sampling_rate', 1000)
        })
        
        return processed_data
    
    def validate_data_quality(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform data quality checks.
        
        Args:
            data: Data to validate
            
        Returns:
            Quality metrics dictionary
        """
        sensor_data = data['sensor_data']
        
        quality_metrics = {
            'signal_to_noise_ratio': self._compute_snr(sensor_data),
            'data_completeness': self._compute_completeness(sensor_data),
            'baseline_drift': self._compute_baseline_drift(sensor_data),
            'cross_correlation': self._compute_cross_correlation(sensor_data)
        }
        
        return quality_metrics
    
    def _estimate_sampling_rate(self, time_vector: np.ndarray) -> float:
        """Estimate sampling rate from time vector."""
        dt = np.diff(time_vector)
        return 1.0 / np.mean(dt)
    
    def _compute_snr(self, sensor_data: np.ndarray) -> float:
        """Compute signal-to-noise ratio."""
        # Simple SNR estimation: signal power / noise power
        signal_power = np.var(sensor_data, axis=1)
        noise_power = np.var(np.diff(sensor_data, axis=1), axis=1)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        return float(np.mean(snr_db))
    
    def _compute_completeness(self, sensor_data: np.ndarray) -> float:
        """Compute data completeness (fraction of valid readings)."""
        valid_readings = ~np.isnan(sensor_data) & ~np.isinf(sensor_data)
        return float(np.mean(valid_readings))
    
    def _compute_baseline_drift(self, sensor_data: np.ndarray) -> float:
        """Compute baseline drift metric."""
        # Measure variation in signal mean over time windows
        window_size = max(1, sensor_data.shape[1] // 10)
        window_means = []
        
        for i in range(0, sensor_data.shape[1] - window_size, window_size):
            window_mean = np.mean(sensor_data[:, i:i+window_size], axis=1)
            window_means.append(window_mean)
        
        if len(window_means) > 1:
            drift = np.std(window_means, axis=0)
            return float(np.mean(drift))
        else:
            return 0.0
    
    def _compute_cross_correlation(self, sensor_data: np.ndarray) -> float:
        """Compute average cross-correlation between sensors."""
        if sensor_data.shape[0] < 2:
            return 1.0
            
        correlations = []
        for i in range(sensor_data.shape[0]):
            for j in range(i+1, sensor_data.shape[0]):
                corr = np.corrcoef(sensor_data[i], sensor_data[j])[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        return float(np.mean(correlations))


def create_synthetic_experimental_data(num_sensors: int = 3,
                                     duration: float = 2.0,
                                     sampling_rate: int = 1000) -> Dict[str, np.ndarray]:
    """
    Create synthetic experimental data for testing purposes.
    
    Args:
        num_sensors: Number of virtual sensors
        duration: Recording duration in seconds
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Synthetic experimental data dictionary
    """
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Simulate multi-modal response
    frequencies = [25, 50, 100, 200]  # Hz
    amplitudes = [1.0, 0.7, 0.5, 0.3]
    
    sensor_data = np.zeros((num_sensors, len(time)))
    
    for sensor_idx in range(num_sensors):
        # Different phase for each sensor
        phase_shift = sensor_idx * np.pi / 4
        
        for freq, amp in zip(frequencies, amplitudes):
            signal = amp * np.sin(2 * np.pi * freq * time + phase_shift)
            sensor_data[sensor_idx] += signal
    
    # Add noise
    noise = np.random.normal(0, 0.05, sensor_data.shape)
    sensor_data += noise
    
    return {
        'time': time,
        'sensor_data': sensor_data,
        'sensor_names': [f'Sensor_{i+1}' for i in range(num_sensors)],
        'sampling_rate': sampling_rate
    }