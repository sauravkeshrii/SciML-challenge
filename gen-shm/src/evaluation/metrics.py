"""
Evaluation metrics for Structural Health Monitoring performance.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error
)
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.stats import pearsonr


class SHMMetrics:
    """
    Comprehensive metrics for Structural Health Monitoring evaluation.
    
    Includes:
    - Classification metrics (accuracy, precision, recall, F1)
    - Regression metrics (RMSE, MAE)
    - Signal processing metrics (SNR, correlation)
    - Physics compliance metrics
    """
    
    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute classification performance metrics.
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted probabilities or binary predictions
            threshold: Classification threshold for probability predictions
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert probabilities to binary predictions if needed
        if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
            y_pred_binary = (y_pred > threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        y_true = y_true.astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'specificity': recall_score(1 - y_true, 1 - y_pred_binary, zero_division=0)
        }
        
        # ROC-AUC if we have probability scores
        try:
            if len(np.unique(y_true)) > 1 and y_pred.dtype in [np.float32, np.float64]:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            else:
                metrics['roc_auc'] = 0.0
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression performance metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'r2_score': 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        }
        
        return metrics
    
    @staticmethod
    def compute_signal_metrics(signal_true: np.ndarray, 
                             signal_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute signal processing metrics for vibration analysis.
        
        Args:
            signal_true: Ground truth signal
            signal_pred: Predicted signal
            
        Returns:
            Dictionary of signal processing metrics
        """
        # Correlation coefficient
        correlation, _ = pearsonr(signal_true.flatten(), signal_pred.flatten())
        
        # Signal-to-noise ratio
        signal_power = np.mean(signal_true**2)
        noise_power = np.mean((signal_true - signal_pred)**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        
        # Cross-correlation peak
        cross_corr = signal.correlate(signal_true, signal_pred, mode='full')
        cross_corr_normalized = cross_corr / (np.linalg.norm(signal_true) * np.linalg.norm(signal_pred))
        max_correlation = np.max(np.abs(cross_corr_normalized))
        
        # Frequency domain similarity
        fft_true = np.fft.fft(signal_true)
        fft_pred = np.fft.fft(signal_pred)
        freq_similarity = np.abs(np.corrcoef(np.abs(fft_true), np.abs(fft_pred))[0, 1])
        
        metrics = {
            'correlation': correlation,
            'snr_db': snr_db,
            'cross_correlation_peak': max_correlation,
            'frequency_similarity': freq_similarity if not np.isnan(freq_similarity) else 0.0
        }
        
        return metrics
    
    @staticmethod
    def compute_damage_localization_metrics(true_locations: np.ndarray,
                                          pred_locations: np.ndarray,
                                          tolerance: float = 0.1) -> Dict[str, float]:
        """
        Compute damage localization accuracy metrics.
        
        Args:
            true_locations: Ground truth damage locations (normalized 0-1)
            pred_locations: Predicted damage locations (normalized 0-1)
            tolerance: Acceptable error tolerance (normalized units)
            
        Returns:
            Dictionary of localization metrics
        """
        # Location error
        location_errors = np.abs(true_locations - pred_locations)
        
        # Accuracy within tolerance
        correct_predictions = location_errors <= tolerance
        localization_accuracy = np.mean(correct_predictions)
        
        # Mean and standard deviation of errors
        mean_error = np.mean(location_errors)
        std_error = np.std(location_errors)
        
        # Median error (more robust)
        median_error = np.median(location_errors)
        
        metrics = {
            'localization_accuracy': localization_accuracy,
            'mean_location_error': mean_error,
            'std_location_error': std_error,
            'median_location_error': median_error,
            'max_location_error': np.max(location_errors)
        }
        
        return metrics


class PhysicsComplianceMetrics:
    """
    Metrics for evaluating physics constraint satisfaction.
    """
    
    @staticmethod
    def compute_energy_conservation_error(predicted_signals: np.ndarray,
                                        time_vector: np.ndarray) -> Dict[str, float]:
        """
        Evaluate energy conservation in predicted signals.
        
        Args:
            predicted_signals: Predicted acceleration signals (sensors × time)
            time_vector: Time points
            
        Returns:
            Energy conservation metrics
        """
        dt = np.mean(np.diff(time_vector))
        
        # Compute velocity by integration
        velocity_signals = np.cumsum(predicted_signals, axis=1) * dt
        
        # Compute displacement by double integration
        displacement_signals = np.cumsum(velocity_signals, axis=1) * dt
        
        # Kinetic energy (proportional to velocity²)
        kinetic_energy = np.mean(velocity_signals**2, axis=0)
        
        # Potential energy (proportional to displacement²)
        potential_energy = np.mean(displacement_signals**2, axis=0)
        
        # Total energy
        total_energy = kinetic_energy + potential_energy
        
        # Energy conservation error
        energy_variation = np.std(total_energy) / (np.mean(total_energy) + 1e-8)
        
        return {
            'energy_conservation_error': energy_variation,
            'mean_kinetic_energy': np.mean(kinetic_energy),
            'mean_potential_energy': np.mean(potential_energy),
            'energy_ratio_std': np.std(kinetic_energy / (potential_energy + 1e-8))
        }
    
    @staticmethod
    def compute_modal_analysis_metrics(acceleration_signals: np.ndarray,
                                     time_vector: np.ndarray,
                                     sampling_rate: float) -> Dict[str, float]:
        """
        Perform modal analysis on acceleration signals.
        
        Args:
            acceleration_signals: Acceleration measurements (sensors × time)
            time_vector: Time vector
            sampling_rate: Sampling frequency
            
        Returns:
            Modal analysis metrics
        """
        # Compute PSD for each sensor
        psd_list = []
        dominant_frequencies = []
        
        for signal in acceleration_signals:
            frequencies, psd = signal.welch(signal, fs=sampling_rate, nperseg=min(1024, len(signal)))
            psd_list.append(psd)
            
            # Find dominant frequency (peak in PSD)
            dominant_freq = frequencies[np.argmax(psd)]
            dominant_frequencies.append(dominant_freq)
        
        # Consistency across sensors
        freq_consistency = np.std(dominant_frequencies) / (np.mean(dominant_frequencies) + 1e-8)
        
        # Average PSD characteristics
        avg_psd = np.mean(psd_list, axis=0)
        psd_peaks = signal.find_peaks(avg_psd)[0]
        
        return {
            'frequency_consistency': freq_consistency,
            'mean_dominant_frequency': np.mean(dominant_frequencies),
            'std_dominant_frequency': np.std(dominant_frequencies),
            'number_of_modes': len(psd_peaks)
        }


class UncertaintyQuantification:
    """
    Methods for quantifying prediction uncertainty.
    """
    
    @staticmethod
    def compute_prediction_intervals(predictions: np.ndarray,
                                   confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Compute prediction intervals from ensemble predictions.
        
        Args:
            predictions: Ensemble predictions (ensemble_size × ...)
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with lower bound, upper bound, and mean predictions
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': np.mean(predictions, axis=0),
            'lower_bound': np.percentile(predictions, lower_percentile, axis=0),
            'upper_bound': np.percentile(predictions, upper_percentile, axis=0),
            'std': np.std(predictions, axis=0)
        }
    
    @staticmethod
    def compute_calibration_metrics(y_true: np.ndarray,
                                  y_pred_mean: np.ndarray,
                                  y_pred_std: np.ndarray) -> Dict[str, float]:
        """
        Compute uncertainty calibration metrics.
        
        Args:
            y_true: Ground truth values
            y_pred_mean: Predicted mean values
            y_pred_std: Predicted standard deviations
            
        Returns:
            Calibration metrics
        """
        # Compute standardized residuals
        residuals = y_true - y_pred_mean
        standardized_residuals = np.abs(residuals) / (y_pred_std + 1e-8)
        
        # Calibration score (should be close to expected coverage)
        expected_coverage = 0.68  # ~1 std for normal distribution
        actual_coverage = np.mean(standardized_residuals <= 1.0)
        calibration_error = abs(actual_coverage - expected_coverage)
        
        # Sharpness (average prediction interval width)
        sharpness = np.mean(2 * y_pred_std)  # 2σ interval
        
        # Resolution metric
        resolution = np.std(y_pred_std)
        
        return {
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'resolution': resolution,
            'actual_coverage': actual_coverage
        }


def comprehensive_evaluation(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           signal_true: np.ndarray = None,
                           signal_pred: np.ndarray = None,
                           damage_locations_true: np.ndarray = None,
                           damage_locations_pred: np.ndarray = None) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive evaluation across all metric categories.
    
    Args:
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        signal_true: Ground truth signals (optional)
        signal_pred: Predicted signals (optional)
        damage_locations_true: True damage locations (optional)
        damage_locations_pred: Predicted damage locations (optional)
        
    Returns:
        Nested dictionary of all computed metrics
    """
    results = {}
    
    # Classification metrics (if binary problem)
    if len(np.unique(y_true)) <= 2:
        results['classification'] = SHMMetrics.compute_classification_metrics(y_true, y_pred)
    
    # Regression metrics
    results['regression'] = SHMMetrics.compute_regression_metrics(y_true, y_pred)
    
    # Signal metrics (if signals provided)
    if signal_true is not None and signal_pred is not None:
        results['signal'] = SHMMetrics.compute_signal_metrics(signal_true, signal_pred)
    
    # Localization metrics (if locations provided)
    if damage_locations_true is not None and damage_locations_pred is not None:
        results['localization'] = SHMMetrics.compute_damage_localization_metrics(
            damage_locations_true, damage_locations_pred
        )
    
    return results