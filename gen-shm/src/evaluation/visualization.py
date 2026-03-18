"""
Visualization tools for Gen-SHM results and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class SHMVisualizer:
    """
    Comprehensive visualization suite for Gen-SHM results.
    
    Provides plotting functions for:
    - Training progress and convergence
    - Model predictions and comparisons
    - Damage localization results
    - Signal analysis and frequency domain
    - Physics compliance validation
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer with plotting style."""
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 9))
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')
        
        epochs = range(len(history['train_loss']))
        
        # 1. Loss convergence (log scale)
        axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
        axes[0, 0].semilogy(epochs, history['physics_loss'], 'r--', linewidth=1.5, label='Physics Loss')
        axes[0, 0].semilogy(epochs, history['data_loss'], 'g--', linewidth=1.5, label='Data Loss')
        if 'boundary_loss' in history:
            axes[0, 0].semilogy(epochs, history['boundary_loss'], 'm--', linewidth=1.5, label='Boundary Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Learning rate schedule
        axes[0, 1].semilogy(epochs, history['learning_rate'], 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training time per epoch
        axes[0, 2].plot(epochs, history['epoch_time'], 'orange', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].set_title('Training Time per Epoch')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Loss component breakdown (stacked area)
        if all(key in history for key in ['data_loss', 'physics_loss', 'boundary_loss']):
            axes[1, 0].stackplot(epochs,
                               history['data_loss'],
                               history['physics_loss'],
                               history['boundary_loss'],
                               labels=['Data', 'Physics', 'Boundary'],
                               colors=['green', 'red', 'magenta'],
                               alpha=0.7)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Components')
            axes[1, 0].set_title('Loss Component Contributions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Moving average of losses (smoothed view)
        window_size = min(20, len(epochs) // 10)
        if window_size > 1:
            train_loss_smooth = self._moving_average(history['train_loss'], window_size)
            epochs_smooth = epochs[window_size-1:]
            
            axes[1, 1].semilogy(epochs_smooth, train_loss_smooth, 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].set_title(f'Training Loss (Moving Average, w={window_size})')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Loss ratios (balance between components)
        if len(history['physics_loss']) > 0 and len(history['data_loss']) > 0:
            ratios = [p/(d+1e-8) for p, d in zip(history['physics_loss'], history['data_loss'])]
            axes[1, 2].plot(epochs, ratios, 'brown', linewidth=2)
            axes[1, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Balance')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Physics/Data Loss Ratio')
            axes[1, 2].set_title('Loss Balance Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_predictions(self, 
                             x_test: np.ndarray,
                             t_test: np.ndarray,
                             u_true: np.ndarray,
                             u_pred: np.ndarray,
                             damage_info: Dict = None,
                             save_path: str = None) -> plt.Figure:
        """
        Plot model predictions vs ground truth.
        
        Args:
            x_test: Spatial coordinates
            t_test: Temporal coordinates  
            u_true: Ground truth displacements
            u_pred: Predicted displacements
            damage_info: Damage scenario information (optional)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 3D surface plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        X, T = np.meshgrid(np.unique(x_test), np.unique(t_test))
        U_true = u_true.reshape(X.shape)
        U_pred = u_pred.reshape(X.shape)
        
        surf1 = ax1.plot_surface(X, T, U_true, alpha=0.7, cmap='viridis', label='Ground Truth')
        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_zlabel('Displacement (u)')
        ax1.set_title('Ground Truth Solution')
        
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(X, T, U_pred, alpha=0.7, cmap='plasma', label='PINN Prediction')
        ax2.set_xlabel('Position (x)')
        ax2.set_ylabel('Time (t)')
        ax2.set_zlabel('Displacement (u)')
        ax2.set_title('PINN Prediction')
        
        # Error plot
        ax3 = fig.add_subplot(2, 2, 3)
        error = np.abs(U_true - U_pred)
        im = ax3.imshow(error, extent=[x_test.min(), x_test.max(), t_test.min(), t_test.max()],
                       aspect='auto', cmap='Reds')
        ax3.set_xlabel('Position (x)')
        ax3.set_ylabel('Time (t)')
        ax3.set_title('Absolute Error |True - Pred|')
        plt.colorbar(im, ax=ax3)
        
        # Cross-section comparison
        ax4 = fig.add_subplot(2, 2, 4)
        # Take middle time slice for comparison
        mid_time_idx = U_true.shape[0] // 2
        ax4.plot(X[mid_time_idx, :], U_true[mid_time_idx, :], 'b-', linewidth=2, label='True')
        ax4.plot(X[mid_time_idx, :], U_pred[mid_time_idx, :], 'r--', linewidth=2, label='Predicted')
        ax4.fill_between(X[mid_time_idx, :], 
                        U_true[mid_time_idx, :], 
                        U_pred[mid_time_idx, :], 
                        alpha=0.3, color='gray')
        ax4.set_xlabel('Position (x)')
        ax4.set_ylabel('Displacement (u)')
        ax4.set_title(f'Displacement Profile (t = {T[mid_time_idx, 0]:.2f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add damage info to title if provided
        title_suffix = ""
        if damage_info:
            title_suffix = f"\nDamage: {damage_info.get('level', 0)*100:.0f}% at location {damage_info.get('location', 0.5):.1f}"
        
        fig.suptitle(f'Model Predictions Comparison{title_suffix}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_damage_detection_results(self,
                                    true_damages: List[Dict],
                                    predicted_damages: List[Dict],
                                    save_path: str = None) -> plt.Figure:
        """
        Plot damage detection and localization results.
        
        Args:
            true_damages: List of true damage scenarios
            predicted_damages: List of predicted damage scenarios
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Damage Detection Results', fontsize=16, fontweight='bold')
        
        # Scatter plot: True vs Predicted Locations
        true_locations = [d['location'] for d in true_damages]
        pred_locations = [d['location'] for d in predicted_damages]
        true_severities = [d['severity'] for d in true_damages]
        pred_severities = [d['severity'] for d in predicted_damages]
        
        scatter = axes[0, 0].scatter(true_locations, pred_locations, 
                                   c=true_severities, cmap='viridis', s=100, alpha=0.7)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True Damage Location')
        axes[0, 0].set_ylabel('Predicted Damage Location')
        axes[0, 0].set_title('Location Prediction Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Damage Severity')
        
        # Severity correlation
        axes[0, 1].scatter(true_severities, pred_severities, 
                          c=true_locations, cmap='plasma', s=100, alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('True Damage Severity')
        axes[0, 1].set_ylabel('Predicted Damage Severity')
        axes[0, 1].set_title('Severity Prediction Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distributions
        location_errors = np.abs(np.array(true_locations) - np.array(pred_locations))
        severity_errors = np.abs(np.array(true_severities) - np.array(pred_severities))
        
        axes[1, 0].hist(location_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Location Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Location Prediction Error Distribution')
        axes[1, 0].axvline(np.mean(location_errors), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(location_errors):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(severity_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Severity Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Severity Prediction Error Distribution')
        axes[1, 1].axvline(np.mean(severity_errors), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(severity_errors):.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_signal_analysis(self,
                           time_vector: np.ndarray,
                           sensor_data: np.ndarray,
                           sensor_names: List[str] = None,
                           save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive signal analysis.
        
        Args:
            time_vector: Time points
            sensor_data: Sensor measurements (sensors × time)
            sensor_names: Names of sensors (optional)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        num_sensors = sensor_data.shape[0]
        fig, axes = plt.subplots(3, num_sensors, figsize=(5*num_sensors, 12))
        fig.suptitle('Vibration Signal Analysis', fontsize=16, fontweight='bold')
        
        if num_sensors == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_sensors):
            signal = sensor_data[i, :]
            
            # Time domain signal
            axes[0, i].plot(time_vector, signal, 'b-', linewidth=1)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('Amplitude')
            title = f'Sensor {i+1}' if not sensor_names else sensor_names[i]
            axes[0, i].set_title(f'{title} - Time Domain')
            axes[0, i].grid(True, alpha=0.3)
            
            # FFT analysis
            fft_vals = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), time_vector[1] - time_vector[0])
            positive_freq_idx = freqs > 0
            
            axes[1, i].semilogy(freqs[positive_freq_idx], 
                              np.abs(fft_vals[positive_freq_idx]), 'r-', linewidth=1)
            axes[1, i].set_xlabel('Frequency (Hz)')
            axes[1, i].set_ylabel('Magnitude')
            axes[1, i].set_title(f'{title} - Frequency Spectrum')
            axes[1, i].grid(True, alpha=0.3)
            
            # Spectrogram
            f, t, Sxx = signal.spectrogram(signal, fs=1/(time_vector[1]-time_vector[0]))
            im = axes[2, i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            axes[2, i].set_xlabel('Time (s)')
            axes[2, i].set_ylabel('Frequency (Hz)')
            axes[2, i].set_title(f'{title} - Spectrogram')
            plt.colorbar(im, ax=axes[2, i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_animation(self,
                        x_grid: np.ndarray,
                        t_grid: np.ndarray,
                        u_solution: np.ndarray,
                        save_path: str = None) -> animation.FuncAnimation:
        """
        Create animated visualization of wave propagation.
        
        Args:
            x_grid: Spatial grid points
            t_grid: Temporal grid points
            u_solution: Solution values on grid
            save_path: Path to save animation (optional, .mp4 or .gif)
            
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize plot
        line, = ax.plot(x_grid[:, 0], u_solution[:, 0], 'b-', linewidth=2)
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(u_solution.min(), u_solution.max())
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Displacement (u)')
        ax.set_title('Wave Propagation Animation')
        ax.grid(True, alpha=0.3)
        
        # Animation update function
        def animate(frame):
            line.set_ydata(u_solution[:, frame])
            ax.set_title(f'Wave Propagation (t = {t_grid[0, frame]:.3f}s)')
            return line,
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=u_solution.shape[1],
                                     interval=50, blit=True, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=20)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=20)
        
        return anim
    
    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        """Compute moving average for smoothing."""
        if len(data) < window_size:
            return np.array(data)
        
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


# Convenience plotting functions
def quick_plot_training(history: Dict[str, List[float]], 
                       save_path: str = None) -> None:
    """Quick function to plot training history."""
    visualizer = SHMVisualizer()
    fig = visualizer.plot_training_history(history, save_path)
    plt.show()


def quick_plot_predictions(x_test: np.ndarray,
                          t_test: np.ndarray,
                          u_true: np.ndarray,
                          u_pred: np.ndarray,
                          save_path: str = None) -> None:
    """Quick function to plot predictions."""
    visualizer = SHMVisualizer()
    fig = visualizer.plot_model_predictions(x_test, t_test, u_true, u_pred, save_path=save_path)
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         classes: List[str] = None,
                         save_path: str = None) -> plt.Figure:
    """Plot confusion matrix for classification results."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    if classes is None:
        classes = [f'Class {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()