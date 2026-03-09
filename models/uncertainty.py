# models/uncertainty.py
# Uncertainty quantification using Monte Carlo Dropout

import torch
import torch.nn as nn
import numpy as np


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation
    Enables dropout during inference to get uncertainty estimates
    """
    def __init__(self, model, n_samples=10, dropout_rate=0.1):
        """
        Args:
            model: The base model (UNet)
            n_samples: Number of forward passes for MC sampling
            dropout_rate: Dropout probability
        """
        super(MCDropout, self).__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout layers to the model if not present
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add dropout to model layers"""
        # Inject dropout after convolutional blocks in the encoder and decoder
        # This modifies the model in-place
        
        def inject_dropout(module):
            if isinstance(module, nn.Sequential):
                # Check if dropout already exists
                has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in module)
                if not has_dropout:
                    # Add dropout at the end of the block
                    module.add_module('dropout', nn.Dropout2d(p=self.dropout_rate))
        
        # Apply to encoder blocks
        if hasattr(self.model, 'enc1'): inject_dropout(self.model.enc1.net)
        if hasattr(self.model, 'enc2'): inject_dropout(self.model.enc2.net)
        if hasattr(self.model, 'enc3'): inject_dropout(self.model.enc3.net)
        if hasattr(self.model, 'enc4'): inject_dropout(self.model.enc4.net)
        
        # Apply to decoder blocks
        if hasattr(self.model, 'dec1'): inject_dropout(self.model.dec1.net)
        if hasattr(self.model, 'dec2'): inject_dropout(self.model.dec2.net)
        if hasattr(self.model, 'dec3'): inject_dropout(self.model.dec3.net)
    
    def forward(self, x):
        """Standard forward pass"""
        return self.model(x)
    
    def predict_with_uncertainty(self, x):
        """
        Perform multiple forward passes with dropout enabled
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            mean: Mean prediction (B, 1, H, W)
            uncertainty: Uncertainty map (B, 1, H, W)
            samples: All predictions (n_samples, B, 1, H, W)
        """
        self.model.train()  # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                pred_prob = torch.sigmoid(pred)
                samples.append(pred_prob)
        
        # Stack samples
        samples = torch.stack(samples, dim=0)  # (n_samples, B, 1, H, W)
        
        # Compute mean and variance
        mean = samples.mean(dim=0)  # (B, 1, H, W)
        variance = samples.var(dim=0)  # (B, 1, H, W)
        
        # Uncertainty can be measured as variance or entropy
        uncertainty = variance
        
        self.model.eval()  # Disable dropout
        
        return mean, uncertainty, samples


class EnsembleUncertainty:
    """
    Uncertainty estimation using model ensemble
    """
    def __init__(self, models):
        """
        Args:
            models: List of trained models
        """
        self.models = models
    
    def predict_with_uncertainty(self, x, device='cpu'):
        """
        Get predictions from all models and compute uncertainty
        
        Args:
            x: Input tensor (B, C, H, W)
            device: Device to run inference on
            
        Returns:
            mean: Mean prediction (B, 1, H, W)
            uncertainty: Uncertainty map (B, 1, H, W)
            predictions: All model predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                model.to(device)
                pred = model(x.to(device))
                pred_prob = torch.sigmoid(pred)
                predictions.append(pred_prob.cpu())
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (n_models, B, 1, H, W)
        
        # Compute mean and variance
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        uncertainty = variance
        
        return mean, uncertainty, predictions


def compute_predictive_entropy(probabilities):
    """
    Compute predictive entropy as uncertainty measure
    
    Args:
        probabilities: Predicted probabilities (n_samples, B, 1, H, W)
        
    Returns:
        entropy: Entropy map (B, 1, H, W)
    """
    # Mean probability
    mean_prob = probabilities.mean(dim=0)
    
    # Entropy: -p*log(p) - (1-p)*log(1-p)
    eps = 1e-8
    entropy = -(mean_prob * torch.log(mean_prob + eps) + 
                (1 - mean_prob) * torch.log(1 - mean_prob + eps))
    
    return entropy


def compute_mutual_information(probabilities):
    """
    Compute mutual information as uncertainty measure
    
    Args:
        probabilities: Predicted probabilities (n_samples, B, 1, H, W)
        
    Returns:
        mutual_info: Mutual information map (B, 1, H, W)
    """
    # Predictive entropy
    pred_entropy = compute_predictive_entropy(probabilities)
    
    # Expected entropy
    eps = 1e-8
    sample_entropies = -(probabilities * torch.log(probabilities + eps) + 
                         (1 - probabilities) * torch.log(1 - probabilities + eps))
    expected_entropy = sample_entropies.mean(dim=0)
    
    # Mutual information = predictive entropy - expected entropy
    mutual_info = pred_entropy - expected_entropy
    
    return mutual_info


def visualize_uncertainty(image, prediction, uncertainty, save_path=None):
    """
    Create visualization of prediction with uncertainty overlay
    
    Args:
        image: Original image (H, W)
        prediction: Model prediction (H, W)
        uncertainty: Uncertainty map (H, W)
        save_path: Optional path to save visualization
        
    Returns:
        visualization: Combined visualization
    """
    import matplotlib.pyplot as plt
    import cv2
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Uncertainty map
    im = axes[2].imshow(uncertainty, cmap='hot')
    axes[2].set_title('Uncertainty Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Overlay high uncertainty regions on prediction
    # Normalize uncertainty
    uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
    
    # Create RGB overlay
    pred_rgb = cv2.cvtColor((prediction * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    uncertainty_colored = cv2.applyColorMap((uncertainty_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    uncertainty_colored = cv2.cvtColor(uncertainty_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(pred_rgb, 0.6, uncertainty_colored, 0.4, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Prediction + Uncertainty')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    visualization = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return visualization


def get_high_uncertainty_regions(uncertainty, threshold_percentile=90):
    """
    Identify regions with high uncertainty
    
    Args:
        uncertainty: Uncertainty map (H, W)
        threshold_percentile: Percentile threshold for high uncertainty
        
    Returns:
        high_uncertainty_mask: Binary mask of high uncertainty regions
        threshold_value: Threshold value used
    """
    threshold_value = np.percentile(uncertainty, threshold_percentile)
    high_uncertainty_mask = (uncertainty > threshold_value).astype(np.uint8)
    
    return high_uncertainty_mask, threshold_value
