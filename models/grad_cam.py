# models/grad_cam.py
# Grad-CAM implementation for UNet interpretability

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for medical image segmentation.
    Visualizes which regions the model focuses on during prediction.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The layer to compute gradients from (e.g., model.dec1)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_mask=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_mask: Optional target mask for supervised CAM
            
        Returns:
            cam: Grad-CAM heatmap (H, W) normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If no target mask, use the prediction itself
        if target_mask is None:
            target_mask = torch.sigmoid(output)
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=target_mask, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            image: Original grayscale image (H, W) or (H, W, 1)
            cam: Grad-CAM heatmap (H, W)
            alpha: Transparency of overlay
            colormap: OpenCV colormap
            
        Returns:
            overlay: RGB image with heatmap overlay
        """
        # Ensure image is 2D
        if len(image.shape) == 3:
            image = image.squeeze()
        
        # Resize CAM to match image size
        if cam.shape != image.shape:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert to uint8
        cam_uint8 = np.uint8(255 * cam)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = np.uint8(255 * image)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = np.uint8(255 * image)
        
        # Overlay
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


class MultiLayerGradCAM:
    """
    Generate Grad-CAM for multiple layers and combine them
    """
    
    def __init__(self, model, target_layers):
        """
        Args:
            model: The neural network model
            target_layers: List of layers to compute Grad-CAM
        """
        self.grad_cams = [GradCAM(model, layer) for layer in target_layers]
    
    def generate_multi_cam(self, input_tensor, target_mask=None, weights=None):
        """
        Generate combined Grad-CAM from multiple layers
        
        Args:
            input_tensor: Input image tensor
            target_mask: Optional target mask
            weights: Optional weights for each layer (default: equal weights)
            
        Returns:
            combined_cam: Combined Grad-CAM heatmap
        """
        cams = []
        for grad_cam in self.grad_cams:
            cam = grad_cam.generate_cam(input_tensor, target_mask)
            cams.append(cam)
        
        # Combine CAMs
        if weights is None:
            weights = [1.0 / len(cams)] * len(cams)
        
        combined_cam = np.zeros_like(cams[0])
        for cam, weight in zip(cams, weights):
            # Resize to same size if needed
            if cam.shape != combined_cam.shape:
                cam = cv2.resize(cam, (combined_cam.shape[1], combined_cam.shape[0]))
            combined_cam += weight * cam
        
        # Normalize
        combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
        
        return combined_cam


def visualize_gradcam_comparison(image, mask, prediction, cam, save_path=None):
    """
    Create a comparison visualization with original, mask, prediction, and Grad-CAM
    
    Args:
        image: Original image (H, W)
        mask: Ground truth mask (H, W)
        prediction: Model prediction (H, W)
        cam: Grad-CAM heatmap (H, W)
        save_path: Optional path to save the visualization
        
    Returns:
        comparison: Combined visualization image
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Grad-CAM overlay
    grad_cam_obj = GradCAM(None, None)  # Dummy object for overlay method
    overlay = grad_cam_obj.overlay_heatmap(image, cam)
    axes[3].imshow(overlay)
    axes[3].set_title('Grad-CAM Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array for display
    fig.canvas.draw()
    comparison = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison = comparison.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison
