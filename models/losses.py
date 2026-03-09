# models/losses.py
# Advanced loss functions for medical image segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        # Convert to probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha term
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss
        focal_loss = alpha_t * focal_term * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    Allows control over false positives and false negatives
    
    alpha: weight for false positives
    beta: weight for false negatives
    
    When alpha=beta=0.5, it's equivalent to Dice loss
    alpha > beta: penalize false positives more (increase precision)
    alpha < beta: penalize false negatives more (increase recall)
    
    Reference: Salehi et al. "Tversky loss function for image segmentation"
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky_index


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - combines Focal and Tversky losses
    Better for handling class imbalance and small ROIs
    
    Reference: Abraham & Khan "A Novel Focal Tversky Loss Function"
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Apply focal term
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky


class ComboLoss(nn.Module):
    """
    Combination of multiple losses for robust training
    Combines BCE, Dice, and Focal losses
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.0, bce_weight=1.0, dice_weight=1.0, focal_weight=0.0):
        super(ComboLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma) if focal_weight > 0 else None
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        loss = 0.0
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(pred, target)
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(pred, target)
        
        if self.focal_weight > 0 and self.focal is not None:
            loss += self.focal_weight * self.focal(pred, target)
        
        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - focuses on boundary accuracy
    Useful for improving segmentation edges
    
    Reference: Kervadec et al. "Boundary loss for highly unbalanced segmentation"
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, pred, target, distance_map):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
            distance_map: Distance transform of target (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        # Multiply prediction by distance map
        # Pixels far from boundary have higher distance values
        # Penalize errors more at boundaries (low distance values)
        loss = (pred * distance_map).mean()
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Measures structural similarity between images
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


def get_loss_function(loss_name='combo', **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        loss_fn: Loss function instance
    """
    loss_functions = {
        'bce': nn.BCEWithLogitsLoss(),
        'dice': DiceLoss(),
        'focal': FocalLoss(**kwargs),
        'tversky': TverskyLoss(**kwargs),
        'focal_tversky': FocalTverskyLoss(**kwargs),
        'combo': ComboLoss(**kwargs),
        'boundary': BoundaryLoss(),
        'ssim': SSIMLoss()
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]
