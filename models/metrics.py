# models/metrics.py
# Comprehensive medical imaging metrics

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


class SegmentationMetrics:
    """
    Comprehensive metrics for medical image segmentation evaluation
    """
    
    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        """
        Dice Similarity Coefficient (DSC)
        
        Args:
            pred: Predicted binary mask (H, W) or (B, H, W)
            target: Ground truth binary mask (H, W) or (B, H, W)
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            dice: Dice coefficient [0, 1], higher is better
        """
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)
    
    @staticmethod
    def iou(pred, target, smooth=1e-6):
        """
        Intersection over Union (IoU) / Jaccard Index
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            smooth: Smoothing factor
            
        Returns:
            iou: IoU score [0, 1], higher is better
        """
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou_score = (intersection + smooth) / (union + smooth)
        return float(iou_score)
    
    @staticmethod
    def sensitivity(pred, target):
        """
        Sensitivity / Recall / True Positive Rate (TPR)
        Measures the proportion of actual positives correctly identified
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            sensitivity: Sensitivity score [0, 1], higher is better
        """
        pred = pred.flatten()
        target = target.flatten()
        
        tp = ((pred == 1) & (target == 1)).sum()
        fn = ((pred == 0) & (target == 1)).sum()
        
        if (tp + fn) == 0:
            return 0.0
        
        return float(tp / (tp + fn))
    
    @staticmethod
    def specificity(pred, target):
        """
        Specificity / True Negative Rate (TNR)
        Measures the proportion of actual negatives correctly identified
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            specificity: Specificity score [0, 1], higher is better
        """
        pred = pred.flatten()
        target = target.flatten()
        
        tn = ((pred == 0) & (target == 0)).sum()
        fp = ((pred == 1) & (target == 0)).sum()
        
        if (tn + fp) == 0:
            return 0.0
        
        return float(tn / (tn + fp))
    
    @staticmethod
    def precision(pred, target):
        """
        Precision / Positive Predictive Value (PPV)
        Measures the proportion of positive predictions that are correct
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            precision: Precision score [0, 1], higher is better
        """
        pred = pred.flatten()
        target = target.flatten()
        
        tp = ((pred == 1) & (target == 1)).sum()
        fp = ((pred == 1) & (target == 0)).sum()
        
        if (tp + fp) == 0:
            return 0.0
        
        return float(tp / (tp + fp))
    
    @staticmethod
    def f1_score(pred, target):
        """
        F1 Score (harmonic mean of precision and recall)
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            f1: F1 score [0, 1], higher is better
        """
        prec = SegmentationMetrics.precision(pred, target)
        sens = SegmentationMetrics.sensitivity(pred, target)
        
        if (prec + sens) == 0:
            return 0.0
        
        return float(2 * (prec * sens) / (prec + sens))
    
    @staticmethod
    def hausdorff_distance(pred, target, percentile=95):
        """
        Hausdorff Distance (95th percentile)
        Measures the maximum distance between boundary points
        
        Args:
            pred: Predicted binary mask (H, W)
            target: Ground truth binary mask (H, W)
            percentile: Percentile to use (95 is standard)
            
        Returns:
            hd: Hausdorff distance in pixels, lower is better
        """
        # Ensure numpy arrays
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # If either mask is empty, return infinity
        if not pred.any() or not target.any():
            return float('inf')
        
        # Compute distance transforms
        dt_pred = distance_transform_edt(~pred)
        dt_target = distance_transform_edt(~target)
        
        # Get boundary points
        pred_boundary = pred & ~distance_transform_edt(pred).astype(bool)
        target_boundary = target & ~distance_transform_edt(target).astype(bool)
        
        # Distances from pred boundary to target
        dist_pred_to_target = dt_target[pred_boundary]
        
        # Distances from target boundary to pred
        dist_target_to_pred = dt_pred[target_boundary]
        
        # Combine and compute percentile
        all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
        
        if len(all_distances) == 0:
            return 0.0
        
        hd = np.percentile(all_distances, percentile)
        return float(hd)
    
    @staticmethod
    def average_surface_distance(pred, target):
        """
        Average Surface Distance (ASD)
        Average distance between boundary points
        
        Args:
            pred: Predicted binary mask (H, W)
            target: Ground truth binary mask (H, W)
            
        Returns:
            asd: Average surface distance in pixels, lower is better
        """
        # Ensure numpy arrays
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # If either mask is empty, return infinity
        if not pred.any() or not target.any():
            return float('inf')
        
        # Compute distance transforms
        dt_pred = distance_transform_edt(~pred)
        dt_target = distance_transform_edt(~target)
        
        # Get boundary points
        pred_boundary = pred & ~distance_transform_edt(pred).astype(bool)
        target_boundary = target & ~distance_transform_edt(target).astype(bool)
        
        # Distances
        dist_pred_to_target = dt_target[pred_boundary]
        dist_target_to_pred = dt_pred[target_boundary]
        
        # Average
        all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
        
        if len(all_distances) == 0:
            return 0.0
        
        asd = np.mean(all_distances)
        return float(asd)
    
    @staticmethod
    def volumetric_similarity(pred, target):
        """
        Volumetric Similarity Coefficient
        Measures volume agreement
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            vs: Volumetric similarity [-1, 1], higher is better
        """
        pred_vol = pred.sum()
        target_vol = target.sum()
        
        if (pred_vol + target_vol) == 0:
            return 1.0
        
        vs = 1.0 - abs(pred_vol - target_vol) / (pred_vol + target_vol)
        return float(vs)
    
    @staticmethod
    def compute_all_metrics(pred, target):
        """
        Compute all metrics at once
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            metrics: Dictionary with all metrics
        """
        # Ensure binary masks
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = (pred > 0.5).astype(np.float32)
        target = (target > 0.5).astype(np.float32)
        
        metrics = {
            'dice': SegmentationMetrics.dice_coefficient(pred, target),
            'iou': SegmentationMetrics.iou(pred, target),
            'sensitivity': SegmentationMetrics.sensitivity(pred, target),
            'specificity': SegmentationMetrics.specificity(pred, target),
            'precision': SegmentationMetrics.precision(pred, target),
            'f1_score': SegmentationMetrics.f1_score(pred, target),
            'hausdorff_95': SegmentationMetrics.hausdorff_distance(pred, target, 95),
            'avg_surface_distance': SegmentationMetrics.average_surface_distance(pred, target),
            'volumetric_similarity': SegmentationMetrics.volumetric_similarity(pred, target)
        }
        
        return metrics


def compute_roc_curve(y_true, y_pred_proba):
    """
    Compute ROC curve and AUC
    
    Args:
        y_true: Ground truth binary labels (flattened)
        y_pred_proba: Predicted probabilities (flattened)
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred_proba.flatten())
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, thresholds


def compute_pr_curve(y_true, y_pred_proba):
    """
    Compute Precision-Recall curve
    
    Args:
        y_true: Ground truth binary labels (flattened)
        y_pred_proba: Predicted probabilities (flattened)
        
    Returns:
        precision: Precision values
        recall: Recall values
        pr_auc: Area under PR curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_pred_proba.flatten())
    pr_auc = auc(recall, precision)
    
    return precision, recall, pr_auc, thresholds


def compute_confusion_matrix(pred, target):
    """
    Compute confusion matrix
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        cm: Confusion matrix [[TN, FP], [FN, TP]]
    """
    pred = pred.flatten()
    target = target.flatten()
    
    cm = confusion_matrix(target, pred, labels=[0, 1])
    
    return cm
