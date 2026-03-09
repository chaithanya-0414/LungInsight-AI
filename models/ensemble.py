# models/ensemble.py
# Model ensemble for improved prediction accuracy

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ModelEnsemble:
    """
    Ensemble of multiple models for robust predictions
    Supports different aggregation strategies
    """
    
    def __init__(self, models, weights=None, strategy='average'):
        """
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
            strategy: Aggregation strategy ('average', 'voting', 'weighted', 'max')
        """
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        else:
            assert len(weights) == self.n_models, "Number of weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.strategy = strategy
    
    def predict(self, x, device='cpu', return_all=False):
        """
        Get ensemble prediction
        
        Args:
            x: Input tensor (B, C, H, W)
            device: Device to run inference on
            return_all: If True, return all individual predictions
            
        Returns:
            ensemble_pred: Ensemble prediction (B, 1, H, W)
            individual_preds: (Optional) Individual predictions from each model
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
        
        # Apply aggregation strategy
        if self.strategy == 'average' or self.strategy == 'weighted':
            # Weighted average
            weights_tensor = torch.tensor(self.weights).view(-1, 1, 1, 1, 1)
            ensemble_pred = (predictions * weights_tensor).sum(dim=0)
        
        elif self.strategy == 'voting':
            # Majority voting (binary)
            binary_preds = (predictions > 0.5).float()
            ensemble_pred = (binary_preds.sum(dim=0) > (self.n_models / 2)).float()
        
        elif self.strategy == 'max':
            # Maximum probability
            ensemble_pred = predictions.max(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if return_all:
            return ensemble_pred, predictions
        else:
            return ensemble_pred
    
    def predict_with_confidence(self, x, device='cpu'):
        """
        Get prediction with confidence score based on model agreement
        
        Args:
            x: Input tensor (B, C, H, W)
            device: Device to run inference on
            
        Returns:
            ensemble_pred: Ensemble prediction (B, 1, H, W)
            confidence: Confidence map (B, 1, H, W) - based on variance
        """
        ensemble_pred, all_preds = self.predict(x, device, return_all=True)
        
        # Compute variance as inverse of confidence
        variance = all_preds.var(dim=0)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance)
        
        return ensemble_pred, confidence
    
    @classmethod
    def from_checkpoints(cls, model_class, checkpoint_paths, device='cpu', **model_kwargs):
        """
        Create ensemble from saved model checkpoints
        
        Args:
            model_class: Model class (e.g., UNet)
            checkpoint_paths: List of paths to model checkpoints
            device: Device to load models on
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            ensemble: ModelEnsemble instance
        """
        models = []
        
        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            
            # Load checkpoint
            if Path(path).exists():
                state_dict = torch.load(path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models.append(model)
            else:
                print(f"Warning: Checkpoint not found: {path}")
        
        if len(models) == 0:
            raise ValueError("No valid checkpoints found")
        
        return cls(models)


class StackedEnsemble:
    """
    Stacked ensemble - uses a meta-model to combine predictions
    """
    
    def __init__(self, base_models, meta_model):
        """
        Args:
            base_models: List of base models
            meta_model: Meta-model that takes base predictions as input
        """
        self.base_models = base_models
        self.meta_model = meta_model
    
    def predict(self, x, device='cpu'):
        """
        Get stacked ensemble prediction
        
        Args:
            x: Input tensor (B, C, H, W)
            device: Device to run inference on
            
        Returns:
            final_pred: Final prediction from meta-model
        """
        # Get predictions from base models
        base_predictions = []
        
        with torch.no_grad():
            for model in self.base_models:
                model.eval()
                model.to(device)
                pred = model(x.to(device))
                pred_prob = torch.sigmoid(pred)
                base_predictions.append(pred_prob)
        
        # Concatenate base predictions
        stacked_input = torch.cat(base_predictions, dim=1)  # (B, n_models, H, W)
        
        # Meta-model prediction
        with torch.no_grad():
            self.meta_model.eval()
            self.meta_model.to(device)
            final_pred = self.meta_model(stacked_input.to(device))
            final_pred = torch.sigmoid(final_pred)
        
        return final_pred.cpu()


def train_ensemble_weights(models, val_loader, device='cpu', n_iterations=100):
    """
    Learn optimal ensemble weights using validation data
    
    Args:
        models: List of models
        val_loader: Validation data loader
        device: Device to run on
        n_iterations: Number of optimization iterations
        
    Returns:
        optimal_weights: Learned weights for each model
    """
    from torch.optim import Adam
    
    n_models = len(models)
    
    # Initialize learnable weights
    weights = torch.nn.Parameter(torch.ones(n_models) / n_models)
    optimizer = Adam([weights], lr=0.01)
    
    criterion = nn.BCELoss()
    
    for iteration in range(n_iterations):
        total_loss = 0.0
        
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions from all models
            predictions = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    pred = model(images)
                    pred_prob = torch.sigmoid(pred)
                    predictions.append(pred_prob)
            
            # Stack and weight
            predictions = torch.stack(predictions, dim=0)  # (n_models, B, 1, H, W)
            
            # Normalize weights with softmax
            normalized_weights = torch.softmax(weights, dim=0).view(-1, 1, 1, 1, 1)
            
            # Weighted ensemble
            ensemble_pred = (predictions * normalized_weights).sum(dim=0)
            
            # Compute loss
            loss = criterion(ensemble_pred, masks)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {total_loss / len(val_loader):.4f}")
    
    # Return normalized weights
    optimal_weights = torch.softmax(weights, dim=0).detach().cpu().numpy()
    
    return optimal_weights.tolist()


def diversity_score(predictions):
    """
    Compute diversity score among ensemble members
    Higher diversity generally leads to better ensemble performance
    
    Args:
        predictions: Predictions from all models (n_models, B, 1, H, W)
        
    Returns:
        diversity: Diversity score (higher is more diverse)
    """
    n_models = predictions.shape[0]
    
    # Compute pairwise disagreement
    disagreements = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pred_i = (predictions[i] > 0.5).float()
            pred_j = (predictions[j] > 0.5).float()
            
            # Disagreement rate
            disagreement = (pred_i != pred_j).float().mean()
            disagreements.append(disagreement.item())
    
    # Average disagreement
    diversity = np.mean(disagreements) if disagreements else 0.0
    
    return diversity
