import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from src.data_subtype import get_subtype_dataloaders
from tqdm import tqdm

def build_classifier(num_classes=4, in_channels=2, device='cpu'):
    """
    Loads pretrained DenseNet121 and modifies it for multi-channel input.
    """
    print(f"Loading Pretrained DenseNet121 for {in_channels}-channel input on {device}...")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # 1. Modify first convolution layer
    original_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Initialize the new conv0 weights
    with torch.no_grad():
        # Reuse available channels if possible
        c_to_copy = min(in_channels, 3)
        model.features.conv0.weight[:, :c_to_copy, :, :] = original_conv.weight[:, :c_to_copy, :, :]
        # If we had a 3rd channel (mask), we might initialize it randomly or with some mean
        
    # Freeze early layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Keep the new conv0 and last block trainable
    model.features.conv0.weight.requires_grad = True
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
        
    # Modify final classification layer
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)


def train_classifier(epochs=15, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # 1. Load Data
    train_loader, val_loader = get_subtype_dataloaders(batch_size=batch_size)
    
    if train_loader is None or val_loader is None:
        print("Dataset failed to load. Aborting training.")
        return
    
    # Get Metadata
    try:
        class_to_idx = train_loader.dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = [idx_to_class[i] for i in range(len(class_to_idx))]
        num_classes = len(classes)
        print(f"Detected {num_classes} classes: {class_to_idx}")
    except:
        num_classes = 4
        classes = ['adenocarcinoma', 'large_cell', 'normal', 'squamous_cell']

    # 2. Build Model
    model = build_classifier(num_classes=num_classes, device=device)
    
    # 3. Calculate Class Weights for Imbalance
    # We'll do a simple frequency based inverse weighting
    all_labels = [label for _, label in train_loader.dataset.base_ds.samples]
    class_counts = np.bincount(all_labels)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class Weights applied: {weights}")

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_val_acc = 0.0
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "classifier_densenet121_v2.pth")
    
    # 5. Training Loop
    print("\nStarting Phase 2 Training (Cropped + 2-Channel)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} Results:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"⭐ Saving improved model (v2) to {save_path}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'class_to_idx': class_to_idx,
                'val_acc': val_acc,
                'input_channels': 2
            }, save_path)
    
    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    train_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
