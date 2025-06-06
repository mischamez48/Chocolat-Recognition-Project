import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#=============================================================================
# TRAINING FUNCTIONS FOR SEGMENTATION MODEL
#=============================================================================

def train_segmentation_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                            device, num_epochs=30, patience=7, is_cosine_scheduler=True,
                            checkpoint_path='src/checkpoints/best_segmentation_model.pth'):
    """
    Train the segmentation model with monitoring and early stopping
    
    Args:
        model: The UNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function (BCEDiceLoss)
        optimizer: Optimizer (Adam or AdamW)
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Maximum number of epochs to train for
        patience: Number of epochs to wait for improvement before early stopping
        is_cosine_scheduler: Whether the scheduler is cosine-based
        checkpoint_path: Path to save the best model
        
    Returns:
        history: Dictionary containing training history
        model: The trained model
    """
    # Create directory for checkpoints if needed
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Initialize history dictionary with all metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'lr': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_val_dice = 0.0
    counter = 0
    best_epoch = 0
    
    print(f"Starting training with early stopping (patience={patience}, monitoring Dice)")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou_values = []
        train_dice_values = []
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            images = images.to(device)
            
            # Handle masks based on format (ensure correct dimensions)
            if isinstance(masks, torch.Tensor) and masks.dim() == 3:
                # Add channel dimension to masks and move to device
                masks = masks.unsqueeze(1).to(device)  # Convert [B, H, W] to [B, 1, H, W]
            else:
                masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # If output size doesn't match target size, resize the output
            if outputs.shape != masks.shape:
                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
            # Calculate loss    
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping for more stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics for this batch
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            
            # IoU
            intersection = (pred_masks * masks).sum((1, 2, 3))
            union = pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
            batch_iou = (intersection / (union + 1e-8)).mean().item()
            
            # Dice coefficient = 2*intersection / (sum1 + sum2)
            dice_coeff = (2 * intersection) / (pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-8)
            batch_dice = dice_coeff.mean().item()
            
            # Update statistics
            train_loss += loss.item() * images.size(0)
            train_iou_values.append(batch_iou)
            train_dice_values.append(batch_dice)
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_iou = np.mean(train_iou_values)
        avg_train_dice = np.mean(train_dice_values)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou_values = []
        val_dice_values = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                images = images.to(device)
                
                # Handle masks based on format
                if isinstance(masks, torch.Tensor) and masks.dim() == 3:
                    # Add channel dimension to masks and move to device
                    masks = masks.unsqueeze(1).to(device)  # Convert [B, H, W] to [B, 1, H, W]
                else:
                    masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # If output size doesn't match target size, resize the output
                if outputs.shape != masks.shape:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Calculate metrics
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                
                # IoU
                intersection = (pred_masks * masks).sum((1, 2, 3))
                union = pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
                batch_iou = (intersection / (union + 1e-8)).mean().item()
                
                # Dice coefficient
                dice_coeff = (2 * intersection) / (pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-8)
                batch_dice = dice_coeff.mean().item()
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
                val_iou_values.append(batch_iou)
                val_dice_values.append(batch_dice)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_iou = np.mean(val_iou_values)
        avg_val_dice = np.mean(val_dice_values)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)
        
        # Update scheduler based on scheduler type
        if is_cosine_scheduler:
            # For cosine scheduler: step every epoch regardless of performance
            scheduler.step()
        else:
            # For ReduceLROnPlateau: step based on validation metric
            scheduler.step(avg_val_dice)
        
        # Check for improvement in Dice score (higher is better)
        is_best_loss = avg_val_loss < best_val_loss
        is_best_dice = avg_val_dice > best_val_dice
        
        if is_best_loss:
            best_val_loss = avg_val_loss
        
        if is_best_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            # Save best model checkpoint
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'dice': best_val_dice,
                    'iou': avg_val_iou,
                    'history': history
                }, checkpoint_path)
                print(f"Saved best model at epoch {epoch+1} with Dice: {best_val_dice:.4f}")
                counter = 0  # Reset patience counter
        
        # Early stopping check based on Dice score
        if not is_best_dice:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} - No improvement in Dice score for {patience} epochs")
                break
        else:
            counter = 0  # Reset counter if we found an improvement
        
        # Print progress with all metrics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}, "
              f"Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}, "
              f"LR: {current_lr:.6f}")
    
    # Save final training history to a file for later loading
    os.makedirs('src/checkpoints', exist_ok=True)
    np.save('src/checkpoints/segmentation_training_history.npy', history)
    print(f"Saved training history to src/checkpoints/segmentation_training_history.npy")
    
    # Load the best model if we did early stopping
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading best model from epoch {best_epoch+1} with Dice: {best_val_dice:.4f}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return history, model

def visualize_segmentation_history(history=None, history_path=None, save_path=None, title_suffix=""):
    """
    Visualize and save segmentation training metrics
    
    Args:
        history: Dictionary containing training metrics (optional)
        history_path: Path to a saved .npy history file (optional)
        save_path: Path to save the plot image (optional)
        title_suffix: Additional text to add to plot titles (e.g., model name)
        
    Returns:
        fig: The matplotlib figure object
    """
    # Load history from file if provided and history is not
    if history is None and history_path is not None:
        try:
            history = np.load(history_path, allow_pickle=True).item()
            print(f"Loaded training history from {history_path}")
        except Exception as e:
            print(f"Error loading history from {history_path}: {e}")
            return None
    
    if history is None:
        print("No history provided or loaded.")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Number of epochs (x-axis)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title(f'Loss{" " + title_suffix if title_suffix else ""}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Validation')
    axes[0, 1].set_title(f'IoU{" " + title_suffix if title_suffix else ""}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Dice coefficient
    axes[1, 0].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_dice'], 'r-', label='Validation')
    axes[1, 0].set_title(f'Dice Coefficient{" " + title_suffix if title_suffix else ""}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'g-')
    axes[1, 1].set_title(f'Learning Rate{" " + title_suffix if title_suffix else ""}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    if len(history['lr']) > 1:
        axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")
    
    # Return the figure object
    return fig

#=============================================================================
# TRAINING FUNCTIONS FOR FEATURE EXTRACTION MODEL
#=============================================================================

def calculate_direct_f1_score(y_true, y_pred):
    """
    Calculate F1 score directly based on the competition formula
    
    Args:
        y_true: Ground truth labels (tensor or numpy array)
        y_pred: Predicted labels (tensor or numpy array)
        
    Returns:
        mean_f1: Mean F1 score across all samples
    """
    # Move tensors to CPU and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if y_true is class indices (not one-hot)
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        # Convert to one-hot encoding
        batch_size = y_true.shape[0]
        num_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            if len(y_true.shape) == 1:
                y_true_onehot[i, int(y_true[i])] = 1
            else:
                y_true_onehot[i, int(y_true[i, 0])] = 1
        y_true = y_true_onehot

    # Check if y_pred is class indices (not one-hot)
    if len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1):
        # Convert to one-hot encoding
        batch_size = y_pred.shape[0]
        num_classes = y_true.shape[1]
        y_pred_onehot = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            if len(y_pred.shape) == 1:
                y_pred_onehot[i, int(y_pred[i])] = 1
            else:
                y_pred_onehot[i, int(y_pred[i, 0])] = 1
        y_pred = y_pred_onehot

    # Calculate TP and FPN for each image
    tp = np.sum(np.minimum(y_true, y_pred), axis=1)
    fpn = np.sum(np.abs(y_true - y_pred), axis=1)

    # Calculate F1 for each image
    f1 = (2 * tp) / (2 * tp + fpn + 1e-8)  # Add small epsilon to prevent division by zero

    # Average F1 across all images
    mean_f1 = np.mean(f1)
    return mean_f1

def train_feature_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                      class_names, num_epochs=20, save_dir='src/checkpoints',
                      early_stopping_patience=5, is_cosine_scheduler=False):
    """
    Train a feature extraction model while monitoring and optimizing for F1 score
    
    Args:
        model: The feature extraction model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        class_names: List of class names
        num_epochs: Maximum number of epochs to train for
        save_dir: Directory to save model checkpoints
        early_stopping_patience: Number of epochs to wait for improvement before early stopping
        is_cosine_scheduler: Whether the scheduler is cosine-based
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
        best_val_f1: Best validation F1 score
    """
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'learning_rates': []
    }

    print("\n Starting training...")
    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_f1_values = []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply gradient clipping at 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            targets_one_hot = torch.zeros(targets.size(0), len(class_names), device=device)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

            preds_one_hot = torch.zeros(targets.size(0), len(class_names), device=device)
            preds_one_hot.scatter_(1, predicted.unsqueeze(1), 1)

            batch_f1 = calculate_direct_f1_score(targets_one_hot, preds_one_hot)
            train_f1_values.append(batch_f1)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_f1 = np.mean(train_f1_values)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_f1_values = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                targets_one_hot = torch.zeros(targets.size(0), len(class_names), device=device)
                targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

                preds_one_hot = torch.zeros(targets.size(0), len(class_names), device=device)
                preds_one_hot.scatter_(1, predicted.unsqueeze(1), 1)

                batch_f1 = calculate_direct_f1_score(targets_one_hot, preds_one_hot)
                val_f1_values.append(batch_f1)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_f1 = np.mean(val_f1_values)

        # Update scheduler based on scheduler type
        if is_cosine_scheduler:
            # For cosine scheduler: step every epoch regardless of performance
            scheduler.step()
        else:
            # For ReduceLROnPlateau: step based on validation metric
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
              f"LR: {current_lr:.7f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/feature_extractor_best_loss.pth')
            print(f"Saved best loss model (val_loss: {val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'{save_dir}/feature_extractor_best_f1.pth')
            print(f"Saved best F1 model (val_f1: {val_f1:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{save_dir}/feature_extractor_best_acc.pth')
            print(f"Saved best accuracy model (val_acc: {val_acc:.4f})")

        # Check early stopping condition based on validation loss
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered. No validation loss improvement in {early_stopping_patience} epochs.")
            break

    # Load best model by F1 score
    model.load_state_dict(torch.load(f'{save_dir}/feature_extractor_best_f1.pth'))
    
    # Save training history
    np.save(f'{save_dir}/feature_training_history.npy', history)
    print(f"Saved training history to {save_dir}/feature_training_history.npy")

    return model, history, best_val_f1

def visualize_feature_history(history=None, history_path=None, save_path=None):
    """
    Visualize feature model training history
    
    Args:
        history: Dictionary containing training metrics (optional)
        history_path: Path to a saved .npy history file (optional)
        save_path: Path to save the plot image (optional)
        
    Returns:
        fig: The matplotlib figure object
    """
    # Load history from file if provided and history is not
    if history is None and history_path is not None:
        try:
            history = np.load(history_path, allow_pickle=True).item()
            print(f"Loaded training history from {history_path}")
        except Exception as e:
            print(f"Error loading history from {history_path}: {e}")
            return None
    
    if history is None:
        print("No history provided or loaded.")
        return None
    
    # Plot training history with F1 scores
    fig = plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')

    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")
    
    return fig