import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        """
        Compute Dice Loss
        Args:
            logits: Raw predictions from model [B, C, H, W]
            targets: Ground truth masks [B, C, H, W]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors
        batch_size = targets.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum(1)
        union = probs_flat.sum(1) + targets_flat.sum(1)
        
        # Calculate Dice loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        """
        Compute combined BCE and Dice Loss
        Args:
            logits: Raw predictions from model [B, C, H, W]
            targets: Ground truth masks [B, C, H, W]
        """
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        
        # Combine losses
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss
    
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    This loss function applies label smoothing to the target distribution. Instead of assigning
    a probability of 1 to the correct class and 0 to all others, it assigns a slightly lower
    probability (confidence) to the correct class and distributes the remaining probability mass
    uniformly across the other classes.

    Args:
        classes (int): The number of classes in the classification task.
        smoothing (float): The smoothing factor. Default is 0.1.

    Forward Args:
        pred (Tensor): Logits of shape (batch_size, classes).
        target (Tensor): Ground truth labels of shape (batch_size,).

    Returns:
        Tensor: The computed label smoothing loss.
    """
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
