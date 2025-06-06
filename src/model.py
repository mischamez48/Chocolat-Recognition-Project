import torch
import torch.nn as nn
import torch.nn.functional as F


#=============================================================================
# SEGMENTATION MODEL
#=============================================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Corrected to handle proper channel dimensions
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation, we have (in_channels // 2) + skip_channels channels
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Get input dimensions
        input_size = x.size()[2:]  # Get height and width
        
        # Apply convolutions to input features and skip connection features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match the spatial dimensions of x1 if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # Element-wise sum followed by ReLU
        psi = self.relu(g1 + x1)
        
        # Channel-wise attention map
        psi = self.psi(psi)
        
        # Attention-weighted features
        return x * psi

class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates model for binary segmentation
    """
    def __init__(self, n_channels=3, n_classes=1, base_filters=32):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters*2)
        self.down2 = Down(base_filters*2, base_filters*4)
        self.down3 = Down(base_filters*4, base_filters*8)
        
        # Bridge
        self.down4 = Down(base_filters*8, base_filters*16)
        
        # Attention Gates
        self.attention1 = AttentionGate(F_g=base_filters*16, F_l=base_filters*8, F_int=base_filters*8)
        self.attention2 = AttentionGate(F_g=base_filters*8, F_l=base_filters*4, F_int=base_filters*4)
        self.attention3 = AttentionGate(F_g=base_filters*4, F_l=base_filters*2, F_int=base_filters*2)
        self.attention4 = AttentionGate(F_g=base_filters*2, F_l=base_filters, F_int=base_filters)
        
        # Decoder (upsampling path)
        self.up1 = Up(base_filters*16, base_filters*8, base_filters*8)
        self.up2 = Up(base_filters*8, base_filters*4, base_filters*4)
        self.up3 = Up(base_filters*4, base_filters*2, base_filters*2)
        self.up4 = Up(base_filters*2, base_filters, base_filters)
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)           # level 1 features
        x2 = self.down1(x1)        # level 2 features
        x3 = self.down2(x2)        # level 3 features
        x4 = self.down3(x3)        # level 4 features
        x5 = self.down4(x4)        # bottleneck features
        
        # Apply attention gates
        x4_att = self.attention1(g=x5, x=x4)
        
        # Decoder path with attention-gated skip connections
        x = self.up1(x5, x4_att)  # Up from bottleneck
        
        x3_att = self.attention2(g=x, x=x3)
        x = self.up2(x, x3_att)
        
        x2_att = self.attention3(g=x, x=x2)
        x = self.up3(x, x2_att)
        
        x1_att = self.attention4(g=x, x=x1)
        x = self.up4(x, x1_att)
        
        logits = self.outc(x)
        
        return logits
        
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

#=============================================================================
# FEATURE EXTRACTOR MODEL
#=============================================================================

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=13, class_names=None, dropout_rate=0.3):
        super(FeatureExtractor, self).__init__()
        self.class_names = class_names if class_names else []
        
        # CNN backbone with higher capacity
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(384)

        # Skip connection
        self.skip_conv = nn.Conv2d(192, 384, kernel_size=1)
        self.skip_bn = nn.BatchNorm2d(384)

        # Feature output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 384

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classification head
        self.fc1 = nn.Linear(384, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward_features(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Save for skip connection
        skip = self.skip_bn(self.skip_conv(x))

        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))

        # Fifth block with skip connection
        x = self.bn5(self.conv5(x))
        x = F.relu(x + skip)  # Add skip connection

        # Global pooling
        x = self.global_pool(x)
        features = x.view(x.size(0), -1)
        return features

    def forward(self, x):
        features = self.forward_features(x)

        # Apply dropout to features
        features = self.dropout1(features)

        # Two-layer classifier with dropout
        x = F.relu(self.fc_bn(self.fc1(features)))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x, features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
