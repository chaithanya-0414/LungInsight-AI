# models/attention_unet.py
# Attention UNet for improved feature learning

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module
    Highlights salient features passed through skip connections
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature maps in gating signal (from decoder)
            F_l: Number of feature maps in skip connection (from encoder)
            F_int: Number of intermediate feature maps
        """
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
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)
            
        Returns:
            out: Attention-weighted features (B, F_l, H, W)
            attention: Attention coefficients (B, 1, H, W)
        """
        # Apply transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine and apply attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights
        out = x * psi
        
        return out, psi


class DoubleConv(nn.Module):
    """Double Convolution block"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for medical image segmentation
    
    Reference: Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    """
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 16)
        
        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=base * 8, F_l=base * 8, F_int=base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8)
        
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=base * 4, F_l=base * 4, F_int=base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=base * 2, F_l=base * 2, F_int=base)
        self.dec2 = DoubleConv(base * 4, base * 2)
        
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(F_g=base, F_l=base, F_int=base // 2)
        self.dec1 = DoubleConv(base * 2, base)
        
        # Output
        self.outc = nn.Conv2d(base, out_ch, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, in_ch, H, W)
            
        Returns:
            out: Output segmentation (B, out_ch, H, W)
        """
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder with Attention
        u4 = self.up4(bn)
        if u4.shape[2:] != c4.shape[2:]:
            u4 = F.interpolate(u4, size=c4.shape[2:], mode='bilinear', align_corners=False)
        c4_att, _ = self.att4(g=u4, x=c4)
        d4 = self.dec4(torch.cat([u4, c4_att], dim=1))
        
        u3 = self.up3(d4)
        if u3.shape[2:] != c3.shape[2:]:
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c3_att, _ = self.att3(g=u3, x=c3)
        d3 = self.dec3(torch.cat([u3, c3_att], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        c2_att, _ = self.att2(g=u2, x=c2)
        d2 = self.dec2(torch.cat([u2, c2_att], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c1_att, _ = self.att1(g=u1, x=c1)
        d1 = self.dec1(torch.cat([u1, c1_att], dim=1))
        
        # Output
        out = self.outc(d1)
        
        return out
    
    def forward_with_attention(self, x):
        """
        Forward pass that also returns attention maps
        
        Args:
            x: Input tensor (B, in_ch, H, W)
            
        Returns:
            out: Output segmentation (B, out_ch, H, W)
            attention_maps: Dictionary of attention maps from each level
        """
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        attention_maps = {}
        
        # Decoder with Attention
        u4 = self.up4(bn)
        if u4.shape[2:] != c4.shape[2:]:
            u4 = F.interpolate(u4, size=c4.shape[2:], mode='bilinear', align_corners=False)
        c4_att, att4 = self.att4(g=u4, x=c4)
        attention_maps['level4'] = att4
        d4 = self.dec4(torch.cat([u4, c4_att], dim=1))
        
        u3 = self.up3(d4)
        if u3.shape[2:] != c3.shape[2:]:
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c3_att, att3 = self.att3(g=u3, x=c3)
        attention_maps['level3'] = att3
        d3 = self.dec3(torch.cat([u3, c3_att], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        c2_att, att2 = self.att2(g=u2, x=c2)
        attention_maps['level2'] = att2
        d2 = self.dec2(torch.cat([u2, c2_att], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c1_att, att1 = self.att1(g=u1, x=c1)
        attention_maps['level1'] = att1
        d1 = self.dec1(torch.cat([u1, c1_att], dim=1))
        
        # Output
        out = self.outc(d1)
        
        return out, attention_maps
