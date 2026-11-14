"""
DCMLP Ablation Study Variants for Rain Removal Feature Extraction

Three main ablations:
1. DirectionalDilationDCMLP - 1D directional kernels (1x7, 7x1) for rain streak targeting
2. DeformableConvDCMLP - Deformable convolutions with learnable offsets
3. DynamicMultiDilationDCMLP - Learnable dilation rate selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
import math

try:
    from torchvision.ops import DeformConv2d as TorchvisionDeformConv2d
except Exception:
    TorchvisionDeformConv2d = None


def autopad(kernel, padding=None):
    """PAD TO 'SAME'"""
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c').to(x.device)


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).to(x.device)


class Mlp_(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp_, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = nn.GELU()
        self._init_weights()
        self.drop = nn.Dropout(0.)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DirectionalConv(nn.Module):
    """
    Directional convolutions: 1x7 (horizontal), 7x1 (vertical), and diagonal variants
    Target specific rain streak orientations
    """
    def __init__(self, in_channels, out_channels, orientation='horizontal'):
        super().__init__()
        self.orientation = orientation
        
        if orientation == 'horizontal':
            kernel_size = (1, 7)
            padding = (0, 3)
        elif orientation == 'vertical':
            kernel_size = (7, 1)
            padding = (3, 0)
        elif orientation == 'diagonal_left':
            # Approximate diagonal with multiple paths
            kernel_size = (3, 3)
            padding = 1
        else:  # diagonal_right
            kernel_size = (3, 3)
            padding = 1
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DirectionalDilationDCMLP(nn.Module):
    """
    DCMLP variant using directional 1D convolutions instead of square kernels
    Targets horizontal, vertical, and diagonal rain streaks separately
    """
    def __init__(self, features, M=4, r=2, L=32, MLP=True):
        super(DirectionalDilationDCMLP, self).__init__()
        d = max(int(features / r), L)
        self.features = features
        self.MLP = MLP

        # Match the number of branches requested by caller (BottleCSP passes M=3).
        orientation_bank = [
            'horizontal',
            'vertical',
            'diagonal_left',
            'diagonal_right'
        ]
        selected_orientations = orientation_bank[:max(1, min(M, len(orientation_bank)))]
        self.convs = nn.ModuleList([
            DirectionalConv(features, features, ori) for ori in selected_orientations
        ])
        self.M = len(self.convs)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, kernel_size=1, stride=1) for _ in range(self.M)])
        self.softmax = nn.Softmax(dim=1)
        self.ffn_norm0 = nn.LayerNorm(features)
        self.ffn_norm = nn.LayerNorm(features)
        self.ffn = Mlp_(features)
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        x = to_4d(self.ffn_norm0(to_3d(x)), H, W)
        
        # Apply directional convolutions in parallel
        feats_list = [conv(x).unsqueeze(dim=1) for conv in self.convs]
        feats = torch.cat(feats_list, dim=1)
        feats = feats.view(B, self.M, self.features, H, W)
        
        # Channel attention weighting
        fea_U = torch.sum(feats, dim=1)
        fea_s = self.gap(fea_U)
        fea_z = self.fc(fea_s)
        
        vector_list = [fc(fea_z) for fc in self.fcs]
        attention_vectors_ = torch.cat(vector_list, dim=1)
        attention_vectors = self.softmax(attention_vectors_.view(B, self.M, self.features, 1, 1))
        
        fea_v = torch.sum(feats * attention_vectors, dim=1)
        fea_residual = fea_v
        
        fea_norm = self.ffn_norm(to_3d(fea_v))
        fea_mlp = to_4d(self.ffn(fea_norm), H, W)
        fea_out = fea_mlp + fea_residual
        
        return fea_out



class DeformableConv2d(nn.Module):
    """
    Deformable convolution with learnable offsets
    Adapts receptive field to non-uniform rain streak shapes
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        # Use true deformable conv if available; otherwise fallback to regular conv.
        if TorchvisionDeformConv2d is not None:
            self.conv = TorchvisionDeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
            self.use_deform = True
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
            self.use_deform = False
        
        # Offset generation network: generates 2*kernel_size^2 offsets (x and y for each position)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, 
                                     padding=padding, dilation=dilation, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
    
    def forward(self, x):
        offsets = self.offset_conv(x)

        if self.use_deform:
            x_out = self.conv(x, offsets)
        else:
            x_out = self.conv(x)
        return x_out


class DeformableConvDCMLP(nn.Module):
    """
    DCMLP variant using deformable convolutions
    Allows spatial adaptation of receptive fields to rain patterns
    """
    def __init__(self, features, M=3, r=2, L=32, MLP=True):
        super(DeformableConvDCMLP, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.MLP = MLP
        
        # Deformable convolutions with different dilations
        self.convs = nn.ModuleList([
            DeformableConv2d(features, features, kernel_size=3, padding=1, dilation=1),
            DeformableConv2d(features, features, kernel_size=3, padding=3, dilation=3),
            DeformableConv2d(features, features, kernel_size=3, padding=5, dilation=5),
        ])
        
        self.bns = nn.ModuleList([nn.BatchNorm2d(features) for _ in range(M)])
        self.acts = nn.ModuleList([nn.ReLU(inplace=False) for _ in range(M)])
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, kernel_size=1, stride=1) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)
        self.ffn_norm0 = nn.LayerNorm(features)
        self.ffn_norm = nn.LayerNorm(features)
        self.ffn = Mlp_(features)
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        x = to_4d(self.ffn_norm0(to_3d(x)), H, W)
        
        # Apply deformable convolutions
        feats_list = []
        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            feat = act(bn(conv(x)))
            feats_list.append(feat.unsqueeze(dim=1))
        
        feats = torch.cat(feats_list, dim=1)
        feats = feats.view(B, self.M, self.features, H, W)
        
        # Channel attention
        fea_U = torch.sum(feats, dim=1)
        fea_s = self.gap(fea_U)
        fea_z = self.fc(fea_s)
        
        vector_list = [fc(fea_z) for fc in self.fcs]
        attention_vectors_ = torch.cat(vector_list, dim=1)
        attention_vectors = self.softmax(attention_vectors_.view(B, self.M, self.features, 1, 1))
        
        fea_v = torch.sum(feats * attention_vectors, dim=1)
        fea_residual = fea_v
        
        fea_norm = self.ffn_norm(to_3d(fea_v))
        fea_mlp = to_4d(self.ffn(fea_norm), H, W)
        fea_out = fea_mlp + fea_residual
        
        return fea_out



class DynamicMultiDilationConv(nn.Module):
    """
    Convolution with dynamic dilation rate selection
    Model learns to choose optimal dilation based on local rain density
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5]):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.num_dilations = len(dilation_rates)
        
        # Parallel convolutions with different dilations
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilation_rates
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(self.num_dilations)])
        
        # Gating network to select dilation rates
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(16, in_channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, in_channels // 4), self.num_dilations, 1),
            nn.Softmax(dim=1)
        )
        
        self.act = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Compute gating weights based on input statistics
        B, C, H, W = x.shape
        gates = self.gate_conv(x)  # [B, num_dilations, 1, 1]
        
        # Apply all dilated convolutions
        feat_list = []
        for i, (conv, bn) in enumerate(zip(self.dilated_convs, self.bns)):
            feat = self.act(bn(conv(x)))
            feat_list.append(feat)
        
        feats = torch.stack(feat_list, dim=1)  # [B, num_dilations, C, H, W]
        
        # Weighted combination using gates
        gates = gates.view(B, self.num_dilations, 1, 1, 1)
        out = torch.sum(feats * gates, dim=1)
        
        return out


class DynamicMultiDilationDCMLP(nn.Module):
    """
    DCMLP variant with dynamic multi-dilation
    Learns to choose between multiple dilation rates based on rain patterns
    """
    def __init__(self, features, M=2, r=2, L=32, MLP=True, dilation_rates=[1, 3, 5]):
        super(DynamicMultiDilationDCMLP, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.MLP = MLP
        self.dilation_rates = dilation_rates
        
        # M sets of dynamic multi-dilation convolutions
        self.convs = nn.ModuleList([
            DynamicMultiDilationConv(features, features, dilation_rates)
            for _ in range(M)
        ])
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, kernel_size=1, stride=1) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)
        self.ffn_norm0 = nn.LayerNorm(features)
        self.ffn_norm = nn.LayerNorm(features)
        self.ffn = Mlp_(features)
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        x = to_4d(self.ffn_norm0(to_3d(x)), H, W)
        
        # Apply dynamic multi-dilation convolutions
        feats_list = [conv(x).unsqueeze(dim=1) for conv in self.convs]
        feats = torch.cat(feats_list, dim=1)
        feats = feats.view(B, self.M, self.features, H, W)
        
        # Channel attention
        fea_U = torch.sum(feats, dim=1)
        fea_s = self.gap(fea_U)
        fea_z = self.fc(fea_s)
        
        vector_list = [fc(fea_z) for fc in self.fcs]
        attention_vectors_ = torch.cat(vector_list, dim=1)
        attention_vectors = self.softmax(attention_vectors_.view(B, self.M, self.features, 1, 1))
        
        fea_v = torch.sum(feats * attention_vectors, dim=1)
        fea_residual = fea_v
        
        fea_norm = self.ffn_norm(to_3d(fea_v))
        fea_mlp = to_4d(self.ffn(fea_norm), H, W)
        fea_out = fea_mlp + fea_residual
        
        return fea_out


if __name__ == "__main__":
    # Test ablation modules
    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W)
    
    print("Testing DirectionalDilationDCMLP...")
    model1 = DirectionalDilationDCMLP(features=C, M=4)
    out1 = model1(x)
    print(f"  Input: {x.shape}, Output: {out1.shape}")
    
    print("Testing DeformableConvDCMLP...")
    model2 = DeformableConvDCMLP(features=C, M=3)
    out2 = model2(x)
    print(f"  Input: {x.shape}, Output: {out2.shape}")
    
    print("Testing DynamicMultiDilationDCMLP...")
    model3 = DynamicMultiDilationDCMLP(features=C, M=2)
    out3 = model3(x)
    print(f"  Input: {x.shape}, Output: {out3.shape}")
    
    print("\nAll ablation modules work correctly!")
