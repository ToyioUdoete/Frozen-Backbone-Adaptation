import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoomGLRA_v2(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 4)

        self.zoom_in = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mid_scale = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, dilation=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.zoom_out_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.zoom_out_v = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.zoom_out_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.density_skip_in = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.density_skip_mid = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.density_skip_out = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)

        self.density_map = nn.Sequential(
            nn.Conv2d(channels + mid_channels * 3, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.head_density = nn.Conv2d(mid_channels, 3, kernel_size=1, bias=True)
        self.head_horizontal = nn.Conv2d(mid_channels, 3, kernel_size=1, bias=True)
        self.head_vertical = nn.Conv2d(mid_channels, 3, kernel_size=1, bias=True)

        self.num_heads = 3
        self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.scale_fusion = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.output_proj = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x

        zoom_in_feat = self.zoom_in(x)
        mid_feat = self.mid_scale(x)
        h_context = self.zoom_out_h(x)
        v_context = self.zoom_out_v(x)
        h_context = h_context.expand(b, c, h, w)
        v_context = v_context.expand(b, c, h, w)
        zoom_out_feat = self.zoom_out_fuse(torch.cat([h_context, v_context], dim=1))

        skip_in = self.density_skip_in(zoom_in_feat)
        skip_mid = self.density_skip_mid(mid_feat)
        skip_out = self.density_skip_out(zoom_out_feat)
        density_input = torch.cat([x, skip_in, skip_mid, skip_out], dim=1)
        density_features = self.density_map(density_input)

        gates_density = self.head_density(density_features)
        gates_horizontal = self.head_horizontal(density_features)
        gates_vertical = self.head_vertical(density_features)

        gates_all = torch.stack([gates_density, gates_horizontal, gates_vertical], dim=1)
        gates_all = F.softmax(gates_all, dim=1)

        head_weights_norm = F.softmax(self.head_weights, dim=0)
        gates = (gates_all * head_weights_norm.view(1, -1, 1, 1, 1)).sum(dim=1)

        scale_weights = F.softmax(self.scale_fusion, dim=0)

        blended = (gates[:, 0:1, :, :] * scale_weights[0] * zoom_in_feat +
                   gates[:, 1:2, :, :] * scale_weights[1] * mid_feat +
                   gates[:, 2:3, :, :] * scale_weights[2] * zoom_out_feat)
        out = self.output_proj(torch.cat([zoom_in_feat, mid_feat, zoom_out_feat], dim=1))

        return x_in + self.residual_weight * out


# Keep backward compatibility
class ZoomGLRA(nn.Module):
    """Alias for ZoomGLRA_v2 for backward compatibility"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.module = ZoomGLRA_v2(channels, reduction)

    def forward(self, x):
        return self.module(x)
