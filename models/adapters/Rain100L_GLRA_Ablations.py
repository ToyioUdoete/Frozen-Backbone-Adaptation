import torch
import torch.nn as nn
import torch.nn.functional as F

from GLRA import SP, HVSP, DSP

try:
    from torchvision.ops import DeformConv2d as TorchvisionDeformConv2d
except Exception:
    TorchvisionDeformConv2d = None


class DirectionalGLRA(nn.Module):
    """
    GLRA ablation variant emphasizing directional feature extraction.
    Uses anisotropic depthwise branches (1x7, 7x1) plus diagonal strip pooling.
    """

    def __init__(self, inplanes):
        super().__init__()

        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)

        self.branch_h = nn.Sequential(
            nn.Conv2d(midplanes, midplanes, kernel_size=(1, 7), padding=(0, 3), groups=midplanes, bias=False),
            nn.Conv2d(midplanes, midplanes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(midplanes, midplanes, kernel_size=(7, 1), padding=(3, 0), groups=midplanes, bias=False),
            nn.Conv2d(midplanes, midplanes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, True),
        )

        self.dsp = DSP(midplanes, k_size=3, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=3, diagonal=True, types=True)
        self.sp_h = SP(midplanes, k_size=3, types=True)
        self.sp_v = SP(midplanes, k_size=3, types=False)
        self.sp_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.conv = nn.Conv2d(5 * midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.input_relu(self.input_conv(x))

        sp = self.sp_fusion(self.sp_h(feat) * self.sp_v(feat))
        h = self.branch_h(feat)
        v = self.branch_v(feat)
        d = self.dsp(feat)
        ad = self.adsp(feat)

        mix = torch.cat([sp, h, v, d, ad], dim=1)
        gate = self.sigmoid(self.conv(mix))
        out = gate * x
        return nn.LeakyReLU(0.2, True)(out + x)


class DeformableConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        if TorchvisionDeformConv2d is not None:
            self.conv = TorchvisionDeformConv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
            self.use_deform = True
        else:
            self.conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
            self.use_deform = False

        self.offset = nn.Conv2d(
            channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)

    def forward(self, x):
        offs = self.offset(x)
        if self.use_deform:
            return self.conv(x, offs)
        return self.conv(x)


class DeformableGLRA(nn.Module):
    """
    GLRA ablation variant using deformable branch processing to adapt
    to irregular rain-streak geometry before attention gating.
    """

    def __init__(self, inplanes):
        super().__init__()

        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)

        self.sp_h = SP(midplanes, k_size=3, types=True)
        self.sp_v = SP(midplanes, k_size=3, types=False)
        self.sp_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.hor = HVSP(midplanes, k_size=3, types=True)
        self.ver = HVSP(midplanes, k_size=3, types=False)
        self.dsp = DSP(midplanes, k_size=3, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=3, diagonal=True, types=True)

        self.deform_refine = nn.Sequential(
            DeformableConv2d(5 * midplanes, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(5 * midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.input_relu(self.input_conv(x))

        sp = self.sp_fusion(self.sp_h(feat) * self.sp_v(feat))
        h = self.hor(feat)
        v = self.ver(feat)
        d = self.dsp(feat)
        ad = self.adsp(feat)

        mix = torch.cat([sp, h, v, d, ad], dim=1)
        gate = self.sigmoid(self.deform_refine(mix))
        out = gate * x
        return nn.LeakyReLU(0.2, True)(out + x)


class DynamicScaleMixer(nn.Module):
    def __init__(self, channels, dilation_rates=(1, 3, 5)):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.num_scales = len(dilation_rates)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False)
                for d in dilation_rates
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(self.num_scales)])
        self.act = nn.LeakyReLU(0.2, True)

        hidden = max(16, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        b = x.shape[0]
        weights = self.gate(x).view(b, self.num_scales, 1, 1, 1)

        feats = []
        for conv, bn in zip(self.convs, self.bns):
            feats.append(self.act(bn(conv(x))))
        stacked = torch.stack(feats, dim=1)
        return torch.sum(stacked * weights, dim=1)


class DynamicGLRA(nn.Module):
    """
    GLRA ablation variant with dynamic dilation selection.
    Uses an adaptive multi-scale mixer on branch features before gating.
    """

    def __init__(self, inplanes):
        super().__init__()

        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)

        self.sp_h = SP(midplanes, k_size=3, types=True)
        self.sp_v = SP(midplanes, k_size=3, types=False)
        self.sp_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.hor = HVSP(midplanes, k_size=3, types=True)
        self.ver = HVSP(midplanes, k_size=3, types=False)
        self.dsp = DSP(midplanes, k_size=3, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=3, diagonal=True, types=True)

        self.dynamic = DynamicScaleMixer(5 * midplanes, dilation_rates=(1, 3, 5))
        self.proj = nn.Conv2d(5 * midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.input_relu(self.input_conv(x))

        sp = self.sp_fusion(self.sp_h(feat) * self.sp_v(feat))
        h = self.hor(feat)
        v = self.ver(feat)
        d = self.dsp(feat)
        ad = self.adsp(feat)

        mix = torch.cat([sp, h, v, d, ad], dim=1)
        mix = self.dynamic(mix)
        gate = self.sigmoid(self.proj(mix))
        out = gate * x
        return nn.LeakyReLU(0.2, True)(out + x)
