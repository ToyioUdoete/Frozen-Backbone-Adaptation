# Directional Pooling Attention

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torch.nn.modules.utils import _pair, _quadruple
from torchinfo import summary
from einops import rearrange

def inital_kernel(k_size, batch, channel, diagonal=False, types=False):
    if not diagonal:
        if types: #True for hor 
            kernel = (torch.ones(1, k_size)/k_size).cuda()
            m = nn.ZeroPad2d((0, 0, int(k_size//2), int(k_size//2))) # (left,right,top,bottom)
            kernel = m(kernel)
        else:
            kernel = (torch.ones(k_size, 1)/k_size).cuda()
            m = nn.ZeroPad2d((int(k_size//2), int(k_size//2), 0, 0)) # (left,right,top,bottom)
            kernel = m(kernel)
    else: # False for ani-diagonal
        kernel = torch.zeros(k_size, k_size).cuda()
        kernel = kernel.fill_diagonal_(1)/k_size
        if types:
            kernel = torch.flip(kernel, dims=[1])
    kernel = kernel.view(1, 1, k_size, k_size).repeat(batch, channel, 1, 1)
    return kernel


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
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

# Diagonal Strip pooling
class DSP(nn.Module):  
    def __init__(self, midplanes, k_size=9, diagonal=True, types=False):
        super(DSP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        self.kernel = inital_kernel(k_size=k_size, batch=midplanes, channel= midplanes, diagonal=diagonal, types=types)
        self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
    def forward(self, x):
        with torch.no_grad():
            diapool = F.conv2d(x, self.kernel, padding='same')
        # avg_result = self.norm(self.conv_1(diapool))
        avg_result = self.conv_1(diapool)

        return avg_result 

class HVSP(nn.Module):  
    def __init__(self, midplanes, k_size, types=False):
        super(HVSP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        if types == True: #True for hor; False for ver
            self.pool = nn.AvgPool2d((k_size, 1))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        else:
            self.pool = nn.AvgPool2d((1, k_size))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        _, _, h, w = x.size()
        # avg_result = self.norm(self.conv_1( self.pool(x)))
        avg_result = self.conv_1( self.pool(x))

        return F.interpolate(avg_result, (h, w))

class SP(nn.Module):  
    def __init__(self, midplanes, k_size, types=False):
        super(SP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        if types == True: #True for hor; False for ver
            self.pool = nn.AdaptiveAvgPool2d((None, 1))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, None))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
         
    def forward(self, x):
        _, _, h, w = x.size()
        # avg_result = self.norm(self.conv_1( self.pool(x)))
        avg_result = self.conv_1( self.pool(x))

        return avg_result.expand(-1, -1, h, w)# avg_result

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class GLRA(nn.Module):
    def __init__(self, inplanes):
        super(GLRA, self).__init__()

        midplanes = int(inplanes//4)
        # Input conv
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3

        self.SP_hor = SP(midplanes, k_size = ks, types=True)
        self.SP_ver = SP(midplanes, k_size = ks, types=False)
        # self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.SP_cat_proj = nn.Conv2d(2*midplanes, midplanes, kernel_size=3, padding=1, bias=False)  # variant 2: concat

        self.hor = HVSP(midplanes, k_size = ks, types=True)
        self.ver = HVSP(midplanes, k_size = ks, types=False)
        self.dsp = DSP(midplanes, k_size = ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size = ks, diagonal=True, types=True)
        
        self.conv = nn.Conv2d(5*midplanes, inplanes, kernel_size=3, stride = 1, padding = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        in_feature = self.input_relu((self.input_conv(x)))
        _, _, h, w = in_feature.size()

        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)
        # sp = self.SP_fusion(x_1_h_sp*x_1_w_sp)          # variant 0: multiplicative (original)
        # sp = self.SP_fusion(x_1_h_sp + x_1_w_sp)        # variant 1: additive
        sp = self.SP_cat_proj(torch.cat([x_1_h_sp, x_1_w_sp], dim=1))  # variant 2: concatenation
        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)

        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)   
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa+x)

class AdditiveFusionGLRA(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3
        self.SP_hor = SP(midplanes, k_size=ks, types=True)
        self.SP_ver = SP(midplanes, k_size=ks, types=False)
        self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.hor = HVSP(midplanes, k_size=ks, types=True)
        self.ver = HVSP(midplanes, k_size=ks, types=False)
        self.dsp = DSP(midplanes, k_size=ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=ks, diagonal=True, types=True)
        self.conv = nn.Conv2d(5*midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_feature = self.input_relu(self.input_conv(x))
        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)
        sp = self.SP_fusion(x_1_h_sp + x_1_w_sp)
        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)
        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa + x)


class ConcatFusionGLRA(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3
        self.SP_hor = SP(midplanes, k_size=ks, types=True)
        self.SP_ver = SP(midplanes, k_size=ks, types=False)
        self.SP_cat_proj = nn.Conv2d(2*midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.hor = HVSP(midplanes, k_size=ks, types=True)
        self.ver = HVSP(midplanes, k_size=ks, types=False)
        self.dsp = DSP(midplanes, k_size=ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=ks, diagonal=True, types=True)
        self.conv = nn.Conv2d(5*midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_feature = self.input_relu(self.input_conv(x))
        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)
        sp = self.SP_cat_proj(torch.cat([x_1_h_sp, x_1_w_sp], dim=1))
        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)
        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa + x)


class GatedFusionGLRA(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        midplanes = int(inplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3
        self.SP_hor = SP(midplanes, k_size=ks, types=True)
        self.SP_ver = SP(midplanes, k_size=ks, types=False)
        self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.gate_conv = nn.Conv2d(2*midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.hor = HVSP(midplanes, k_size=ks, types=True)
        self.ver = HVSP(midplanes, k_size=ks, types=False)
        self.dsp = DSP(midplanes, k_size=ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=ks, diagonal=True, types=True)
        self.conv = nn.Conv2d(5*midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_feature = self.input_relu(self.input_conv(x))
        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)
        gate = torch.sigmoid(self.gate_conv(torch.cat([x_1_h_sp, x_1_w_sp], dim=1)))
        sp = self.SP_fusion(gate * x_1_h_sp + (1 - gate) * x_1_w_sp)
        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)
        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa + x)


class ResidualGatedFusionGLRA(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        midplanes = int(inplanes // 4)
        gate_hidden = max(1, midplanes // 4)
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3
        self.SP_hor = SP(midplanes, k_size=ks, types=True)
        self.SP_ver = SP(midplanes, k_size=ks, types=False)
        self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.spatial_gate_conv = nn.Conv2d(3 * midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * midplanes, gate_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, midplanes, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.hor = HVSP(midplanes, k_size=ks, types=True)
        self.ver = HVSP(midplanes, k_size=ks, types=False)
        self.dsp = DSP(midplanes, k_size=ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size=ks, diagonal=True, types=True)
        self.conv = nn.Conv2d(5 * midplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_feature = self.input_relu(self.input_conv(x))
        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)

        strip_pair = torch.cat([x_1_h_sp, x_1_w_sp], dim=1)
        spatial_gate = torch.sigmoid(
            self.spatial_gate_conv(torch.cat([in_feature, x_1_h_sp, x_1_w_sp], dim=1))
        )
        channel_gate = self.channel_gate(strip_pair)
        gated_mix = spatial_gate * x_1_h_sp + (1 - spatial_gate) * x_1_w_sp
        multiplicative_path = x_1_h_sp * x_1_w_sp
        sp = self.SP_fusion(multiplicative_path + channel_gate * gated_mix)

        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)
        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa + x)


class ChannelAttnGLRA(nn.Module):
    """Variant 8: SE-block channel attention replacing strip pooling."""
    def __init__(self, inplanes):
        super().__init__()
        r = max(1, inplanes // 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(inplanes, r, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(r, inplanes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return nn.LeakyReLU(0.2, True)(w * x + x)


class SelfAttnGLRA(nn.Module):
    """Variant 9: Global self-attention with downsampled k/v for memory efficiency."""
    def __init__(self, inplanes):
        super().__init__()
        midplanes = inplanes // 4
        self.input_conv = nn.Conv2d(inplanes, midplanes, 1)
        self.q_proj = nn.Conv2d(midplanes, midplanes, 1)
        self.k_proj = nn.Conv2d(midplanes, midplanes, 1)
        self.v_proj = nn.Conv2d(midplanes, midplanes, 1)
        self.out_proj = nn.Conv2d(midplanes, inplanes, 1)
        self.sigmoid = nn.Sigmoid()
        self.scale = midplanes ** -0.5
        self.pool_size = 16

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.input_conv(x)
        Cm = feat.shape[1]

        q = self.q_proj(feat).reshape(B, Cm, H * W).permute(0, 2, 1)  # B, HW, Cm

        feat_down = F.adaptive_avg_pool2d(feat, self.pool_size)
        k = self.k_proj(feat_down).reshape(B, Cm, -1)           # B, Cm, P^2
        v = self.v_proj(feat_down).reshape(B, Cm, -1).permute(0, 2, 1)  # B, P^2, Cm

        attn = torch.softmax(q @ k * self.scale, dim=-1)        # B, HW, P^2
        out = (attn @ v).permute(0, 2, 1).reshape(B, Cm, H, W)

        gate = self.sigmoid(self.out_proj(out))
        return nn.LeakyReLU(0.2, True)(gate * x + x)


class SwinAttnGLRA(nn.Module):
    """Variant 10: Local window self-attention (simplified Swin, no shift)."""
    def __init__(self, inplanes, window_size=8):
        super().__init__()
        midplanes = inplanes // 4
        self.window_size = window_size
        self.input_conv = nn.Conv2d(inplanes, midplanes, 1)
        self.qkv = nn.Linear(midplanes, 3 * midplanes)
        self.proj = nn.Linear(midplanes, midplanes)
        self.out_proj = nn.Conv2d(midplanes, inplanes, 1)
        self.sigmoid = nn.Sigmoid()
        self.scale = midplanes ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.input_conv(x)
        Cm, ws = feat.shape[1], self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            feat = F.pad(feat, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = feat.shape

        # Partition into windows: (B*nH*nW, ws^2, Cm)
        feat_wins = feat.permute(0, 2, 3, 1)  # B, Hp, Wp, Cm
        feat_wins = feat_wins.view(B, Hp // ws, ws, Wp // ws, ws, Cm)
        feat_wins = feat_wins.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, Cm)

        qkv = self.qkv(feat_wins).reshape(-1, ws * ws, 3, Cm)
        q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = self.proj(attn @ v)  # B*nH*nW, ws^2, Cm

        # Reverse windows back to feature map
        nH, nW = Hp // ws, Wp // ws
        out = out.view(B, nH, nW, ws, ws, Cm)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, Cm)
        out = out[:, :H, :W, :].contiguous().permute(0, 3, 1, 2)  # B, Cm, H, W

        gate = self.sigmoid(self.out_proj(out))
        return nn.LeakyReLU(0.2, True)(gate * x + x)


if __name__ == '__main__':
    import numpy as np
    # inputs = torch.Tensor(16, 256, 96, 96)
    # x = torch.from_numpy(np.array([[[[1.,2.,3.],
    #                                     [4.,5.,6.],
    #                                     [7.,8.,9.]]]])).to(torch.float32)
    # x = torch.from_numpy(np.array([[[[1.,2.,3.,4.],
    #                                     [5.,6.,7.,8.],
    #                                     [9.,10.,11.,12.]]]])).to(torch.float32)
    pool = DPA(256)
    summary(pool, (16, 256, 96, 96))
