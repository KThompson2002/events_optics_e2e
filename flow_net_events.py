"""
FlowNetEvents — Standard ANN optical flow network for event-camera inputs.

Architecture mirrors FlowNetS_spike's decoder exactly, but replaces the SNN
encoder (conv_s + IF neurons with surrogate gradients) with a standard
conv+BatchNorm+LeakyReLU encoder.  This gives clean gradient flow to all
encoder layers and trains reliably from scratch.

Input:  [B, 4, H, W]  — event representation (former ON/OFF, latter ON/OFF),
        produced by summing Spike-FlowNet's [B, 4, H, W, T_bins] over T_bins.

Output (training): (flow1, flow2, flow3, flow4)
        flow1 : [B, 2, H,   W]    full resolution
        flow2 : [B, 2, H/2, W/2]
        flow3 : [B, 2, H/4, W/4]
        flow4 : [B, 2, H/8, W/8]

Output (eval):    flow1 only.
"""
import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_


# ---------------------------------------------------------------------------
# Building blocks  (same signatures as Spike-FlowNet's util.py)
# ---------------------------------------------------------------------------
def _conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )


def _predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=False)


def _deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )


def _crop_like(x, target):
    if x.size()[2:] == target.size()[2:]:
        return x
    return x[:, :, :target.size(2), :target.size(3)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FlowNetEvents(nn.Module):
    """
    ANN event-flow network.  Drop-in replacement for FlowNetS_spike that
    accepts a 4-channel event tensor and produces the same multi-scale flow
    outputs.
    """

    def __init__(self, batchNorm=True, in_channels=4):
        super().__init__()
        self.batchNorm = batchNorm

        # ---- Encoder  (conv + BN + LeakyReLU, stride-2 downsampling) ----
        self.conv1 = _conv(batchNorm, in_channels, 64,  kernel_size=7, stride=2)
        self.conv2 = _conv(batchNorm, 64,  128, kernel_size=5, stride=2)
        self.conv3 = _conv(batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv4 = _conv(batchNorm, 256, 512, kernel_size=3, stride=2)

        # ---- Bottleneck residual convolutions (same as Spike-FlowNet) ----
        self.conv_r11 = _conv(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = _conv(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = _conv(batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = _conv(batchNorm, 512, 512, kernel_size=3, stride=1)

        # ---- Decoder (identical channel sizes to Spike-FlowNet) ----
        self.deconv3 = _deconv(batchNorm, 512,     128)
        self.deconv2 = _deconv(batchNorm, 256+128+2, 64)   # concat3: 256+128+2
        self.deconv1 = _deconv(batchNorm, 128+64+2,  32)   # concat2: 128+64+2

        # Transposed-conv upsampling before predict_flow (matches Spike-FlowNet dims)
        self.up4_to_3 = nn.ConvTranspose2d(512,      32, kernel_size=4, stride=2, padding=1, bias=False)
        self.up3_to_2 = nn.ConvTranspose2d(256+128+2, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2_to_1 = nn.ConvTranspose2d(128+64+2,  32, kernel_size=4, stride=2, padding=1, bias=False)
        self.up1_to_0 = nn.ConvTranspose2d(64+32+2,   32, kernel_size=4, stride=2, padding=1, bias=False)

        self.predict_flow4 = _predict_flow(32)
        self.predict_flow3 = _predict_flow(32)
        self.predict_flow2 = _predict_flow(32)
        self.predict_flow1 = _predict_flow(32)

        # ---- Weight init ----
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        """
        x : [B, 4, H, W]  (events summed over temporal bins)
        """
        # Encoder
        c1 = self.conv1(x)     # [B,  64, H/2,  W/2 ]
        c2 = self.conv2(c1)    # [B, 128, H/4,  W/4 ]
        c3 = self.conv3(c2)    # [B, 256, H/8,  W/8 ]
        c4 = self.conv4(c3)    # [B, 512, H/16, W/16]

        # Bottleneck residual
        r11 = self.conv_r11(c4)
        r12 = self.conv_r12(r11) + c4
        r21 = self.conv_r21(r12)
        r22 = self.conv_r22(r21) + r12

        # Scale-4  (coarsest: H/16)
        flow4 = self.predict_flow4(self.up4_to_3(r22))
        flow4_up = _crop_like(flow4, c3)
        d3 = _crop_like(self.deconv3(r22), c3)

        # Scale-3  (H/8)
        concat3 = torch.cat([c3, d3, flow4_up], dim=1)        # 256+128+2 = 386
        flow3 = self.predict_flow3(self.up3_to_2(concat3))
        flow3_up = _crop_like(flow3, c2)
        d2 = _crop_like(self.deconv2(concat3), c2)

        # Scale-2  (H/4)
        concat2 = torch.cat([c2, d2, flow3_up], dim=1)        # 128+64+2  = 194
        flow2 = self.predict_flow2(self.up2_to_1(concat2))
        flow2_up = _crop_like(flow2, c1)
        d1 = _crop_like(self.deconv1(concat2), c1)

        # Scale-1  (H/2)
        concat1 = torch.cat([c1, d1, flow2_up], dim=1)        # 64+32+2   = 98
        flow1 = self.predict_flow1(self.up1_to_0(concat1))    # [B, 2, H, W]

        if self.training:
            return flow1, flow2, flow3, flow4
        return flow1

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if 'weight' in n]

    def bias_parameters(self):
        return [p for n, p in self.named_parameters() if 'bias' in n]
