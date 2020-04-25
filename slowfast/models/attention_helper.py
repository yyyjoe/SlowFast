import torch
from torch import nn
from torch.nn import functional as F

class Conv3D_Block(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(Conv3D_Block, self).__init__()
        self.conv3d = nn.Conv3d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv3d(x)))
        
class SpatialAttn(nn.Module):
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = Conv3D_Block(1, 1, 3, s=2, p=1)
        self.conv2 = Conv3D_Block(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # trilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2, x.size(4) * 2),
            mode='trilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        x = F.avg_pool3d(x, (x.size(2),1,1))
        return x

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = Conv3D_Block(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = Conv3D_Block(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)

        x = F.avg_pool3d(x, (1,x.size(3),x.size(4)))
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(1, keepdim=True)
        return x

class SoftAttn(nn.Module):

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels[1])


    def forward(self, x):
        x_s = torch.sigmoid(self.spatial_attn(x[0]))
        x_f = torch.sigmoid(self.channel_attn(x[1]))

        return [x_s,x_f]