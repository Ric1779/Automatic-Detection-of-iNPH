from torch import nn
import torch.nn.functional as F

class ConvNormActi(nn.Module):
    """
    The basic block for convolutions
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, last_layer=False) -> None:
        super(ConvNormActi,self).__init__()
        self.last_layer = last_layer
        self.conv = nn.Conv3d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, padding=(kernel_size//2))
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.last_layer:
            return self.conv(input)
        return self.relu(self.bn(self.conv(input)))
    

class HighResBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernels, channel_matching='pad', dilation=1) -> None:
        super(HighResBlock, self).__init__()
        self.project, self.pad = None, None
        if in_channels != out_channels:
            if channel_matching not in ('pad', 'project'):
                raise ValueError('channel matching must be pad or project, got {}.'.format(channel_matching))
            if channel_matching == 'project':
                self.project = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if channel_matching == 'pad':
                if in_channels > out_channels:
                    raise ValueError('in_channels > out_channels is incompatible with `channel_matching=pad`.')
                pad_1 = (out_channels - in_channels) // 2
                pad_2 = out_channels - in_channels - pad_1
                pad = [0, 0] * 3 + [pad_1, pad_2] + [0, 0]
                self.pad = lambda input: F.pad(input,pad)
        # Layer 1
        self.bn1 = nn.BatchNorm3d(num_features=in_channels)
        pad_size1 = dilation * (kernels[0]//2)
        pad1 = [pad_size1] * 6 
        self.pad1 = lambda input: F.pad(input, pad1)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernels[0], dilation=dilation)

        # Layer 2
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        pad_size2 = dilation * (kernels[1]//2)
        pad2 = [pad_size2] * 6 
        self.pad2 = lambda input: F.pad(input, pad2)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernels[1], dilation=dilation)

        self.relu = nn.ReLU()

    def forward(self, input):

        x = self.conv1(self.pad1(self.relu(self.bn1(input))))
        x = self.conv2(self.pad2(self.relu(self.bn2(x))))
        if self.project is not None:
            return x + self.project(input)
        if self.pad is not None:
            return x + self.pad(input)
        return x + input