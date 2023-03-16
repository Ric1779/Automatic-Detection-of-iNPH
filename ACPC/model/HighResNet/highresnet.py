from .modules import ConvNormActi, HighResBlock
from torch import nn
from model.HighResNet.config import HighResNetParams

class HighResNet(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super(HighResNet,self).__init__()
        layer_params = HighResNetParams.DEFAULT_LAYER_PARAMS_3D
        blocks = nn.ModuleList()
        _in_chns, _out_chns = in_channels, layer_params[0]['n_features']
        blocks.append(ConvNormActi(_in_chns, _out_chns))
        for i,params in enumerate(layer_params[1:-2]):
            _in_chns, _out_chns = _out_chns, params['n_features']
            _dilation = 2**i
            for _ in range(params['repeat']):
                blocks.append(HighResBlock(_in_chns, _out_chns, params['kernels'], dilation=_dilation))
                _in_chns = _out_chns

        blocks.append(ConvNormActi(layer_params[3]['n_features'], layer_params[4]['n_features'], kernel_size=1))
        blocks.append(ConvNormActi(layer_params[4]['n_features'], num_classes, kernel_size=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self,input):
        return self.blocks(input)