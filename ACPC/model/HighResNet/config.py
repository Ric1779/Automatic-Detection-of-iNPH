from dataclasses import dataclass

@dataclass
class HighResNetParams:
    DEFAULT_LAYER_PARAMS_3D = [
    # initial conv layer
    {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},

    # residual blocks
    {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
    {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
    {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
    
    # final conv layers
    {'name': 'conv_1', 'n_features': 80, 'kernel_size': 1},
    {'name': 'conv_2', 'kernel_size': 1},
    ]