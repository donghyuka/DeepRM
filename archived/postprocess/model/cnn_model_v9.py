import torch
from torch import nn, Tensor

def _get_activation_fn(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim = -1)
    else:
        raise ValueError(f"Activation function {activation} not supported.")

class MaskedSequential(nn.Sequential):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        assert len(mask.shape) == len(input.shape)

        for module in self._modules.values():
            ## check if module is a ResidualBlock
            if isinstance(module, (ResidualBlock1D, ResidualBlock2D)):
                input = module(input, mask)
            else:
                input = module(input)
                ## check if input has maskable dimension
                if len(input.shape) == len(mask.shape):
                    input = input * mask
                else:
                    pass

        return input


class ResidualBlock1D(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int = 1, padding = 'same',
                 dropout: float = 0.1,activation: str = "silu", initrange: float = 0.1):
        super(ResidualBlock1D, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        self.init_weights(initrange)

    def init_weights(self, initrange: float):
        self.conv1.weight.data.uniform_(-initrange, initrange)
        self.conv2.weight.data.uniform_(-initrange, initrange)
        if len(self.shortcut) > 0:
            self.shortcut[0].weight.data.uniform_(-initrange, initrange)
        return None

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out) * pad_mask
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out) * pad_mask
        out = self.dropout(out)
        out += self.shortcut(x)
        out = out * pad_mask
        return out


class ResidualBlock2D(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int = 1, padding = 'same',
                 dropout: float = 0.1,activation: str = "silu"):
        super(ResidualBlock2D, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out) * pad_mask
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out) * pad_mask
        out = self.dropout(out)
        out += self.shortcut(x)
        out = out * pad_mask
        return out


class CNNModel(torch.nn.Module):
    def __init__(self, input_height, input_width, hidden_dim, output_dim, num_err_layers, num_meta_layers,
                 num_pred_layers, num_output_layers, dropout_rate, kernel_size):
        super(CNNModel, self).__init__()
        self.activation = _get_activation_fn("silu")

        ## Error Module
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=(1,input_width), stride=(1,input_width), padding="valid"))
        layers.append(self.activation)
        self.error_module = MaskedSequential(*layers)

        ## Metadata Module
        layers = []
        layers.append(torch.nn.Conv1d(in_channels=7, out_channels=hidden_dim, kernel_size=1, stride=1, padding="same"))
        layers.append(self.activation)
        self.metadata_module = MaskedSequential(*layers)

        ## Prediction Module
        layers = []
        layers.append(torch.nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=1, stride=1, padding="same"))
        for i in range(num_pred_layers):
            layers.append(ResidualBlock1D(kernel_size=kernel_size, in_channels=hidden_dim, out_channels=hidden_dim,
                                          stride=1, padding="same", activation="silu", dropout=dropout_rate))
        layers.append(self.activation)
        self.pred_module = MaskedSequential(*layers)

        ## Output Module

        layers = []
        layers.append(self.activation)
        layers.append(torch.nn.Conv1d(in_channels=hidden_dim*3, out_channels=hidden_dim, kernel_size=1, stride=1, padding="same"))
        for i in range(num_output_layers):
            layers.append(ResidualBlock1D(kernel_size=kernel_size, in_channels=hidden_dim, out_channels=hidden_dim,
                                          stride=1, padding="same", activation="silu", dropout=dropout_rate))

        layers.append(self.activation)
        layers.append(torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 4, kernel_size=kernel_size, stride=1, padding="same"))
        layers.append(self.activation)
        layers.append(torch.nn.Flatten(start_dim=1))
        layers.append(torch.nn.Linear((hidden_dim // 4) *input_height, hidden_dim))
        layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim // 4))
        layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim // 4, hidden_dim // 16))
        layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_dim // 16, output_dim))
        self.output_module = MaskedSequential(*layers)

        self.depth_module = torch.nn.Sequential(torch.nn.Linear(1, 1))
        self.final_module = torch.nn.Sequential(torch.nn.Linear(output_dim, output_dim))

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        return None

    def forward(self, error, metadata, pred, mask):
        error = error.permute(0, 3, 1, 2).contiguous()
        metadata = torch.permute(metadata, (0, 2, 1)).contiguous()
        pred = torch.permute(pred, (0, 2, 1)).contiguous()
        mask = mask.permute(0, 2, 1).contiguous()
        error = self.error_module(error, mask.unsqueeze(3)).squeeze(3)
        metadata = self.metadata_module(metadata, mask)
        pred = self.pred_module(pred, mask)
        x = torch.cat([error, metadata, pred], dim=1)
        x = self.output_module(x, mask)
        depth = mask.shape[2] / torch.sum(mask, dim = 2, keepdim = False)
        depth = self.depth_module(depth)
        x = x * depth
        x = self.final_module(x)
        x = torch.clamp(x, 0, 1)
        return x



