import torch
import torch.nn as nn


class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNet(nn.Module):
    """See WaveNet paper here: https://arxiv.org/abs/1609.03499.

    Code adapted from https://www.kaggle.com/cswwp347724/wavenet-pytorch.
    """
    def __init__(self, num_features=2, kernel_size=3):
        super().__init__()
        self.wave_block1 = Wave_Block(num_features, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc1 = nn.Linear(11648, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Tensor of shape `(batch_size, seq_length,
            num_features)` containing the input data for the network.

        Returns:
            torch.Tensor: Tensor containing the network predictions
        """
        # Permute necessary to switch to (batch_size, num_features, seq_length).
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = self.pool(x)
        x = self.fc1(x.contiguous().view(x.size(0), -1))
        x = self.fc2(x)
        return x


def conv_layer(window, in_channels=1, kernel_size=3, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, 1, kernel_size=kernel_size, bias=False, dilation=dilation),
        nn.AdaptiveAvgPool1d(window),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class FilterNet(nn.Module):
    """Thanks to https://github.com/Mikata-Project/FilterNet for the code."""
    def __init__(self, num_features, window=24):
        super().__init__()
        self.conv1a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=1, dilation=1)
        self.conv2a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=2, dilation=1)
        self.conv3a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=3, dilation=1)
        self.conv4a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=4, dilation=1)
        self.conv5a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=5, dilation=1)
        self.conv6a = conv_layer(window=window // 2, in_channels=num_features, kernel_size=6, dilation=1)
        self.conv1b = conv_layer(window=window // 4, kernel_size=1, dilation=2)
        self.conv2b = conv_layer(window=window // 4, kernel_size=2, dilation=2)
        self.conv3b = conv_layer(window=window // 4, kernel_size=3, dilation=2)
        self.conv4b = conv_layer(window=window // 4, kernel_size=4, dilation=2)
        self.conv5b = conv_layer(window=window // 4, kernel_size=5, dilation=2)
        self.conv6b = conv_layer(window=window // 4, kernel_size=6, dilation=2)

        self.fc = nn.Linear(108, 1, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        self.f1a = self.conv1a(x)
        self.f2a = self.conv2a(x)
        self.f3a = self.conv3a(x)
        self.f4a = self.conv4a(x)
        self.f5a = self.conv5a(x)
        self.f6a = self.conv6a(x)
        self.f1b = self.conv1b(self.f1a)
        self.f2b = self.conv2b(self.f2a)
        self.f3b = self.conv3b(self.f3a)
        self.f4b = self.conv4b(self.f4a)
        self.f5b = self.conv5b(self.f5a)
        self.f6b = self.conv6b(self.f6a)
        x = torch.cat([self.f1a, self.f1b, self.f2a, self.f2b,
                       self.f3a, self.f3b, self.f4a, self.f4b,
                       self.f5a, self.f5b, self.f6a, self.f6b], 2)

        x = self.fc(x.contiguous().view(x.size(0), -1))
        return x


class Conv1DModel(nn.Module):
    """Basic 1D convolutional time series prediction model."""
    def __init__(self, num_features: int, dropout_rate: float = 0.0) -> None:
        super(Conv1DModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_features = num_features

        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(in_features=11584, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool(x)
        x = self.fc1(x.contiguous().view(x.size(0), -1))
        x = self.fc2(x)
        return x
