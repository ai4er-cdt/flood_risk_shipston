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
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = self.pool(x)
        x = self.fc1(x.contiguous().view(x.size(0), -1))
        x = self.fc2(x)
        return x


class Conv1DModel(nn.Module):
    """PyTorch `Module` class for an LSTM model."""

    def __init__(self, num_features: int, dropout_rate: float = 0.0) -> None:
        """
        Initialise model.

        Args:
            hidden_units (int): Number of hidden units/LSTM cells per layer.
            num_features (int): Number of features to input to the LSTM.
            dropout_rate (float, optional): Dropout rate of the last fully
            connected layer. Defaults to 0.0.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        """
        super(Conv1DModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_features = num_features

        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc1 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Tensor of shape `(batch_size, seq_length,
            num_features)` containing the input data for the network.

        Returns:
            torch.Tensor: Tensor containing the network predictions
        """
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool(x)
        x = self.fc1(x)
        return x
