import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """PyTorch `Module` class for an LSTM model."""

    def __init__(self, hidden_units: int, num_features: int, dropout_rate: float = 0.0, num_layers: int = 1) -> None:
        """
        Initialise model.

        Args:
            hidden_units (int): Number of hidden units/LSTM cells per layer.
            num_features (int): Number of features to input to the LSTM.
            dropout_rate (float, optional): Dropout rate of the last fully
            connected layer. Defaults to 0.0.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        """
        super(LSTMModel, self).__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_features = num_features

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.hidden_units,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Tensor of shape `(batch_size, seq_length,
            num_features)` containing the input data for the LSTM network.

        Returns:
            torch.Tensor: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)

        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        return pred


class LSTNet(nn.Module):

    def __init__(self, num_features=3, dropout_rate=0.2):
        super(LSTNet, self).__init__()
        self.num_features = num_features
        self.conv1_out_channels = 32
        self.conv1_kernel_height = 7
        self.recc1_out_channels = 64
        self.skip_steps = [4, 24]
        self.skip_reccs_out_channels = [4, 4]
        self.output_out_features = 1
        self.ar_window_size = 7
        self.dropout = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv2d(1, self.conv1_out_channels,
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.skip_reccs = {}
        for i in range(len(self.skip_steps)):
            self.skip_reccs[i] = nn.GRU(self.conv1_out_channels, self.skip_reccs_out_channels[i], batch_first=True)
        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)

    def forward(self, X):
        """Forward pass.

        Args:
            X (tensor): [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)

        # Convolutional Layer
        C = X.unsqueeze(1)  # [batch_size, num_channels=1, time_steps, num_features]
        C = F.relu(self.conv1(C))  # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = self.dropout(C)
        C = torch.squeeze(C, 3)  # [batch_size, conv1_out_channels, shrinked_time_steps]

        # Recurrent Layer
        R = C.permute(0, 2, 1)  # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R)  # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :]  # [batch_size, recc_out_channels]
        R = self.dropout(R)

        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip_steps)):
            skip_step = self.skip_steps[i]
            skip_sequence_len = shrinked_time_steps // skip_step
            # shrinked_time_steps shrinked further
            # [batch_size, conv1_out_channels, shrinked_time_steps]
            S = C[:, :, - skip_sequence_len * skip_step:]
            # [batch_size, conv1_out_channels, skip_sequence_len, skip_step=num_skip_components]
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step)
            # note that num_skip_components = skip_step
            # [batch_size, skip_step=num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.permute(0, 3, 2, 1).contiguous()
            # [batch_size*num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.view(S.size(0) * S.size(1), S.size(2), S.size(3))
            # [batch_size*num_skip_components, skip_sequence_len, skip_reccs_out_channels[i]]
            out, hidden = self.skip_reccs[i](S)
            S = out[:, -1, :]  # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_step * S.size(1))
            S = self.dropout(S)
            R = torch.cat((R, S), 1)  # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]

        # Output Layer
        output = F.relu(self.output(R))  # [batch_size, output_out_features=1]

        if self.ar_window_size > 0:
            # set dim3 based on output_out_features
            AR = X[:, -self.ar_window_size:, 3:4]  # [batch_size, ar_window_size, output_out_features=1]
            AR = AR.permute(0, 2, 1).contiguous()  # [batch_size, output_out_features, ar_window_size]
            AR = self.ar(AR)  # [batch_size, output_out_features, 1]
            AR = AR.squeeze(2)  # [batch_size, output_out_features]
            output = output + AR

        return output
