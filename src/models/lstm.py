import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    """PyTorch `Module` class for an LSTM model."""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.0, num_layers: int = 1) -> None:
        """
        Initialise model.

        Args:
            hidden_size (int): Number of hidden units/LSTM cells
            dropout_rate (float, optional): Dropout rate of the last fully
            connected layer. Defaults to 0.0.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        """
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

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
