import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SparseAutoencoder, self).__init__()
        self.encoder: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.decoder: nn.Linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
        Returns:
            hidden: hidden representation of the input tensor of shape (batch_size, hidden_dim)
            output: output tensor of shape (batch_size, input_dim)
        """
        hidden = torch.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return hidden, reconstruction
