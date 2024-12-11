import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.functional import F
from typing import Any, Type, Callable
from .utils import *
from .sae import *


class SparseAutoencoderManager:
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        activation_dim,
        sparse_dim: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.layer = layer
        self.sparse_dim = sparse_dim
        self.device = device

        self.sparse_autoencoder: SparseAutoencoder = SparseAutoencoder(
            activation_dim, sparse_dim
        ).to(self.device)
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.sparse_autoencoder.parameters()
        )

    def train(
        self,
        data: Dataset,
        num_epochs: int = 10,
        batch_size: int = 32,
        recon_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        sparsity_loss: Callable[[torch.Tensor], torch.Tensor] = default_sparsity_loss,
        sparsity_weight: float = 1e-3,
        verbose: bool = True,
        learning_rate: float = 1e-3,
        stop_in_hook: bool = True,
    ) -> None:
        """
        Args:
            data: input data of shape (batch_size, input_dim)
            num_epochs: number of epochs to train the model
            batch_size: batch size to use for the DataLoader
            recon_loss: reconstruction loss function
            sparsity_loss: sparsity loss function
            sparsity_weight: weight to apply to the sparsity loss
            verbose: whether to display a progress bar
            learning_rate: learning rate for the optimizer
            stop_in_hook: whether the model should run a full forward pass
            or just run until it first generates the activations for the
            target layer
        """
        self.model.eval()
        self.layer.eval()
        self.optimizer.param_groups[0]["lr"] = learning_rate

        dataloader = DataLoader(data, batch_size=batch_size)

        def loss_fn(
            original: torch.Tensor,
            hidden: torch.Tensor,
            reconstruction: torch.Tensor,
        ) -> torch.Tensor:
            return sparsity_weight * sparsity_loss(hidden) + recon_loss(
                original, reconstruction
            )

        def hook(_module, _input, output):
            self.optimizer.zero_grad()
            hidden, reconstruction = self.sparse_autoencoder(output)
            loss = loss_fn(output, hidden, reconstruction)
            loss.backward()
            self.optimizer.step()
            del hidden, reconstruction, loss
            if stop_in_hook:
                raise StopForwardHookException

        handle = self.layer.register_forward_hook(hook)
        for i in range(num_epochs):
            for batch in tqdm.tqdm(
                dataloader, disable=not verbose, desc=f"Epoch {i+1}/{num_epochs}"
            ):
                try:
                    self.model(batch[0].to(self.device))
                except StopForwardHookException:
                    pass
        handle.remove()

    def get_encoder_features(self) -> torch.Tensor:
        """
        Returns:
            encoder features of shape (hidden_dim, input_dim)
        """
        return self.sparse_autoencoder.encoder.weight.detach()

    def get_decoder_features(self) -> torch.Tensor:
        """
        Returns:
            decoder features of shape (hidden_dim, input_dim)
        """
        return self.sparse_autoencoder.decoder.weight.T.detach()
