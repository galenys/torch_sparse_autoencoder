# Torch Sparse Autoencoder

A Python library for implementing sparse autoencoders using PyTorch.

## Installation

```bash
pip install torch-sparse-autoencoder
```

## Usage

```python
from torch_sparse_autoencoder import SparseAutoencoderManager

# Create a sparse autoencoder
manager = SparseAutoencoderManager(
    model=model,
    layer=target_layer,
    activation_dim=activation_dim,
    sparse_dim=activation_dim*4,
    device=device
)

# Train the autoencoder
manager.train(
    torch_dataset,
    num_epochs=5,
    batch_size=4,
    sparsity_weight=1e-3,
    verbose=True
)

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
