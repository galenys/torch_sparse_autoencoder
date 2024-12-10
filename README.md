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
    model,
    layer=model.model.layers[0].self_attn.o_proj.weight.shape,
    sparse_dim=4000,
    device=device
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
