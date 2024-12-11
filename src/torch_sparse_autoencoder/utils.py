import torch


def default_sparsity_loss(hidden: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(hidden))


class StopForwardHookException(Exception):
    pass
