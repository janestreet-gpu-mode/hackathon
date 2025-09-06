import torch
import torch.nn as nn


def get_model_device(model):
    return next(iter(model.parameters())).device


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int, proj_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, proj_size)
        self.fc2 = nn.Linear(proj_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CausalConv1d(nn.Module):
    """
    Causal convolution layer: the hidden state contains the last kernel_size - 1 events
    (zeros initially) so we can have compute the convolution on a full window when
    receiving a new event.
    """

    def __init__(self, hidden_size, kernel_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        return torch.zeros(
            batch_size, self.hidden_size, self.kernel_size - 1, device=device
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        x_with_state = torch.concat([state, x[:, :, None]], dim=-1)
        out = self.conv(x_with_state)
        new_state = x_with_state[:, :, 1:]
        return out.squeeze(-1), new_state


class BlockLinear(nn.Module):
    """
    Linear layers that has multiple blocks with different weights. This could
    also be implemented with a regular linear layer having diagonal-by-block weights.
    """

    def __init__(self, num_blocks: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(num_blocks, self.block_size, self.block_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.hidden_size))
        else:
            self.bias = None

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == self.hidden_size
        x = x.view(batch_size, self.num_blocks, self.block_size)
        out = torch.einsum("bnh,nkh->bnk", x, self.weight)
        out = out.reshape(batch_size, self.hidden_size)
        if self.bias is not None:
            out += self.bias
        return out
