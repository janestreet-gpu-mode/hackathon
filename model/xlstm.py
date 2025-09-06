import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BlockLinear, CausalConv1d, get_model_device

# NX-AI Reference
# https://github.com/NX-AI/xlstm/tree/main/xlstm/blocks

CausalConv1dState = torch.Tensor
MLSTMCellState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MLSTMBlockState = tuple[CausalConv1dState, MLSTMCellState]
SLSTMCellState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
SLSTMBlockState = tuple[CausalConv1dState, torch.Tensor, SLSTMCellState]
XLSTMState = tuple[SLSTMBlockState, MLSTMBlockState]


class MLSTMCell(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.eps = 1e-6

        self.igate_proj = nn.Linear(3 * hidden_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(3 * hidden_size, num_heads, bias=True)
        self.outnorm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_size)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: MLSTMCellState
    ) -> tuple[torch.Tensor, MLSTMCellState]:
        batch_size, hidden_size = q.shape

        cell_state, norm_state, max_state = state

        qkv_cat = torch.cat([q, k, v], dim=-1)
        igate_preact = self.igate_proj(qkv_cat)
        fgate_preact = self.fgate_proj(qkv_cat)

        q = q.view(batch_size, self.num_heads, self.head_size)
        k = k.view(batch_size, self.num_heads, self.head_size)
        v = v.view(batch_size, self.num_heads, self.head_size)

        # Trick for numerical stability
        log_f = torch.nn.functional.logsigmoid(fgate_preact)
        max_new = torch.maximum(igate_preact, max_state + log_f)

        i_gate = torch.exp(igate_preact - max_new)
        f_gate = torch.exp(log_f + max_state - max_new)

        # Scale keys
        k = k / math.sqrt(self.head_size)

        # Update memory and normalizer
        # C_new = f * C + i * k^T * v
        cell_new = (
            f_gate[:, :, None, None] * cell_state
            + i_gate[:, :, None, None] * k[:, :, :, None] * v[:, :, None]
        )
        # n_new = f * n + i * k
        norm_new = f_gate[:, :, None] * norm_state + i_gate[:, :, None] * k

        # Compute output: h = (q @ C) / max(q @ n, 1)
        numerator = torch.einsum("bnh,bnhk->bnk", q, cell_new)
        qn_dotproduct = torch.einsum("bnh,bnh->bn", q, norm_new)
        max_val = torch.exp(-max_new)
        denominator = torch.maximum(qn_dotproduct.abs(), max_val) + self.eps
        out = numerator / denominator[:, :, None]

        out = self.outnorm(out.view(batch_size, self.hidden_size))

        out = out.reshape(batch_size, self.hidden_size)

        assert cell_new.shape == cell_state.shape
        assert norm_new.shape == norm_state.shape
        assert max_new.shape == max_state.shape

        return out, (cell_new, norm_new, max_new)

    def init_state(self, batch_size: int, device: torch.device) -> MLSTMCellState:
        return (
            torch.zeros(
                batch_size,
                self.num_heads,
                self.head_size,
                self.head_size,
                device=device,
            ),
            torch.zeros(batch_size, self.num_heads, self.head_size, device=device),
            torch.zeros(batch_size, self.num_heads, device=device),
        )


class MLSTMBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        conv_kernel_size: int = 4,
        qkv_proj_block_size: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.inner_size = expand_factor * hidden_size

        self.norm = nn.LayerNorm(hidden_size, bias=False)

        self.x_proj = nn.Linear(hidden_size, self.inner_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.inner_size, bias=False)

        num_blocks = self.inner_size // qkv_proj_block_size
        self.q_proj = BlockLinear(num_blocks, self.inner_size, bias=False)
        self.k_proj = BlockLinear(num_blocks, self.inner_size, bias=False)
        self.v_proj = BlockLinear(num_blocks, self.inner_size, bias=False)

        self.conv1d = CausalConv1d(self.inner_size, kernel_size=conv_kernel_size)

        self.mlstm_cell = MLSTMCell(self.inner_size, num_heads)
        self.proj_down = nn.Linear(self.inner_size, hidden_size, bias=False)
        self.learnable_skip = nn.Parameter(torch.ones(self.inner_size))

        self.head_size = self.inner_size // num_heads

    def forward(
        self, x: torch.Tensor, state: MLSTMBlockState
    ) -> tuple[torch.Tensor, MLSTMBlockState]:
        conv_state, mlstm_state = state

        skip = x

        x = self.norm(x)
        x_mlstm = self.x_proj(x)
        x_gate = self.gate_proj(x)

        x_conv, new_conv_state = self.conv1d(x_mlstm, conv_state)
        x_mlstm_conv = F.silu(x_conv)

        q = self.q_proj(x_mlstm_conv)
        k = self.k_proj(x_mlstm_conv)
        v = self.v_proj(x_mlstm)

        mlstm_out, new_mlstm_state = self.mlstm_cell(q, k, v, mlstm_state)

        mlstm_out_skip = mlstm_out + (self.learnable_skip * x_mlstm_conv)
        h_state = mlstm_out_skip * F.silu(x_gate)
        y = self.proj_down(h_state)

        return y + skip, (new_conv_state, new_mlstm_state)

    def init_state(self, batch_size: int, device: torch.device) -> MLSTMBlockState:
        return (
            self.conv1d.init_state(batch_size, device),
            self.mlstm_cell.init_state(batch_size, device),
        )


class SLSTMCell(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.eps = 1e-6

    def forward(
        self,
        i: torch.Tensor,
        f: torch.Tensor,
        z: torch.Tensor,
        o: torch.Tensor,
        state: SLSTMCellState,
    ) -> tuple[torch.Tensor, SLSTMCellState]:
        cell_state, norm_state, max_state = state

        log_f_plus_m = max_state + torch.nn.functional.logsigmoid(f)

        # Use torch.where to avoid data-dependent branching
        max_new = torch.maximum(i, log_f_plus_m)

        # Compute stabilized exponential gates
        o_gate = torch.sigmoid(o)
        i_gate = torch.exp(i - max_new)
        f_gate = torch.exp(log_f_plus_m - max_new)

        cell_new = f_gate * cell_state + i_gate * torch.tanh(z)
        norm_new = f_gate * norm_state + i_gate
        y_new = o_gate * cell_new / (norm_new + self.eps)

        return y_new, (cell_new, norm_new, max_new)

    def init_state(self, batch_size: int, device: torch.device) -> SLSTMCellState:
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device) - float("inf"),
        )


class SLSTMBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, conv_kernel_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(hidden_size, bias=False)
        self.conv1d = CausalConv1d(hidden_size, kernel_size=conv_kernel_size)
        self.igate_input = BlockLinear(num_heads, hidden_size, bias=False)
        self.fgate_input = BlockLinear(num_heads, hidden_size, bias=False)
        self.zgate_input = BlockLinear(num_heads, hidden_size, bias=False)
        self.ogate_input = BlockLinear(num_heads, hidden_size, bias=False)

        self.igate_state = BlockLinear(num_heads, hidden_size)
        self.fgate_state = BlockLinear(num_heads, hidden_size)
        self.zgate_state = BlockLinear(num_heads, hidden_size)
        self.ogate_state = BlockLinear(num_heads, hidden_size)

        self.slstm_cell = SLSTMCell(hidden_size, num_heads)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: SLSTMBlockState,
    ) -> tuple[torch.Tensor, SLSTMBlockState]:
        conv_state, recurrent_state, slstm_state = state

        skip = x
        x = self.norm(x)

        x_conv, new_conv_state = self.conv1d(x, conv_state)
        x_conv_act = F.silu(x_conv)

        i = self.igate_input(x_conv_act) + self.igate_state(recurrent_state)
        f = self.fgate_input(x_conv_act) + self.fgate_state(recurrent_state)
        z = self.zgate_input(x) + self.zgate_state(recurrent_state)
        o = self.ogate_input(x) + self.ogate_state(recurrent_state)

        new_recurrent_state, new_slstm_state = self.slstm_cell(i, f, z, o, slstm_state)
        slstm_out = self.group_norm(new_recurrent_state)

        return slstm_out + skip, (new_conv_state, new_recurrent_state, new_slstm_state)

    def init_state(self, batch_size: int, device: torch.device) -> SLSTMBlockState:
        return (
            self.conv1d.init_state(batch_size, device),
            torch.zeros(batch_size, self.hidden_size, device=device),
            self.slstm_cell.init_state(batch_size, device),
        )


class XLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlstm_num_heads: int = 8,
        slstm_num_heads: int = 4,
    ):
        super().__init__()

        self.mlstm_num_heads = mlstm_num_heads
        self.slstm_num_heads = slstm_num_heads

        self.slstm = SLSTMBlock(hidden_size, self.slstm_num_heads)
        self.mlstm = MLSTMBlock(hidden_size, self.mlstm_num_heads)
        self.final_norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, state: XLSTMState
    ) -> tuple[torch.Tensor, XLSTMState]:
        slstm_state, mlstm_state = state
        x, new_slstm_state = self.slstm(x, slstm_state)
        x, new_mlstm_state = self.mlstm(x, mlstm_state)

        out = self.final_norm(x)
        new_state = (new_slstm_state, new_mlstm_state)
        return out, new_state

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> XLSTMState:
        if device is None:
            device = get_model_device(self)

        assert device is not None
        return (
            self.slstm.init_state(batch_size, device),
            self.mlstm.init_state(batch_size, device),
        )
