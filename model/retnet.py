import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import get_model_device


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_rot


class RetNet(nn.Module):
    decay: torch.Tensor
    angle: torch.Tensor

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.scaling = self.head_size**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = nn.RMSNorm(self.head_size, eps=1e-6, elementwise_affine=False)

        self.register_buffer("decay", torch.empty(num_heads))
        self.register_buffer("angle", torch.empty(self.head_size))

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        seq_offsets, scales, recurrent_state = state
        assert seq_offsets.shape == (batch_size,)
        assert scales.shape == (batch_size, self.num_heads)
        assert recurrent_state.shape == (
            batch_size,
            self.num_heads,
            self.head_size,
            self.head_size,
        )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k = k * self.scaling

        q_heads = q.view(batch_size, self.num_heads, self.head_size)
        k_heads = k.view(batch_size, self.num_heads, self.head_size)
        v_heads = v.view(batch_size, self.num_heads, self.head_size)

        # Rope
        sin = torch.sin(seq_offsets[:, None, None] * self.angle[None, None, :])
        cos = torch.cos(seq_offsets[:, None, None] * self.angle[None, None, :])

        q_rope = q_heads * cos + rotate_every_two(q_heads) * sin
        k_rope = k_heads * cos + rotate_every_two(k_heads) * sin

        # State update
        kv_outer_prod = k_rope[:, :, :, None] * v_heads[:, :, None, :]
        new_recurrent_state = (
            recurrent_state * self.decay[None, :, None, None] + kv_outer_prod
        )

        # State scaling
        new_scales = scales * self.decay + 1.0
        scale_factor = (1.0 / new_scales.sqrt())[:, :, None, None]
        scaled_state = new_recurrent_state * scale_factor

        # Out
        out = torch.einsum("bnh,bnhk->bnk", q_rope, scaled_state)
        out = self.norm(out).reshape(batch_size, self.hidden_size)
        out = F.silu(g) * out
        out = self.out_proj(out)
        return out, (seq_offsets + 1, new_scales, new_recurrent_state)

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = get_model_device(self)
        return (
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            torch.zeros(batch_size, self.num_heads, device=device),
            torch.zeros(
                batch_size,
                self.num_heads,
                self.head_size,
                self.head_size,
                device=device,
            ),
        )
