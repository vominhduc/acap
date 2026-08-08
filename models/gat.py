from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    self_loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    if edge_index.numel() == 0:
        return self_loops
    return torch.cat([edge_index, self_loops], dim=1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.W_o = nn.Linear(out_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        batch_size, num_nodes, _ = x.shape
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        edge_index = add_self_loops(edge_index, num_nodes)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0, :], edge_index[1, :]

            attn_logits = (Q[:, :, dst] * K[:, :, src]).sum(dim=-1) / (self.head_dim ** 0.5)

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = self.dropout(attn_weights)

            weighted_msgs = attn_weights.unsqueeze(-1) * V[:, :, src]
            index = dst.view(1, 1, -1, 1).expand(batch_size, self.num_heads, -1, self.head_dim)
            out = torch.zeros(batch_size, self.num_heads, num_nodes, self.head_dim, device=x.device)
            out.scatter_add_(2, index, weighted_msgs)

            out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
            out = self.W_o(out)

            x = self.layer_norm(residual + out)
        else:
            x = self.layer_norm(residual)

        return x


class ConceptGNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 1536,
        output_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_projection: bool = True,
    ):
        super().__init__()
        self.use_projection = use_projection

        if not use_projection:
            hidden_dim = output_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(
                GraphAttentionLayer(in_d, hidden_dim, num_heads, dropout)
            )

        if use_projection:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.mlp = nn.LayerNorm(output_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)

        x = node_embeddings
        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.mlp(x)
        x = x.squeeze(0) if x.size(0) == 1 else x
        return x
