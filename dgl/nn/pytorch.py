"""
Pure-PyTorch replacements for ``dgl.nn.pytorch`` graph pooling modules.

RFantibody builds single, un-batched graphs, so pooling reduces over all nodes
and returns a leading batch dimension of size 1 to match DGL's output shape.
"""

import torch
import torch.nn as nn

__all__ = ["AvgPooling", "MaxPooling"]


class AvgPooling(nn.Module):
    """Average all node features into a single ``(1, D)`` vector."""

    def forward(self, graph, feat):
        return feat.mean(dim=0, keepdim=True)


class MaxPooling(nn.Module):
    """Max-reduce all node features into a single ``(1, D)`` vector."""

    def forward(self, graph, feat):
        return feat.max(dim=0, keepdim=True).values