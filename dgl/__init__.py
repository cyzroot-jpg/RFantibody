"""
Pure-PyTorch drop-in replacement for the small subset of the Deep Graph Library
(DGL) API that RFantibody depends on.

This package exists so the project can run on devices/platforms where DGL has no
build (e.g. Tecorigin SDAA on loongarch64). It implements only what the pipeline
actually uses:

    - ``dgl.graph((src, dst), num_nodes=N)``
    - ``DGLGraph.edges()`` / ``DGLGraph.num_nodes()`` / ``DGLGraph.to(device)``
    - ``DGLGraph.edata`` (plain dict-like storage)
    - ``dgl.ops.{copy_e_sum, copy_e_mean, copy_e_max, e_dot_v, u_dot_v,
      copy_u, copy_v, edge_softmax}``
    - ``dgl.nn.pytorch.{AvgPooling, MaxPooling}``

All operators are implemented with native PyTorch scatter/index ops and are
autograd-compatible, so they work on CPU, CUDA and SDAA without modification.
"""

import torch

from . import nn
from . import ops


class _EdgeData(dict):
    """Minimal dict-like edge feature store matching the DGL ``edata`` surface."""

    def to(self, device):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device)
        return self


class DGLGraph:
    """A homogeneous directed graph holding COO edges and edge features."""

    def __init__(self, src, dst, num_nodes):
        self._src = src.to(torch.long)
        self._dst = dst.to(torch.long)
        self._num_nodes = int(num_nodes)
        self.edata = _EdgeData()

    def to(self, device):
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self.edata.to(device)
        return self

    @property
    def device(self):
        return self._src.device

    def edges(self):
        """Return the (src, dst) endpoint tensors of every edge."""
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def number_of_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return self._src.shape[0]

    def number_of_edges(self):
        return self._src.shape[0]

    def __repr__(self):
        return (
            f"DGLGraph(num_nodes={self._num_nodes}, "
            f"num_edges={self.num_edges()}, device={self.device})"
        )


def graph(data, num_nodes=None):
    """Create a :class:`DGLGraph` from a ``(src, dst)`` tuple."""
    src, dst = data
    src = src.to(torch.long)
    dst = dst.to(torch.long)
    if num_nodes is None:
        num_nodes = int(torch.max(torch.stack([src.max(), dst.max()]))) + 1
    return DGLGraph(src, dst, num_nodes)


__all__ = ["DGLGraph", "graph", "ops", "nn"]