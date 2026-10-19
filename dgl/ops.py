"""
Pure-PyTorch implementations of the DGL message-passing operators used by the
SE(3)-Transformer / RFantibody graph blocks.

Semantics match ``dgl.ops`` (https://github.com/dmlc/dgl):

    copy_e_*    : reduce edge features into *destination* nodes
    e_dot_v     : per-edge dot product of an edge feature with the dst node feature
    u_dot_v     : per-node dot product of two node features
    copy_u/copy_v: broadcast a node feature onto edges (src / dst)
    edge_softmax: softmax over a node's incoming (or outgoing) edges
"""

import torch

__all__ = [
    "copy_e_sum",
    "copy_e_mean",
    "copy_e_max",
    "copy_e_min",
    "e_dot_v",
    "u_dot_v",
    "copy_u",
    "copy_v",
    "edge_softmax",
]


def _feat_shape(x):
    return (x.shape[0],) + tuple(x.shape[1:])


def copy_e_sum(graph, x):
    """Sum edge features into destination nodes: ``out[dst[e]] += x[e]``."""
    out = x.new_zeros((graph.num_nodes(),) + tuple(x.shape[1:]))
    return out.index_add(0, graph._dst, x)


def copy_e_mean(graph, x):
    """Average edge features per destination node."""
    num_nodes = graph.num_nodes()
    out = x.new_zeros((num_nodes,) + tuple(x.shape[1:]))
    out = out.index_add(0, graph._dst, x)
    ones = x.new_ones((x.shape[0],) + (1,) * (x.dim() - 1))
    count = x.new_zeros((num_nodes,) + (1,) * (x.dim() - 1)).index_add(0, graph._dst, ones)
    return out / count.clamp_min(1)


def copy_e_max(graph, x):
    """Max-reduce edge features per destination node."""
    num_nodes = graph.num_nodes()
    out = x.new_full((num_nodes,) + tuple(x.shape[1:]), torch.finfo(x.dtype).min)
    return out.index_reduce(0, graph._dst, x, "amax", include_self=True)


def copy_e_min(graph, x):
    """Min-reduce edge features per destination node."""
    num_nodes = graph.num_nodes()
    out = x.new_full((num_nodes,) + tuple(x.shape[1:]), torch.finfo(x.dtype).max)
    return out.index_reduce(0, graph._dst, x, "amin", include_self=True)


def e_dot_v(graph, x, y):
    """Per-edge dot product between edge feature ``x`` and dst node feature ``y``."""
    return (x * y[graph._dst]).sum(dim=-1, keepdim=True)


def u_dot_v(graph, x, y):
    """Per-node dot product between node features ``x`` and ``y``."""
    return (x * y).sum(dim=-1, keepdim=True)


def copy_u(graph, x):
    """Broadcast source node features onto edges: ``out[e] = x[src[e]]``."""
    return x[graph._src]


def copy_v(graph, x):
    """Broadcast destination node features onto edges: ``out[e] = x[dst[e]]``."""
    return x[graph._dst]


def edge_softmax(graph, logits, eids=None, norm_by="dst"):
    """Softmax over the incoming (``norm_by='dst'``) or outgoing edges per node."""
    index = graph._dst if norm_by == "dst" else graph._src
    num_nodes = graph.num_nodes()
    tail = tuple(logits.shape[1:])

    neg = torch.finfo(logits.dtype).min
    value_max = logits.new_full((num_nodes,) + tail, neg)
    value_max = value_max.index_reduce(0, index, logits.detach(), "amax", include_self=True)

    shifted = torch.exp(logits - value_max[index])
    value_sum = logits.new_zeros((num_nodes,) + tail).index_add(0, index, shifted)
    return shifted / value_sum[index].clamp_min(torch.finfo(logits.dtype).tiny)