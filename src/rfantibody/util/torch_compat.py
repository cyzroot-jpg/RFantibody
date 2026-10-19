"""
PyTorch/SDAA compatibility shims.

`torch 2.12.0 + torch_sdaa 3.3.0` returns *incorrect rows* for `nn.Embedding`
lookups whose index tensor has >= 3 dimensions (1-D and 2-D lookups are
correct). Both RF2 and RFdiffusion feed 3-D (and 4-D) index tensors to
embeddings (sequence-separation, MSA, template embeddings), which corrupts the
activations and destabilises the whole network on SDAA.

`SafeEmbedding` keeps the exact same parameter layout (``weight``) as
``nn.Embedding`` so existing checkpoints load unchanged, but performs the lookup
on a flattened index and restores the shape — avoiding the broken kernel path on
any device.
"""

import torch.nn as nn


class SafeEmbedding(nn.Embedding):
    """Drop-in ``nn.Embedding`` that works with N-D index tensors on SDAA."""

    def forward(self, input):  # noqa: A002 - match nn.Embedding signature
        original_shape = input.shape
        out = super().forward(input.reshape(-1))
        return out.reshape(*original_shape, self.embedding_dim)