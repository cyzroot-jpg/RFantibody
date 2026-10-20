#!/usr/bin/env python3
"""
算子输出指纹：同 seed → 同输入 → 打印每个算子 fp32 输出的若干点位（完整精度）。
在 SDAA 与 A100 各跑一次，逐算子 diff：
  - 若指纹不同 → 直接证明"同输入下 SDAA 与 NVIDIA 的输出不同（舍入模式不同）"。

用法: python op_fingerprint.py [cuda|sdaa|cpu]
"""
import sys
import torch
import torch.nn as nn


def pick():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if hasattr(torch, "sdaa") and torch.sdaa.is_available():
        return "sdaa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


dev = pick()
torch.manual_seed(0)
x_cpu = torch.randn(512, 256)          # CPU 生成 → 各平台同输入
m = nn.Linear(256, 256)                # 同权重（seed 决定）
ln = nn.LayerNorm(256)

def fingerprint(y):
    f = y.detach().float().flatten()
    idx = [0, 1, 2, len(f) // 3, len(f) // 2, len(f) - 1]
    return [f"{float(f[i]):.12e}" for i in idx]

cases = {
    "Linear":   lambda: m.to(dev)(x_cpu.to(dev)),
    "matmul":   lambda: x_cpu.to(dev) @ x_cpu.to(dev).T,
    "einsum":   lambda: torch.einsum("ik,jk->ij", x_cpu.to(dev), x_cpu.to(dev)),
    "LayerNorm":lambda: ln.to(dev)(x_cpu.to(dev)),
    "softmax":  lambda: torch.softmax(x_cpu.to(dev), -1),
    "sum(dim)": lambda: x_cpu.to(dev).sum(-1),
    "cdist":    lambda: torch.cdist(x_cpu[:64].to(dev), x_cpu[:64].to(dev)),
}

print(f"# backend={dev}")
for name, fn in cases.items():
    y = fn()
    print(f"{name:10s} " + " ".join(fingerprint(y)))