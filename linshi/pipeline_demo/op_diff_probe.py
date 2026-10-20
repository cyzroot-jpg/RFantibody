#!/usr/bin/env python3
"""
算子浮点差异探针：在指定设备上跑一组标准算子，与 CPU 结果比较，报告相对差。
用于对比 SDAA 与 NVIDIA(CUDA) 的"算子实现差异"量级。

用法:
    python op_diff_probe.py            # 自动选设备 (sdaa/cuda/cpu)
    python op_diff_probe.py cuda
    python op_diff_probe.py sdaa
    python op_diff_probe.py cpu

在 SDAA 机器和 A100 机器上各跑一次，比较输出即可。
"""
import sys
import torch
import torch.nn as nn


def backend_name():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if hasattr(torch, "sdaa") and torch.sdaa.is_available():
        return "sdaa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(dev):
    torch.manual_seed(0)
    results = {}

    def record(name, fn):
        y_cpu = fn("cpu")
        if dev == "cpu":
            return
        y_dev = fn(dev)
        d = (y_cpu - y_dev.cpu()).abs().max().item()
        scale = y_cpu.abs().max().item() + 1e-12
        results[name] = (d, d / scale)

    x = torch.randn(512, 256)
    m = nn.Linear(256, 256)
    record("Linear",     lambda d: m.to(d)(x.to(d)))
    record("matmul",     lambda d: x.to(d) @ x.to(d).T)
    record("einsum",     lambda d: torch.einsum("ik,jk->ij", x.to(d), x.to(d)))
    record("LayerNorm",  lambda d: nn.LayerNorm(256).to(d)(x.to(d)))
    record("softmax",    lambda d: torch.softmax(x.to(d), -1))
    record("sum(dim)",   lambda d: x.to(d).sum(-1))
    record("cdist",      lambda d: torch.cdist(x.to(d)[:64], x.to(d)[:64]))

    # 同机"不结合律"演示：换个求和顺序，结果就不同
    a, b, c = torch.randn(64, 64), torch.randn(64, 64), torch.randn(64, 64)
    assoc = ((a @ b) @ c - a @ (b @ c)).abs().max().item()

    dev = torch.device(backend_name())
    name = dev.type
    if name != "cpu":
        try:
            props = (torch.cuda if name == "cuda" else torch.sdaa).get_device_name(0)
        except Exception:
            props = name
    else:
        props = "CPU"

    print(f"# backend={name}  device={props}")
    print(f"{'op':12s} {'max_abs_diff':>14s} {'rel_diff':>12s}")
    for k, (d, r) in results.items():
        print(f"{k:12s} {d:14.3e} {r:12.3e}")
    print(f"{'(A@B)@C vs A@(B@C)':12s} {assoc:14.3e}   (同机不结合律)")


if __name__ == "__main__":
    run(backend_name())