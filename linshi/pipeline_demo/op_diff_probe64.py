#!/usr/bin/env python3
"""
算子浮点误差探针（v2，跨机器可比）。

关键：以 **float64（CPU）为共同真值**，衡量"设备 float32 结果"相对真值的舍入误差。
这样 x86 与 LoongArch 的 CPU 差异被消除（fp64 精度 ~1e-16，远小于 fp32 的差异），
得到的是**该设备 fp32 本身的舍入误差**，可在不同机器间直接对比。

用法:
    python op_diff_probe64.py            # 自动选设备
    python op_diff_probe64.py cuda       # A100
    python op_diff_probe64.py sdaa       # SDAA
"""
import sys
import torch
import torch.nn as nn


def pick_device():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if hasattr(torch, "sdaa") and torch.sdaa.is_available():
        return "sdaa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# 每个算子给出：(fp32 输入, fn(x))  —— fn 用 float64 在 CPU 上算真值，用 float32 在设备上算
def build_cases():
    torch.manual_seed(0)
    x = torch.randn(512, 256)
    m = nn.Linear(256, 256)
    cases = {
        "Linear":     (x, lambda t, d: m.to(d).to(t)(x.to(d).to(t))),
        "matmul":     (x, lambda t, d: x.to(d).to(t) @ x.to(d).to(t).T),
        "einsum":     (x, lambda t, d: torch.einsum("ik,jk->ij", x.to(d).to(t), x.to(d).to(t))),
        "LayerNorm":  (x, lambda t, d: nn.LayerNorm(256).to(d).to(t)(x.to(d).to(t))),
        "softmax":    (x, lambda t, d: torch.softmax(x.to(d).to(t), -1)),
        "sum(dim)":   (x, lambda t, d: x.to(d).to(t).sum(-1)),
        "cdist":      (x, lambda t, d: torch.cdist(x[:64].to(d).to(t), x[:64].to(d).to(t))),
    }
    # 固定 dtype：t=torch.float64 真值，t=torch.float32 设备值
    return cases


def main():
    dev = pick_device()
    if dev == "cpu":
        # 真值机器也允许跑，输出 np 说明
        pass
    cases = build_cases()
    rows = []
    for name, (_, fn) in cases.items():
        ref64 = fn(torch.float64, "cpu")           # 真值（CPU, float64）
        y32 = fn(torch.float32, dev).double().cpu()  # 设备 float32
        d = (y32 - ref64).abs().max().item()
        scale = ref64.abs().max().item() + 1e-12
        rows.append((name, d, d / scale))
    try:
        name = (torch.cuda if dev == "cuda" else torch.sdaa).get_device_name(0) if dev != "cpu" else "CPU"
    except Exception:
        name = dev
    print(f"# backend={dev}  device={name}  reference=float64(CPU)")
    print(f"{'op':12s} {'max_abs_err_vs_fp64':>22s} {'rel_err':>12s}")
    for k, d, r in rows:
        print(f"{k:12s} {d:22.3e} {r:12.3e}")


if __name__ == "__main__":
    main()