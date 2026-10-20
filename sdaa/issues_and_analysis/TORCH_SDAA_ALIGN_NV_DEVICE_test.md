# `TORCH_SDAA_ALIGN_NV_DEVICE=a100` 对 RFantibody 的影响测试

> 目的：验证这个 Torch-SDAA 环境变量能否消除 SDAA 与 NVIDIA 之间的随机数差异，从而让 RFdiffusion / ProteinMPNN 更接近 H100 参考。
> 结论：**该变量确实会改变 SDAA 的设备随机数，但对 RFantibody 三个模型（RFdiffusion/ProteinMPNN/RF2）的输出没有任何影响（逐字节相同）。RFantibody 的跨设备残差是浮点/BLAS 引起的，不是随机数引起的。**

---

## 1. 变量是什么
`TORCH_SDAA_ALIGN_NV_DEVICE` 是 Torch-SDAA 的**随机数发生器（RNG）对齐开关**：让 SDAA 的 `torch.rand/randn`（Philox）输出与指定的 NVIDIA 设备一致。二进制（`torch_sdaa/lib/libtorch_sdaa.so`）中的证据：

```
TORCH_SDAA_ALIGN_NV_DEVICE
"only used with TORCH_SDAA_ALIGN_NV_DEVICE"
"Unrecognized nv device. Only support a100/v100/A100/V100"
"TecoGeneratorImpl can't used with TORCH_SDAA_ALIGN_NV_DEVICE"
```
- 仅支持 `a100` / `v100`（大小写均可）；
- 设置后 SDAA 默认的 `TecoGeneratorImpl` 被禁用。

## 2. 测试方法
在同一 SDAA 环境、相同确定性参数下，分别以「默认」与「`export TORCH_SDAA_ALIGN_NV_DEVICE=a100`」运行官方脚本，比较输出文件（逐字节）与相对 H100 参考的精度：
- ProteinMPNN：`test/proteinmpnn/scripts/ab_seq_design.sh`
- RFdiffusion：`test/rfdiffusion/scripts/antibody_pdbdesign.sh`
- RF2：`test/rf2/scripts/ab_prediction.sh`（本身无采样）

## 3. 结果

### 3.1 变量本身确实生效（设备 RNG 改变）
```python
torch.manual_seed(0); torch.randn(5, device='sdaa')
# 默认                 : [ 0.2925,  0.1000,  2.5470,  0.0083,  0.1290]
# ALIGN_NV_DEVICE=a100 : [-0.9247, -0.4253, -2.6438,  0.1452, -0.1209]
```
（在子进程/bash 中也同样生效。）

### 3.2 但对 RFantibody 无影响
| 模块 | 指标 | 默认 | 对齐 a100 | 是否相同 |
|------|------|------|-----------|----------|
| ProteinMPNN | `dldesign_0/1` 序列一致率 | 97.28% / 94.98% | 97.28% / 94.98% | **逐字节相同** |
| ProteinMPNN | `dldesign_0/1` 骨架坐标偏差 | 0.0 Å | 0.0 Å | — |
| RFdiffusion | `ab_des_0` Cα RMSD vs H100 | 5.7486 Å | 5.7486 Å | **逐字节相同** |
| RFdiffusion | `ab_des_1` Cα RMSD vs H100 | 1.0025 Å | 1.0025 Å | **逐字节相同** |
| RF2 | 预测结构 | — | — | 确定性推理，无采样，不受影响 |

- ProteinMPNN 输出与默认运行 **byte-identical**；
- RFdiffusion 输出与默认运行 **byte-identical**（`CA_RMSD(对齐输出, 默认输出)=0.000000`）。

## 4. 为什么没有影响（根因）
RFantibody 的随机性**不走被该变量对齐的设备 RNG 路径**：
- **ProteinMPNN**：采样噪声用 **CPU** 随机数 —— `randn_2 = torch.randn(chain_M.shape).to(device)`（`util_protein_mpnn.py`，无 `device=` 参数 → CPU RNG）；
- **RFdiffusion**：torrosion/噪声来自 **CPU** 随机数 —— `torch.rand(L,10,1)`（`diffusion.py:836`，无 `device=` 参数）；其 `torch.multinomial` 虽在设备上，但结果由模型 logits（浮点）主导。
- 因此改设备 RNG 对二者输出无影响。

**推论**：之前观察到的「SDAA RNG ≠ NVIDIA RNG」**并不是** RFantibody 跨设备残差的来源；残差由 **浮点/BLAS/kernel 实现差异**（SDAA 与 NVIDIA 差 ~1e-6，比 NVIDIA 内部 ~1e-7 大一个量级）+ **扩散采样对扰动的样本级敏感性** 造成（详见 `SDAA_ADAPTATION.md` 表 1c）。

> 对照：真正影响 RFantibody 精度的是平台 **算子行为 bug**（已修复的 `nn.Embedding` ≥3 维索引问题），而不是随机数。

## 5. 建议
- **无需**为 RFantibody 设置 `TORCH_SDAA_ALIGN_NV_DEVICE`（实测无收益）；
- 该变量适用于**结果确实由设备 `torch.randn/rand` 驱动**的模型（如纯扩散/采样的模型，其噪声在设备上生成）；
- 若要进一步逼近 NVIDIA 参考，应针对**浮点/算子差异**做精度对齐（例如固定更严格的算子精度、逐个排查敏感算子），而非 RNG。

---
*测试时间：2026-10-19；平台：Tecorigin SDAA（Torch-SDAA 3.3.0，PyTorch 2.12.0）；参考：`test/*/reference_outputs/H100_references`。*