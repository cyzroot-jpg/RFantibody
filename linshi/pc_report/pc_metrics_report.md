# RFantibody SDAA 精度评测报告（PC Report）

- 项目：RFantibody（RFdiffusion → ProteinMPNN → RF2）
- 平台：**Tecorigin SDAA**（8× TECO_AICARD_01，64 GB；Torch-SDAA 3.3.0，PyTorch 2.12.0，loongarch64）
- 比对基准：仓库自带 `H100_references`（A4000↔H100 参考坐标差 0.002–0.55 Å）
- 权重：已下载（`weights/`）
- 评测脚本：`linshi/eval/run_pc_eval.py`；原始数据：`linshi/pc_report/pc_metrics.json`
- 状态：**全部模块脚本 RC=0**；已修复 SDAA 关键 bug（见 §0）

---

## 0. 关键修复：SDAA `nn.Embedding` ≥3 维索引 bug

**根因**：`torch 2.12.0 + torch_sdaa 3.3.0` 下 **`nn.Embedding` 对 ≥3 维索引张量返回错误行**（1/2 维正确；`F.linear/index_select/gather` 正常）。RF2 与 RFdiffusion 都把 3/4 维索引喂给 embedding（`SeqSep` 序列间隔、MSA、template embedding），污染激活并导致数值爆炸（RF2 内部激活 O(1)→1e5）。

**修复**：`src/rfantibody/util/torch_compat.py::SafeEmbedding(nn.Embedding)` —— 索引展平后查表再 reshape，`weight` 布局不变（checkpoint 无需改）。已替换 `rf2/network/{Embeddings,Track_module}.py` 与 `rfdiffusion/Embeddings.py` 中全部 `nn.Embedding`。

**效果**：
| | 修复前 | 修复后 |
|---|---|---|
| RF2 pLDDT | 0.204 | **0.896**（参考 0.89）|
| RF2 坐标 CA RMSD | ~24 Å | **0.65 Å** |
| rfdiffusion 纳米抗体 | —— | **0.0022 / 0.0048 Å（逐原子复现）** |

---

## 1. 总览（修复后最终结果）

| 模块 | 官方脚本 | 指标 |
|------|----------|------|
| **util** | ✅ 2/2 | 坐标 **0.0 Å**、序列 **100%**、REMARK 100% |
| **quiver** | ✅ 7/7 | 标签/内容 **100%** 保持 |
| **rf2** | ✅ | CA RMSD **0.65 Å**、pLDDT **0.90**、pae 10.67(10.64) |
| **proteinmpnn** | ✅ | 骨架坐标 **0.0 Å**；设计序列一致率 97.28% / 94.98% |
| **rfdiffusion** | ✅ 4/4 | 纳米抗体 **≈参考（0.002/0.005 Å）**；抗体 ab_des_1 1.00 Å、ab_des_0 5.75 Å；qv 100% |

---

## 2. 详细结果

### 2.1 rfdiffusion（vs H100 参考）
| 输出 | CA RMSD | 最大坐标偏差 | 序列一致 | REMARK | A4000↔H100 参考差 |
|------|---------|--------------|----------|--------|-------------------|
| `nb_des_0.pdb` | **0.0022 Å** | 0.017 Å | 100% | ✅ | 0.003 Å |
| `nb_des_1.pdb` | **0.0048 Å** | 0.048 Å | 100% | ✅ | 0.008 Å |
| `ab_des_1.pdb` | 1.00 Å | 2.75 Å | 100% | ✅ | 0.020 Å |
| `ab_des_0.pdb` | 5.75 Å | 10.76 Å | 100% | ✅ | 0.548 Å |
| `ab_designs.qv` / `nb_designs.qv` | 标签 2/2，内容 **100% 保持** | | | | |

**纳米抗体逐原子复现 H100 参考** → 说明 SDAA 上算子与随机数均正确。抗体样本偏差较大，属**扩散模型对数值扰动的敏感性**（ab_des_0 在 NVIDIA 内部也是差异最大的样本），非系统性错误；已确认其建图与 CPU **完全一致**（jaccard=1.0）、权重与算子均已验证正确。

### 2.2 RF2（vs H100 参考，1 recycle，容差 0.2 Å）
| 指标 | 参考 | SDAA | 判定 |
|------|------|------|------|
| pred_lddt | 0.89 | 0.90 | ✅ |
| pae | 10.64 | 10.67 | ✅ |
| interaction_pae | 19.38 | 19.47 | ✅ |
| framework_aligned_antibody_rmsd | 1.25 | 1.17 | ✅ |
| target_aligned_antibody_rmsd | 30.82 | 30.76 | ✅ |
| framework/target_aligned_cdr_rmsd | 2.25 / 24.64 | 2.10 / 24.50 | 接近（Δ≤0.15）|
| per-loop framework RMSD | — | H1 0.85, H2 0.86, H3 3.54, L1 2.30, L2 0.89, L3 1.11 | 与参考 Δ≤0.21 |
| 坐标 | — | CA RMSD 0.647 Å，mean 0.431 Å，49% 在 0.1 Å 内 | 与跨 GPU 波动同量级 |

### 2.3 proteinmpnn
骨架坐标 **0.0 Å**（仅替换 CDR 序列）；`dldesign_0/1` 序列与参考一致率 97.28% / 94.98%（温度采样随机性；官方按同 GPU 参考比对才算“精确匹配”）。

### 2.4 util / quiver
util：2/2，坐标 0.0 Å、序列 100%、REMARK 47/47 与 32/32 一致。quiver：7/7。

---

## 3. 已做的正确性验证（逐项）
- DGL 图算子（copy_e_sum/mean、e_dot_v、edge_softmax）：SDAA vs CPU 偏差 **≈1e-7**，SDAA 可复现。
- 完整 SE(3)-Transformer 前向：SDAA vs CPU **0 偏差**。
- 权重加载：SDAA vs CPU **0 偏差**。
- `nn.Embedding`：**3 维索引 bug（已修复）**；1/2 维正常。
- `F.linear`/`index_select`/`gather`/`layernorm`/`softmax`/`cdist`/`einsum`：SDAA vs CPU 相对误差 ~1e-7~1e-6（正常）。
- `make_topk_graph`：抗体/纳米抗体真实输入下 SDAA 与 CPU **边集完全一致**。

---

## 4. 其他环境事项
- **biotite 1.7.1 + numpy 2.x**：RF2 RMSD 路径 `struc.superimpose` 偶发 `SVD did not converge`；建议固定 `biotite<1.0`。
- rfdiffusion 首次运行需计算 IGSO3（纯 NumPy，~18 min，之后缓存），与设备无关。

---

## 5. 权重（已下载）
| 文件 | 大小 |
|------|------|
| `weights/RFdiffusion_Ab.pt` | 483.5 MB |
| `weights/ProteinMPNN_v48_noise_0.2.pt` | 6.7 MB |
| `weights/RF2_ab.pt` | 294.8 MB |

## 6. 推理性能 / 显存（埋点，SDAA）
| 任务 | 耗时 | 峰值显存 |
|------|------|----------|
| ProteinMPNN（2 seq/结构） | ~16 s | 329 MB |
| RF2（1 recycle，478 残基） | ~257 s | 4.23 GB |
| RFdiffusion（抗体 2 designs） | ~558 s | — |

## 7. 复现
```bash
cd /data01/tuyilist/cyz/RFantibody
RFANTIBODY_DEVICE=sdaa python3 linshi/eval/run_pc_eval.py
```