# RFantibody 评测结果（仅官方指标）

- 平台：Tecorigin SDAA（Torch-SDAA 3.3.0，PyTorch 2.12.0）
- 参考：仓库自带 `H100_references`
- 指标来源：**全部取自官方测试口径**（`test/util/util_test_utils.py`），无自定义指标
- 原始数据：`linshi/pc_report/pc_metrics.json`

> 官方结构判定口径（`compare_pdb_structures`）：**逐原子坐标偏差 ≤ 阈值**（rfdiffusion/proteinmpnn/util 为 **0.1 Å**；rf2 为 **0.2 Å**）+ REMARK 一致 + 原子数一致 + 残基(序列)一致 + 元素一致。
> 官方 RF2 打口径（`compare_score_lines`）：每个 SCORE 指标**相对 1% 或绝对 0.1 满足其一**。

---

## 表 1：结构一致性（官方口径，SDAA 输出 vs H100 参考）

| 模块 | 输出 | 逐原子最大偏差 | 阈值 | REMARK | 原子数 | 残基(序列) | 元素 | 官方判定 |
|------|------|----------------|------|--------|--------|------------|------|----------|
| util | `antibody_HLT.pdb` | 0.0000 Å | 0.1 Å | ✅ | ✅ 1602/1602 | 100% | 100% | ✅ PASS |
| util | `nanobody_HLT.pdb` | 0.0000 Å | 0.1 Å | ✅ | ✅ 946/946 | 100% | 100% | ✅ PASS |
| quiver | 7 项 CLI 测试 | — | — | — | — | — | — | ✅ 官方 pytest 7/7 |
| rfdiffusion | `nb_des_0.pdb` | 0.0168 Å | 0.1 Å | ✅ | ✅ 1456/1456 | 100% | 100% | ✅ PASS |
| rfdiffusion | `nb_des_1.pdb` | 0.0476 Å | 0.1 Å | ✅ | ✅ 1472/1472 | 100% | 100% | ✅ PASS |
| rfdiffusion | `ab_des_1.pdb` | 2.7547 Å | 0.1 Å | ✅ | ✅ 1912/1912 | 100% | 100% | ❌ FAIL(坐标) |
| rfdiffusion | `ab_des_0.pdb` | 10.7649 Å | 0.1 Å | ✅ | ✅ 1912/1912 | 100% | 100% | ❌ FAIL(坐标) |
| rfdiffusion | `ab_designs.qv` / `nb_designs.qv` | — | — | ✅ 标签 2/2 | — | — | — | ✅ PASS |
| proteinmpnn | `ab_des_0_dldesign_0.pdb` | 0.0000 Å | 0.1 Å | ✅ | ✅ 1912/1912 | 97.28% | 100% | ❌ FAIL(序列) |
| proteinmpnn | `ab_des_0_dldesign_1.pdb` | 0.0000 Å | 0.1 Å | ✅ | ✅ 1912/1912 | 94.98% | 100% | ❌ FAIL(序列) |
| rf2 | `ab_proteinmpnn_output_best.pdb` | 9.0427 Å | 0.2 Å | ✅ | ✅ 7186/7186 | 100% | 100% | ❌ FAIL(坐标) |

---

## 表 2：RF2 SCORE 指标（官方口径，`compare_score_lines`，相对 1% 或绝对 0.1）

| 指标 | H100 参考 | SDAA | 差值 | 官方判定 |
|------|-----------|------|------|----------|
| `pred_lddt` | 0.89 | 0.90 | 0.01 | ✅ |
| `pae` | 10.64 | 10.67 | 0.03 | ✅ |
| `interaction_pae` | 19.38 | 19.47 | 0.09 | ✅ |
| `target_aligned_antibody_rmsd` | 30.82 | 30.76 | 0.06 | ✅ |
| `framework_aligned_antibody_rmsd` | 1.25 | 1.17 | 0.08 | ✅ |
| `framework_aligned_H1_rmsd` | 0.90 | 0.85 | 0.05 | ✅ |
| `framework_aligned_L2_rmsd` | 0.98 | 0.89 | 0.09 | ✅ |
| `framework_aligned_H2_rmsd` | 0.98 | 0.86 | 0.12 | ❌ |
| `framework_aligned_L3_rmsd` | 1.23 | 1.11 | 0.12 | ❌ |
| `framework_aligned_cdr_rmsd` | 2.25 | 2.10 | 0.15 | ❌ |
| `target_aligned_cdr_rmsd` | 24.64 | 24.50 | 0.14 | ❌ |
| `framework_aligned_L1_rmsd` | 2.51 | 2.30 | 0.21 | ❌ |
| `framework_aligned_H3_rmsd` | 3.75 | 3.54 | 0.21 | ❌ |

---

## 表 3：运行时间与显存（**自采埋点，非官方评估指标**）

| 模块 | 任务 | 输入规模 | 推理耗时 | 峰值显存 |
|------|------|----------|----------|----------|
| ProteinMPNN | 序列设计（2 seq/结构） | 1 结构（~240 残基） | 18.7 s | 330 MB |
| RF2 | 结构预测（1 recycle） | 478 残基 | 259.4 s | 4234 MB |
| RFdiffusion | 抗体设计（1 design，48 步） | 478 残基 | 271.0 / 269.9 s | 3121 / 3161 MB |
| RFdiffusion | 纳米抗体设计（1 design，48 步） | 364 / 368 残基 | 151.3 / 127.1 s | 2270 / 2298 MB |

> 数据来源：`linshi/logs/inference_metrics.jsonl`（自采，非官方指标）。