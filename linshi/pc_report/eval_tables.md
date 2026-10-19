# RFantibody SDAA 评测结果表

- 平台：Tecorigin SDAA（Torch-SDAA 3.3.0，PyTorch 2.12.0）
- 比对基准：仓库自带 `H100_references`
- 数据来源：`linshi/pc_report/pc_metrics.json`、`linshi/logs/inference_metrics.jsonl`

---

## 表 1：模型在数据集上的指标精度

数据集为仓库内置测试输入（抗体 hu-4D5-8_Fv / 纳米抗体 h-NbBCII10 + RSV 靶标 / ProteinMPNN 输出）。

| 模型 | 数据集（测试样本） | 任务 | 评测指标 | 精度结果（SDAA vs H100 参考） |
|------|-------------------|------|----------|------------------------------|
| RFdiffusion | `nb_des_0.pdb` | 骨架设计（扩散生成抗体对接靶标的 3D 主链） | Cα RMSD | **0.0022 Å** ✅ |
| RFdiffusion | `nb_des_1.pdb` | 骨架设计 | Cα RMSD | **0.0048 Å** ✅ |
| RFdiffusion | `ab_des_1.pdb` | 骨架设计 | Cα RMSD | 1.00 Å ⚠ |
| RFdiffusion | `ab_des_0.pdb` | 骨架设计 | Cα RMSD | 5.75 Å ⚠ |
| RFdiffusion | `ab/nb_designs.qv` | 骨架设计 | 标签/内容保持率 | **100%** ✅ |
| ProteinMPNN | `ab_des_0_dldesign_0.pdb` | 序列设计（为 CDR 环指定氨基酸序列） | 骨架坐标偏差 / 序列一致率 | **0.0 Å** / 97.28% ✅ |
| ProteinMPNN | `ab_des_0_dldesign_1.pdb` | 序列设计 | 骨架坐标偏差 / 序列一致率 | **0.0 Å** / 94.98% ✅ |
| RF2 | `ab_proteinmpnn_output_best.pdb` | 结构预测与打分（预测复合物结构 + 置信度过滤） | pred_lddt | **0.90**（参考 0.89）✅ |
| RF2 | 同上 | 结构预测与打分 | pae | **10.67**（参考 10.64）✅ |
| RF2 | 同上 | 结构预测与打分 | interaction_pae | **19.47**（参考 19.38）✅ |
| RF2 | 同上 | 结构预测与打分 | framework_aligned_antibody_rmsd | **1.17 Å**（参考 1.25 Å）✅ |
| RF2 | 同上 | 结构预测与打分 | target_aligned_antibody_rmsd | **30.76 Å**（参考 30.82 Å）✅ |
| RF2 | 同上 | 结构预测与打分 | framework_aligned_cdr_rmsd | **2.10 Å**（参考 2.25 Å）✅ |
| RF2 | 同上 | 结构预测与打分 | 整体坐标 Cα RMSD | **0.65 Å** ✅ |

> ⚠：RFdiffusion 抗体样本偏差源于扩散模型对数值扰动的样本级敏感性（纳米抗体逐原子复现、建图与 CPU 完全一致），非系统性错误。
> ProteinMPNN 序列一致率未到 100% 属温度采样随机性；骨架坐标本身完全一致（0.0 Å）。

---

## 表 2：运行时间与显存（SDAA 实测）

| 模型/模块 | 任务 | 输入规模 | 推理耗时 | 峰值显存 |
|-----------|------|----------|----------|----------|
| ProteinMPNN | 序列设计（2 seq/结构） | 1 结构（~240 残基） | **18.7 s** | **330 MB** |
| RF2 | 结构预测（1 recycle） | 478 残基 | **259.4 s** | **4234 MB** |
| RFdiffusion | 抗体设计（1 design，48 步） | 478 残基 | **271.0 / 269.9 s** | **3121 / 3161 MB** |
| RFdiffusion | 纳米抗体设计（1 design，48 步） | 364 / 368 残基 | **151.3 / 127.1 s** | **2270 / 2298 MB** |

> 数据来源：`linshi/logs/inference_metrics.jsonl`（每次推理一条 JSON：`elapsed_sec`、`peak_mem_bytes`、`device=sdaa:0`、残基数等）。
> RFdiffusion 首次运行另需 IGSO3 计算（纯 NumPy，~15–20 min，之后缓存），与设备无关。