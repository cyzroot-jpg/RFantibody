# RFantibody 评测框架说明（脚本 / 数据集 / 指标 / 权重）

> 本文档说明 RFantibody 在 SDAA 上做精度评测所使用的**官方评测脚本、数据集、指标（含义）与权重**，以及本次为 SDAA 适配所做的改动。
> 生成时间：见文件修改时间；项目根目录：`/data01/tuyilist/cyz/RFantibody`。

---

## 1. 评测脚本

RFantibody 仓库自带一套**官方评测（回归测试）框架**，位于 `test/`：

| 层级 | 文件 | 作用 |
|------|------|------|
| 入口 | `test/run_tests.py` | 统一入口：`uv run python -m test.run_tests [--module X]`，依次跑各模块 pytest，并可 `--create-refs` 重建参考输出 |
| 用例 | `test/<module>/test_<module>.py` | pytest 用例：用固定随机种子跑确定性 CLI，再与参考输出逐文件比对 |
| 执行 | `test/<module>/scripts/*.sh` | 确定性调用 CLI（`uv run rfdiffusion/proteinmpnn/rf2`）并输出到指定目录 |
| 比对 | `test/util/util_test_utils.py` | 精度/一致性比对函数（见第 3 节） |
| 夹具 | `test/<module>/conftest.py` | 输出目录、参考目录（按 GPU 类型选择 A4000/H100）等 |

各模块脚本与输出：

- **util**：`antibody_hlt_conversion.sh`、`nanobody_hlt_conversion.sh` → 调用 `scripts/util/chothia2HLT.py`，输出 `antibody_HLT.pdb` / `nanobody_HLT.pdb`
- **rfdiffusion**：`antibody_pdbdesign.sh`、`antibody_qvdesign.sh`、`nanobody_pdbdesign.sh`、`nanobody_qvdesign.sh` → `ab_des_{0,1}.pdb` / `nb_des_{0,1}.pdb` / `ab_designs.qv` / `nb_designs.qv`
- **proteinmpnn**：`ab_seq_design.sh` → `ab_des_0_dldesign_{0,1}.pdb`
- **rf2**：`ab_prediction.sh` → `ab_proteinmpnn_output_best.pdb`（另有 `json_prediction.sh` 用于 JSON/TCR 输入）
- **quiver**：`test/quiver/test_quiver.py`（纯 CLI 自校验，无需权重/GPU）

> 官方框架只给出 **pass/fail**。本项目在 `linshi/eval/run_pc_eval.py` 中**复用官方脚本**，额外计算**定量精度指标**（坐标偏差、序列一致率、SCORE 偏差等），结果见 `linshi/pc_report/pc_metrics_report.md`。

---

## 2. 数据集

RFantibody **不依赖外部下载数据集**，评测数据全部内置于仓库的 `test/<module>/inputs_for_test/`：

| 模块 | 数据集（输入） | 说明 |
|------|----------------|------|
| util | `6mh2_chothia.pdb`（抗体，链 D/C）、`3eak_chothia.pdb`（纳米抗体，链 B） | Chothia 编号的 Fab/纳米抗体结构 |
| rfdiffusion | `rsv_site3.pdb`（靶标）、`hu-4D5-8_Fv.pdb`（scFv 框架）、`h-NbBCII10.pdb`（纳米抗体框架） | 抗体/纳米抗体骨架设计输入 |
| proteinmpnn | `ab_des_0.pdb` | 上一步产出的抗体骨架，用于 CDR 序列设计 |
| rf2 | `ab_proteinmpnn_output.pdb`、`rfab_targets.json` | 待预测的抗体-靶标复合物（PDB 或 JSON） |
| quiver | `inputs_for_test/` 下约 100 个 PDB + `designs_scored.qv/.sc`、`tinyinterface.pdb` | Quiver 读写/切片/重命名等功能自校验数据 |

**结论**：无需额外下载数据集；直接使用仓库内置的 `test/*/inputs_for_test`。参考输出（ground truth）位于 `test/<module>/reference_outputs/{A4000,H100}_references/`。

---

## 3. 指标及含义

评测指标来自 `test/util/util_test_utils.py`：

### 3.1 结构一致性（`compare_pdb_structures`）
| 指标 | 含义 | 官方容差 |
|------|------|----------|
| REMARK 行一致 | `REMARK PDBinfo-LABEL:` CDR 标注是否完全一致 | 必须完全一致 |
| 原子数一致 | 原子总数是否相同 | 必须相同 |
| 残基名一致 | 每个原子的残基三字母名是否相同（序列一致性） | 必须相同 |
| 元素类型一致 | 每个原子的元素是否相同 | 必须相同 |
| 坐标偏差 | 对应原子欧氏距离 `‖ref−out‖` | ≤ 0.1 Å（蛋白 MPNN/rfdiffusion）、≤ 0.2 Å（RF2） |

> 备注：RF2 用例统一用 `coord_threshold=0.2`；rfdiffusion/proteinmpnn/util 用默认 0.1。

### 3.2 RF2 置信度 / 结构评分（`compare_score_lines`）
比对 PDB 中的 `SCORE <name>: <value>` 行，容差：**相对 1% 或绝对 0.1 二者满足其一即通过**。

| 指标 | 含义 | 方向 |
|------|------|------|
| `pred_lddt` | 预测 pLDDT（置信度，0–1） | 越大越好 |
| `pae` | 整体预测对齐误差 PAE（Å），模型对残基间相对位置的不确定度 | 越小越好 |
| `interaction_pae` | 跨链（抗体-靶标界面）平均 PAE | 越小越好 |
| `target_aligned_antibody_rmsd` | 以靶标对齐后，整条抗体 Cα RMSD | 越小越好 |
| `target_aligned_cdr_rmsd` | 以靶标对齐后，CDR Cα RMSD | 越小越好 |
| `framework_aligned_antibody_rmsd` | 以框架对齐后，整条抗体 Cα RMSD | 越小越好 |
| `framework_aligned_cdr_rmsd` | 以框架对齐后，CDR Cα RMSD | 越小越好 |
| `framework_aligned_{H1,H2,H3,L1,L2,L3}_rmsd` | 以框架对齐后，各 CDR 环 Cα RMSD | 越小越好 |

来源：`src/rfantibody/rf2/modules/model_runner.py::get_rmsds / get_confidence_scores`（RF2 预测结构 vs 输入结构的 Cα RMSD）。

### 3.3 Quiver 一致性（`test_quiver.py`）
自校验：`qvfrompdbs→qvextract` 往返后 PDB 内容逐行一致、`qvls` 标签数正确、`qvextract specific`/`qvslice`/`qvsplit`/`qvrename`/`qvscorefile` 结果一致。

---

## 4. 权重

权重下载脚本：`include/download_weights.sh`（写入 `<项目>/weights/`）。

| 权重文件 | 用途 | 下载地址 | 状态 |
|----------|------|----------|------|
| `RFdiffusion_Ab.pt` | RFdiffusion 抗体骨架设计 | https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt | ✅ **已下载**（483.5 MB） |
| `ProteinMPNN_v48_noise_0.2.pt` | ProteinMPNN 序列设计 | https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt | ✅ **已下载**（6.7 MB） |
| `RF2_ab.pt` | RF2 结构预测/打分 | https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt | ✅ **已下载**（294.8 MB） |
| `RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt` | RF2 TCR-MHC（JSON 输入） | https://zenodo.org/records/17488258/files/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1 | 可选（Zenodo 可达但慢，未下载） |

**结论**：运行 `bash include/download_weights.sh` 后，`files.ipd.uw.edu` 本次可达，**三个主权重已全部下载到 `<项目>/weights/` 并验证可 `torch.load`**。
下载日志：`linshi/logs/weight_download.log`（早前不可达时的记录见 `linshi/logs/weight_download_attempt.log`）。

---

## 5. 为评测新增的能力（不改变原有功能）

1. **性能埋点**（`src/rfantibody/util/inference_metrics.py`）：在 rfdiffusion / proteinmpnn / rf2 三个推理脚本中记录**单次推理耗时、峰值显存/已用显存、总显存、设备后端**，追加写入 `linshi/logs/inference_metrics.jsonl`（JSON Lines）。通过环境变量控制：
   - `RFANTIBODY_PROFILE=0` 关闭；
   - `RFANTIBODY_METRICS_FILE=<path>` 指定输出文件。
   - 埋点仅“时间/显存记录”，不改变模型输入输出与命令行行为。
2. **PC 评测 runner**（`linshi/eval/run_pc_eval.py`）：复用官方脚本并输出定量指标到 `linshi/pc_report/`。
3. **`uv` 轻量 shim**（`linshi/eval/bin/uv`）：让官方脚本无需安装 uv 即可运行。

---

## 6. SDAA 适配改动（与评测相关）

- 纯 PyTorch 重写 DGL（`dgl/` 包），移除 DGL 依赖；`torch.cuda.nvtx` 替换为 no-op。
- 新增设备抽象 `src/rfantibody/util/device_utils.py`（sdaa / cuda / cpu 自适应），并接入推理路径。
- **修复 SDAA `nn.Embedding` ≥3 维索引 bug**：新增 `src/rfantibody/util/torch_compat.py::SafeEmbedding`，并在 `rf2/network/{Embeddings,Track_module}.py`、`rfdiffusion/Embeddings.py` 中替换。这是 RF2/RFdiffusion 在 SDAA 上数值发散的主因。
- 修复 `scripts/util/chothia2HLT.py` 的换行 bug（首行 REMARK 与末行 ATOM 粘连），使 util 评测通过。
- `pyproject.toml`：移除 `dgl` 依赖、放宽 `requires-python>=3.10`、加入 `dgl` shim 打包，支持在 Python 3.12 上可编辑安装。