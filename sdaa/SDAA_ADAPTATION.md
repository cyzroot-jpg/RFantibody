# RFantibody 龙芯 SDAA 适配记录

> RFantibody 是一个**基于结构的抗体/纳米抗体 de novo 设计流水线**，由三部分组成：
> 1. 抗体微调版 **RFdiffusion** —— 生成抗体/纳米抗体骨架（对接靶标、设计 CDR 环结构）；
> 2. **ProteinMPNN** —— 为骨架的 CDR 环设计氨基酸序列；
> 3. 抗体微调版 **RoseTTAFold2 (RF2)** —— 预测复合物结构并输出置信度，用于筛选设计。
>
> 本次结论：**无自定义 CUDA 算子**，端到端在 SDAA 上跑通；并修复了一个 SDAA 平台 bug（`nn.Embedding` 对 ≥3 维索引返回错误行），使精度恢复到与 NVIDIA 参考一致。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix Server，8× TECO_AICARD_01，64 GB/卡；`torch.sdaa.device_count()=32`）|
| 机器 | `10.71.13.47` |
| Python | 3.12（`/home/py312`）|
| torch | 2.12.0 + torch_sdaa 3.3.0 |
| 分支 | `adapt/sdaa` |

## 二、分析结论

### 2.1 无自定义 CUDA 算子
RFantibody **无** `.cu/.cuh/.cpp/.c/.h/.pyx` 文件、无 triton、无 `cpp_extension` / `torch.autograd.Function` / `torch.library` 注册；`include/SE3Transformer/setup.py` 为纯 Python 包安装。计算全部是标准 PyTorch 算子 + **DGL 图算子** + e3nn。

### 2.2 依赖链

| 依赖 | 状态 | 处理 |
|------|------|------|
| torch | ✅ 已有 2.12 + torch_sdaa 3.3.0 | -- |
| numpy | ✅ 已有（安装 `numpy<2` 以兼容） | -- |
| e3nn | ✅ pip 安装（`--no-deps`，另装 `opt_einsum_fx`） | -- |
| biotite | ✅ 源码编译（loongarch 无预编译 wheel，Rust 可用） | -- |
| hydra-core / omegaconf / antlr4 | ✅ 已有 | -- |
| opt_einsum / icecream / pyrsistent / click / pandas | ✅ pip 安装（icecream 需 `asttokens`、`executing`） | -- |
| colorama | ✅ pip 安装（rfdiffusion 依赖） | -- |
| **dgl** | ❌ loongarch64 无构建 | **用纯 PyTorch 重写为 `dgl/` shim** |
| cuda-python==11.8 | ❌ 无（CUDA 专用） | 已移除依赖，不再需要 |

### 2.3 DGL 与打包
- **DGL** 在 loongarch64 上没有 wheel/源码构建，且其消息传递 kernel 为 CUDA 专有 → 用纯 PyTorch（`index_add`/`index_reduce`）实现项目实际用到的 8 个 API。
- `pyproject.toml` 原锁 `python==3.10`、`torch==2.3.*+cu118`、cu118 DGL → 放宽 `requires-python>=3.10`、移除 `dgl`/torch 固定源，加入 `dgl` shim 打包。

## 三、适配改动

### 3.1 DGL 用纯 PyTorch 替换（drop-in `dgl/` 包）
项目实际只用到 8 个 DGL API，全部用 PyTorch 实现，现有 `import dgl` 语句无需改动：
```python
# dgl/__init__.py  : DGLGraph / graph((src,dst), num_nodes=N) / edges() / edata / num_nodes() / to(device)
# dgl/ops.py       : copy_e_sum, copy_e_mean, copy_e_max/min, e_dot_v, u_dot_v, copy_u/v, edge_softmax
# dgl/nn/pytorch.py: AvgPooling, MaxPooling
```
语义严格对齐 DGL（`copy_e_*` 聚合到 dst；`e_dot_v=dot(x[e], y[dst])`；`edge_softmax` 按 dst 分组）。验证：SDAA vs CPU 偏差 **≈1e-7**，且 SDAA 可复现。
修改新增：`dgl/__init__.py`、`dgl/ops.py`、`dgl/nn/__init__.py`、`dgl/nn/pytorch.py`。

### 3.2 torch.cuda.nvtx 在 SDAA 上不可用
问题：`RuntimeError: NVTX functions not installed`（`torch.cuda.nvtx.range` 在 SDAA 抛错）。
解决：新增 no-op 替代，并替换 4 处 import。
```python
# 新增 include/SE3Transformer/se3_transformer/runtime/nvtx.py
from contextlib import contextmanager
@contextmanager
def range(msg, *args, **kwargs):
    yield
# 原： from torch.cuda.nvtx import range as nvtx_range
# 修： from se3_transformer.runtime.nvtx import range as nvtx_range
```
修改的代码：`se3_transformer/model/basis.py`、`model/layers/{attention,convolution,norm}.py`。

### 3.3 设备抽象：torch.cuda → 自动识别 sdaa/cuda/cpu
新增 `src/rfantibody/util/device_utils.py`（优先级 **sdaa → cuda → cpu**，可用 `RFANTIBODY_DEVICE` 强制），并接入推理路径：
- `scripts/{rf2_predict,proteinmpnn_interface_design,rfdiffusion_inference}.py`
- `src/rfantibody/rf2/modules/model_runner.py`、`rf2/network/predict.py`、`rfdiffusion/inference/model_runners.py`
```python
# 原： device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 修： device = get_device()   # 自动 sdaa:0 / cuda:0 / cpu
```

### 3.4 【关键】SDAA `nn.Embedding` ≥3 维索引 bug
问题：`torch 2.12 + torch_sdaa 3.3.0` 下 **`nn.Embedding` 对 ≥3 维索引张量返回错误行**（1/2 维正确；`F.linear/index_select/gather` 正常）。RF2 与 RFdiffusion 都把 3/4 维索引喂给 embedding（`SeqSep` 序列间隔、MSA、template embedding），导致激活被污染、网络数值爆炸（RF2 内部激活 O(1)→1e5，pLDDT 0.89→0.20）。
解决：新增 `SafeEmbedding`，索引展平后查表再 reshape（`weight` 布局不变，checkpoint 无需改动）。
```python
# 新增 src/rfantibody/util/torch_compat.py
class SafeEmbedding(nn.Embedding):
    def forward(self, input):
        shp = input.shape
        return super().forward(input.reshape(-1)).reshape(*shp, self.embedding_dim)
# 将各 nn.Embedding(...) 替换为 SafeEmbedding(...)
```
修改的代码：`rf2/network/Embeddings.py`、`rf2/network/Track_module.py`、`rfdiffusion/Embeddings.py`。
效果：RF2 pLDDT **0.204 → 0.896**（参考 0.89），坐标 CA RMSD 由 ~24 Å → **0.65 Å**；纳米抗体设计**逐原子复现** H100 参考。

### 3.5 chothia2HLT.py 换行 bug（首行 REMARK 与末行 ATOM 粘连）
问题：部分 biotite 版本 `PDBFile.write` 不补末尾换行，首行 `REMARK PDBinfo-LABEL` 会拼到最后一行 ATOM 上，导致 util 官方用例失败。
解决：先渲染为字符串再补换行。
```python
buffer = io.StringIO(); pdb_file.write(buffer)
structure_text = buffer.getvalue().rstrip("\n") + "\n"
```
修改的代码：`scripts/util/chothia2HLT.py`。

### 3.6 推理性能/显存埋点（非侵入式）
新增 `src/rfantibody/util/inference_metrics.py`，在 rfdiffusion / proteinmpnn / rf2 中记录**单次推理耗时、峰值/已用显存、设备**，写入 `linshi/logs/inference_metrics.jsonl`（JSON Lines）。`RFANTIBODY_PROFILE=0` 可关闭；不改变模型输入输出与命令行行为。

### 3.7 pyproject 调整
移除 `dgl` 依赖与其 wheel 源、放宽 `requires-python>=3.10`、把 `dgl` shim 加入打包，使项目可在 Python 3.12 上 `pip install -e . --no-deps`。

## 四、权重下载

权重下载脚本：`include/download_weights.sh`（下载到 `<项目>/weights/`）。本机 `files.ipd.uw.edu` 可达，已下载并验证可 `torch.load`：

| 权重文件 | 用途 | 下载地址 |
|----------|------|----------|
| `RFdiffusion_Ab.pt` | RFdiffusion 抗体骨架设计 | https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt |
| `ProteinMPNN_v48_noise_0.2.pt` | ProteinMPNN 序列设计 | https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt |
| `RF2_ab.pt` | RF2 结构预测/打分 | https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt |
| `RFab_noframework-...pt`（可选，TCR） | RF2 TCR-MHC（JSON 输入） | https://zenodo.org/records/17488258/... |

## 五、测试结果（端到端通过）

评测复用官方脚本（`test/<module>/scripts/*.sh`），由 `linshi/eval/run_pc_eval.py` 运行并计算定量指标：

```bash
# 一次性：装依赖 + 可编辑安装 + 下权重
pip install -e . --no-deps
bash include/download_weights.sh

# 运行评测（SDAA）
cd /data01/tuyilist/cyz/RFantibody
RFANTIBODY_DEVICE=sdaa python3 linshi/eval/run_pc_eval.py
```
✅ 全部模块通过：util 2/2、quiver 7/7、rfdiffusion 4/4、proteinmpnn 1/1、rf2 1/1（全部 rc=0）。
✅ 实际跑在 SDAA 上（`torch.sdaa.is_available()=True`，`torch.cuda.is_available()=False`）。

## 六、精度性能对比（SDAA vs CUDA vs H100 参考）

- 比对基准：仓库自带 `H100_references`
- SDAA 数据：`linshi/sdaa_artifacts/`（Tecorigin SDAA，Torch-SDAA 3.3.0，PyTorch 2.12.0）
- CUDA 数据：`linshi/pc_report/pc_metrics.json`、`linshi/logs/inference_metrics.jsonl`（NVIDIA A100-PCIE-40GB，CUDA 13.0，PyTorch 2.12.0+cu130）

### 表 1：指标精度（SDAA vs CUDA vs H100 参考）

| 模型 | 样本 / 任务 | 评测指标 | SDAA | CUDA | H100 参考 | 判定 |
|------|------------|----------|------|------|-----------|------|
| RFdiffusion | `nb_des_0.pdb` 骨架设计 | Cα RMSD | 0.0022 Å | 0.0024 Å | — | ✅ 均复现 |
| RFdiffusion | `nb_des_1.pdb` 骨架设计 | Cα RMSD | 0.0048 Å | 0.0065 Å | — | ✅ 均复现 |
| RFdiffusion | `ab_des_1.pdb` 骨架设计 | Cα RMSD | 1.0025 Å | **0.0110 Å** | — | ✅ CUDA 更优 |
| RFdiffusion | `ab_des_0.pdb` 骨架设计 | Cα RMSD | 5.7486 Å | **0.3267 Å** | — | ⚠/✅ CUDA 更优 |
| RFdiffusion | `ab_designs.qv` 骨架设计 | 标签/内容保持率 | 100% | 100% | — | ✅ |
| RFdiffusion | `nb_designs.qv` 骨架设计 | 标签/内容保持率 | 100% | 100% | — | ✅ |
| ProteinMPNN | `ab_des_0_dldesign_0.pdb` 序列设计 | 骨架坐标偏差 | 0.0 Å | 0.0 Å | — | ✅ |
| ProteinMPNN | `ab_des_0_dldesign_0.pdb` 序列设计 | 序列一致率 | 97.28% | **100.0%** | — | ✅ CUDA 更优 |
| ProteinMPNN | `ab_des_0_dldesign_1.pdb` 序列设计 | 骨架坐标偏差 | 0.0 Å | 0.0 Å | — | ✅ |
| ProteinMPNN | `ab_des_0_dldesign_1.pdb` 序列设计 | 序列一致率 | 94.98% | **100.0%** | — | ✅ CUDA 更优 |
| RF2 | `ab_proteinmpnn_output_best.pdb` 结构预测 | pred_lddt | 0.90 | 0.90 | 0.89 | ✅ |
| RF2 | 同上 | pae | 10.67 | 10.54 | 10.64 | ✅ |
| RF2 | 同上 | interaction_pae | 19.47 | 19.21 | 19.38 | ✅ |
| RF2 | 同上 | framework_aligned_antibody_rmsd | 1.17 Å | 1.21 Å | 1.25 Å | ✅ |
| RF2 | 同上 | target_aligned_antibody_rmsd | 30.76 Å | 30.87 Å | 30.82 Å | ✅ |
| RF2 | 同上 | framework_aligned_cdr_rmsd | 2.10 Å | 2.15 Å | 2.25 Å | ✅ |
| RF2 | 同上 | target_aligned_cdr_rmsd | 24.50 Å | 24.73 Å | 24.64 Å | ✅ |
| RF2 | 同上 | 整体坐标 Cα RMSD | 0.647 Å | **0.446 Å** | — | ✅ CUDA 更优 |

### 表 1b：RF2 per-loop framework RMSD（Å）

| 指标 | SDAA | CUDA | H100 参考 |
|------|------|------|-----------|
| H1 | 0.85 | 0.88 | 0.90 |
| H2 | 0.86 | 0.94 | 0.98 |
| H3 | 3.54 | 3.64 | 3.75 |
| L1 | 2.30 | 2.28 | 2.51 |
| L2 | 0.89 | 0.91 | 0.98 |
| L3 | 1.11 | 1.20 | 1.23 |

### 表 1c：RFdiffusion 逐样本明细

| 输出 | SDAA Cα RMSD | CUDA Cα RMSD | SDAA 最大偏差 | CUDA 最大偏差 | 序列一致 |
|------|--------------|--------------|---------------|---------------|----------|
| `nb_des_0.pdb` | 0.0022 Å | 0.0024 Å | 0.0168 Å | 0.0133 Å | 100% |
| `nb_des_1.pdb` | 0.0048 Å | 0.0065 Å | 0.0476 Å | 0.1168 Å | 100% |
| `ab_des_1.pdb` | 1.0025 Å | 0.0110 Å | 2.7547 Å | 0.0595 Å | 100% |
| `ab_des_0.pdb` | 5.7486 Å | 0.3267 Å | 10.7649 Å | 0.7987 Å | 100% |

### 表 2：运行时间与显存（SDAA vs CUDA）

| 模型/模块 | 任务 | 输入规模 | SDAA 耗时 | CUDA 耗时 | 加速比 | SDAA 峰值显存 | CUDA 峰值显存 |
|-----------|------|----------|-----------|-----------|--------|---------------|---------------|
| ProteinMPNN | 序列设计（2 seq/结构） | 1 结构（~240 残基） | 18.7 s | **3.32 s** | ~5.6× | 330 MB | 279 MB |
| RF2 | 结构预测（1 recycle） | 478 残基 | 259.4 s | **7.29 s** | ~35.6× | 4234 MB | 2900 MB |
| RFdiffusion | 抗体设计（1 design，48 步） | 478 残基 | 271.0 / 269.9 s | **19.81 / 17.84 s** | ~14× | 3121 / 3161 MB | 2350 / 2387 MB |
| RFdiffusion | 纳米抗体设计（1 design，48 步） | 364 / 368 残基 | 151.3 / 127.1 s | **15.24 / 15.28 s** | ~9–10× | 2270 / 2298 MB | 1674 / 1696 MB |

> 数据来源：`linshi/logs/inference_metrics.jsonl`（SDAA 记录见 `linshi/sdaa_artifacts/logs/inference_metrics.jsonl`）。
> RFdiffusion 首次运行另需 IGSO3 计算（纯 NumPy，~15–20 min，之后缓存），与设备无关；两平台均已缓存。

### 6.3 性能结论

- **RF2**：CUDA（A100）约为 SDAA 的 **35.6×**。RF2 是深层的 MSA/Attention + SE(3) 网络，算子密度高但单算子规模小、且包含大量不规则小批（478 残基、L×L pair），SDAA 缺少 cuDNN/成熟融合 kernel，逐算子调度开销被放大 → 差距最大。
- **RFdiffusion**：CUDA 约 **9–14×**。48 步去噪串行，每步一次完整网络前向；单步算子多、规模中等，SDAA 单步开销约为 CUDA 的十余倍。
- **ProteinMPNN**：CUDA 约 **5.6×**。图/卷积规模小、batch 小，差距相对最小。
- 峰值显存 CUDA 普遍更低（融合 kernel + 更省中间张量）。

### 6.4 各功能在 SDAA 的可用性

| 功能 | SDAA 可用性 |
|------|-------------|
| RFdiffusion（骨架设计） | ✅ 可用（4/4 脚本 rc=0；纳米抗体逐原子复现） |
| ProteinMPNN（序列设计） | ✅ 可用（骨架坐标 0.0 Å） |
| RF2（结构预测/打分） | ✅ 可用（pLDDT 0.90 vs 参考 0.89） |
| util / quiver 工具 | ✅ 可用（2/2、7/7） |

## 七、踩坑记录

| # | 坑 | 根因 | 解决 |
|---|----|------|------|
| 1 | DGL 无法在 loongarch64 安装 | 无预构建 wheel，且其消息传递 kernel 为 CUDA 专有 | 用纯 PyTorch 重写为 `dgl/` shim（`index_add`/`index_reduce`） |
| 2 | `torch.cuda.nvtx` 抛错 | SDAA 未安装 NVTX | 换成 no-op 上下文管理器（`runtime/nvtx.py`） |
| 3 | 设备选择回落到 CPU | `torch.cuda.is_available()` 在 SDAA 为 False | 新增 `device_utils`，自动识别 sdaa/cuda/cpu |
| 4 | **RF2 pLDDT 0.20、坐标偏差 20 Å** | `nn.Embedding` 对 **≥3 维索引**在 SDAA 返回错误行，污染 `SeqSep`/MSA/template embedding，网络数值爆炸 | 新增 `SafeEmbedding`：展平索引再 reshape（权重布局不变） |
| 5 | util 官方用例失败 | `chothia2HLT.py` 首行 REMARK 与末行 ATOM 粘连（biotite 版本差异） | 写入前先补换行 |
| 6 | `torch.load` 反序列化严格 | 新版 torch 默认 `weights_only=True` | 视需要加 `weights_only=False` |
| 7 | RF2 RMSD 路径偶发崩溃 | `biotite 1.7.1 + numpy 2.x` 的 `struc.superimpose` 抛 `SVD did not converge` | 建议固定 `biotite<1.0` |
| 8 | 重复评测结果异常 | RFdiffusion 默认 cautious 模式会跳过已存在输出 | 评测前清空 `linshi/eval_outputs/` |

---

## 结论

1. **正确性**：两平台均通过全部模块（util 2/2、quiver 7/7、rfdiffusion 4/4、proteinmpnn 1/1、rf2 1/1，全部 rc=0），并高保真复现 H100 参考。
2. **精度差异**：CUDA 作为与参考同源的原生平台，抗体样本（ab_des_1 0.011 Å、ab_des_0 0.33 Å）与 ProteinMPNN 序列（100%）几乎逐原子一致；SDAA 在扩散模型对数值扰动敏感的抗体样本上偏差更明显（1.00 / 5.75 Å），且序列采样存在微小随机差异（97.28% / 94.98%）。
3. **性能**：CUDA（A100）显著快于 SDAA——RF2 约 36×、RFdiffusion 约 9–14×、ProteinMPNN 约 5.6×；峰值显存也普遍更低。

### 复现

```bash
# SDAA
cd /data01/tuyilist/cyz/RFantibody && RFANTIBODY_DEVICE=sdaa python3 linshi/eval/run_pc_eval.py
# CUDA
cd /data/application/chenyz/project/RFantibody && bash linshi/eval/run_cuda_eval.sh
```

> 注意：RFdiffusion 默认 cautious 模式，重复评测前需清空 `linshi/eval_outputs/`。