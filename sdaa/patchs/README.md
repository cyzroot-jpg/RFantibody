# RFantibody SDAA 适配 patch 集

由 `git diff main..adapt/sdaa` 生成（已剔除 `linshi/`、`.opencode/`、`sdaa/` 等评测产物）。

## 文件

| 文件 | 内容 |
|------|------|
| `rfantibody_sdaa_adaptation.patch` | **完整适配 patch**（全部 23 个代码文件，推荐） |
| `01_dgl_pure_torch_shim.patch` | 纯 PyTorch 重写 DGL（`dgl/` 包） |
| `02_nvtx_noop.patch` | `torch.cuda.nvtx` → no-op（SDAA 不可用） |
| `03_embedding_sdda_bugfix.patch` | **`nn.Embedding` ≥3 维索引 SDAA bug 修复**（`SafeEmbedding`） & 替换调用点 |
| `04_device_abstraction_and_metrics.patch` | 设备抽象 `device_utils` + 性能埋点 `inference_metrics` + 各推理脚本/模块接入 |
| `05_packaging_and_util_fix.patch` | `pyproject.toml` 调整 + `chothia2HLT.py` 换行 bug 修复 |

## 应用方式

```bash
# 基于干净的 main 分支
git checkout main
git apply /path/to/sdaa/patchs/rfantibody_sdaa_adaptation.patch

# 或按类别依次应用
for p in 01 02 03 04 05; do git apply sdaa/patchs/${p}_*.patch; done
```

> 注：patch 基于 upstream `main`（`3e5a5bb`）。首次应用前建议 `git apply --check`。
> 若仅需某个修复，可单独应用对应类别 patch。