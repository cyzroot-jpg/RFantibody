#!/bin/bash
# 统一的流水线运行脚本：SDAA 与 CUDA 使用同一份，保证"条件完全一致"。
# 用法: bash run_pipeline.sh <输出目录>
set -e
OUT=${1:-out}
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
mkdir -p "$OUT"

echo "=== 环境/设备 ==="
python -c "from rfantibody.util.device_utils import get_backend,device_str,get_device_name; print('backend=',get_backend(),'device=',device_str(),'name=',get_device_name())"

echo "=== 步骤1: RFdiffusion (1 design, 完整 49 步 final_step=1) ==="
rfdiffusion \
    -t test/rfdiffusion/inputs_for_test/rsv_site3.pdb \
    -f test/rfdiffusion/inputs_for_test/h-NbBCII10.pdb \
    -q "$OUT/1_rfdiffusion.qv" \
    -n 1 \
    -l "H1:7,H2:6,H3:5-13" \
    -h "T305,T456" \
    --diffuser-t 50 --final-step 1 --deterministic --no-trajectory

echo "=== 步骤2: ProteinMPNN (2 seqs, T=0.2) ==="
proteinmpnn \
    -q "$OUT/1_rfdiffusion.qv" \
    --output-quiver "$OUT/2_proteinmpnn.qv" \
    -n 2 -t 0.2 --deterministic

echo "=== 步骤3: RF2 (5 recycles, seed 42) ==="
rf2 \
    -q "$OUT/2_proteinmpnn.qv" \
    --output-quiver "$OUT/3_rf2.qv" \
    -r 5 --hotspot-show-prop 0 --seed 42

echo "=== 分数 ==="
qvscorefile "$OUT/3_rf2.qv"
echo "=== DONE: $OUT ==="