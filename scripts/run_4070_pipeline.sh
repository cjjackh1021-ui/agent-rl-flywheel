#!/usr/bin/env bash
set -euo pipefail

# RTX 4070 SUPER 12GB (Windows WSL2 / Linux) 一键全流程实验脚本
CONFIG="${1:-configs/qwen35_2b_4070.yaml}"
DATA_DIR="${DATA_DIR:-data/generated}"
OUT_DIR="${OUT_DIR:-outputs/qwen35_2b_4070}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[1/4] 构建严格 group_id 防泄漏双工具基准数据集..."
python3 scripts/build_dataset.py --out "${DATA_DIR}" --total-id 1400 --total-ood 100

echo "[2/4] 使用 DeepSeek-V4.1-Flash 合成冷启动验证轨迹（如无需离线重采可跳过）..."
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  python3 scripts/synthesize_teacher.py --data-dir "${DATA_DIR}" --out "${OUT_DIR}/teacher_coldstart.jsonl" --limit 250
else
  echo "未检测到 DEEPSEEK_API_KEY，跳过远程合成，使用基准验证轨迹继续。"
fi

echo "[3/4] 在 RTX 4070 SUPER 12GB 上运行 Base -> SFT -> Prefix OPD/DPO -> GRPO -> DAPO+RAGEN-2 消融..."
python3 scripts/run_experiment.py \
  --config "${CONFIG}" \
  --backend hf_qlora \
  --stage smoke_all \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}"

echo "[4/4] 完成！消融报告已保存至: ${OUT_DIR}/ablation_report.md"
