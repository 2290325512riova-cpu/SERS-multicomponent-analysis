#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/SERS-multicomponent-analysis
STAGE="data/pure63_mainline/models/paper_main/augmentation_ablation"
RUN_ID="spectrum_kan_audit_3090_20260523_005715"
mkdir -p "$STAGE/logs" "$STAGE/run_metadata"
LOG="$STAGE/logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Spectrum-KAN targeted audit started: $RUN_ID ==="
date
hostname
pwd

echo "=== Git state ==="
git rev-parse HEAD | tee "$STAGE/run_metadata/${RUN_ID}_git_commit.txt"
git branch --show-current | tee "$STAGE/run_metadata/${RUN_ID}_git_branch.txt"
git status --short | tee "$STAGE/run_metadata/${RUN_ID}_git_status.txt"

echo "=== Environment ==="
python3 --version | tee "$STAGE/run_metadata/${RUN_ID}_python_version.txt"
python3 - <<'PY' | tee "data/pure63_mainline/models/paper_main/augmentation_ablation/run_metadata/spectrum_kan_audit_3090_20260523_005715_environment.txt"
import sys, platform
print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import torch
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', getattr(torch.version, 'cuda', None))
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('Torch check failed:', repr(exc))
try:
    import xgboost
    print('XGBoost:', xgboost.__version__)
except Exception as exc:
    print('XGBoost check failed:', repr(exc))
PY
pip freeze > "$STAGE/run_metadata/${RUN_ID}_pip_freeze.txt"

cat > "$STAGE/run_metadata/${RUN_ID}_command_manifest.txt" <<'EOF'
Experiment: Spectrum-KAN targeted audit before full DL augmentation ablation
Purpose: determine whether Spectrum-KAN G2 anomaly is reproducible and whether no_aug/raw changes it
Models: Spectrum-KAN
Variants: p1 raw
Modes: no_aug aug_no_mixup composition_mixup
Tasks: TASKS from src.config (6 paper-main tasks)
Split: data/pure63_mainline/splits/cv_split_random_5fold.csv via fold_id
Output: data/pure63_mainline/models/paper_main/augmentation_ablation/{mode}/
EOF

echo "=== Command ==="
echo "python3 -u scripts/analysis/run_augmentation_ablation.py --models Spectrum-KAN --variants p1 raw --modes no_aug aug_no_mixup composition_mixup"

echo "=== Run ==="
python3 -u scripts/analysis/run_augmentation_ablation.py   --models Spectrum-KAN   --variants p1 raw   --modes no_aug aug_no_mixup composition_mixup

echo "=== Verify outputs ==="
for mode in no_aug aug_no_mixup composition_mixup; do
  echo "--- $mode ---"
  for variant in p1 raw; do
    summary="$STAGE/$mode/cv_results_summary_${variant}.csv"
    detail="$STAGE/$mode/cv_results_detail_${variant}.csv"
    preds="$STAGE/$mode/cv_predictions_${variant}.csv"
    for file in "$summary" "$detail" "$preds"; do
      if [ ! -f "$file" ]; then
        echo "MISSING: $file"
        exit 1
      fi
      rows=$(wc -l < "$file")
      echo "$file rows=$rows"
    done
  done
  for idx in benchmark_full_matrix.csv benchmark_best_by_task.csv benchmark_source_manifest.csv benchmark_source_model_mean.csv benchmark_source_preprocess_mean.csv; do
    if [ ! -f "$STAGE/$mode/$idx" ]; then
      echo "MISSING: $STAGE/$mode/$idx"
      exit 1
    fi
  done
done

echo "=== Pack return bundle ==="
tar czf "${RUN_ID}.tar.gz"   data/pure63_mainline/models/paper_main/augmentation_ablation/no_aug   data/pure63_mainline/models/paper_main/augmentation_ablation/aug_no_mixup   data/pure63_mainline/models/paper_main/augmentation_ablation/composition_mixup   data/pure63_mainline/models/paper_main/augmentation_ablation/logs   data/pure63_mainline/models/paper_main/augmentation_ablation/run_metadata

echo "=== Done ==="
date
echo "RETURN_BUNDLE=${RUN_ID}.tar.gz"
