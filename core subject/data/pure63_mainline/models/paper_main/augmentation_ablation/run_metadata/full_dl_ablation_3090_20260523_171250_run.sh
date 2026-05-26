#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/SERS-multicomponent-analysis
STAGE="data/pure63_mainline/models/paper_main/augmentation_ablation"
RUN_ID="full_dl_ablation_3090_20260523_171250"
mkdir -p "$STAGE/logs" "$STAGE/run_metadata"
LOG="$STAGE/logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Full 4-DL augmentation ablation started: $RUN_ID ==="
date
hostname
pwd

echo "=== Source state ==="
echo "local_source_commit=ba0d2b23f22e0a484ac5287c6663c84b21047ec5" | tee "$STAGE/run_metadata/${RUN_ID}_source_commit.txt"
cat > "$STAGE/run_metadata/${RUN_ID}_source_note.txt" <<'EOF'
Minimal project package uploaded from local workspace.
Remote directory is not a git clone; source commit below is local HEAD at packaging time.
Local workspace had documentation/result artifact changes, but src/scripts/data package was used for execution.
EOF

echo "=== Environment ==="
python3 --version | tee "$STAGE/run_metadata/${RUN_ID}_python_version.txt"
python3 - <<'PY' | tee "$STAGE/run_metadata/full_dl_ablation_3090_20260523_171250_environment.txt"
import sys, platform
print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import torch
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', getattr(torch.version, 'cuda', None))
    if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('Torch check failed:', repr(exc))
for mod in ['numpy','pandas','sklearn','scipy','xgboost']:
    try:
        m=__import__(mod)
        print(f'{mod}:', getattr(m, '__version__', 'unknown'))
    except Exception as exc:
        print(f'{mod} check failed:', repr(exc))
PY
python3 -m pip freeze > "$STAGE/run_metadata/${RUN_ID}_pip_freeze.txt"

cat > "$STAGE/run_metadata/${RUN_ID}_command_manifest.txt" <<'EOF'
Experiment: Full DL augmentation ablation after Spectrum-KAN targeted audit
Models: 1D-CNN, 1D-ResNet, Spectrum-KAN, KAN-CNN
Variants: p1, raw
Modes: no_aug, aug_no_mixup, composition_mixup
Tasks: TASKS from src.config (6 paper-main tasks)
Split: data/pure63_mainline/splits/cv_split_random_5fold.csv via fold_id
Output: data/pure63_mainline/models/paper_main/augmentation_ablation/{mode}/
EOF

echo "=== Data check ==="
python3 - <<'PY'
from pathlib import Path
import numpy as np, pandas as pd
for p in ['data/pure63_mainline/processed/wavenumber.npy','data/pure63_mainline/processed/X_p1.npy','data/pure63_mainline/processed/X_raw.npy','data/pure63_mainline/splits/cv_split_random_5fold.csv']:
    path=Path(p)
    print(p, path.exists(), path.stat().st_size if path.exists() else None)
print('X_p1', np.load('data/pure63_mainline/processed/X_p1.npy').shape)
print('X_raw', np.load('data/pure63_mainline/processed/X_raw.npy').shape)
meta=pd.read_csv('data/pure63_mainline/splits/cv_split_random_5fold.csv')
print('split', meta.shape, sorted(meta.fold_id.unique()))
PY

echo "=== Command ==="
echo "python3 -u scripts/analysis/run_augmentation_ablation.py --models 1D-CNN 1D-ResNet Spectrum-KAN KAN-CNN --variants p1 raw --modes no_aug aug_no_mixup composition_mixup"

echo "=== Run ==="
python3 -u scripts/analysis/run_augmentation_ablation.py   --models 1D-CNN 1D-ResNet Spectrum-KAN KAN-CNN   --variants p1 raw   --modes no_aug aug_no_mixup composition_mixup

echo "=== Verify outputs ==="
for mode in no_aug aug_no_mixup composition_mixup; do
  echo "--- $mode ---"
  for variant in p1 raw; do
    summary="$STAGE/$mode/cv_results_summary_${variant}.csv"
    detail="$STAGE/$mode/cv_results_detail_${variant}.csv"
    preds="$STAGE/$mode/cv_predictions_${variant}.csv"
    for file in "$summary" "$detail" "$preds"; do
      if [ ! -f "$file" ]; then echo "MISSING: $file"; exit 1; fi
      rows=$(wc -l < "$file")
      echo "$file rows=$rows"
    done
  done
  for idx in benchmark_full_matrix.csv benchmark_best_by_task.csv benchmark_source_manifest.csv benchmark_source_model_mean.csv benchmark_source_preprocess_mean.csv; do
    if [ ! -f "$STAGE/$mode/$idx" ]; then echo "MISSING: $STAGE/$mode/$idx"; exit 1; fi
  done
done

echo "=== Pack return bundle ==="
tar czf "${RUN_ID}.tar.gz"   data/pure63_mainline/models/paper_main/augmentation_ablation/no_aug   data/pure63_mainline/models/paper_main/augmentation_ablation/aug_no_mixup   data/pure63_mainline/models/paper_main/augmentation_ablation/composition_mixup   data/pure63_mainline/models/paper_main/augmentation_ablation/logs   data/pure63_mainline/models/paper_main/augmentation_ablation/run_metadata

echo "=== Done ==="
date
echo "RETURN_BUNDLE=${RUN_ID}.tar.gz"
