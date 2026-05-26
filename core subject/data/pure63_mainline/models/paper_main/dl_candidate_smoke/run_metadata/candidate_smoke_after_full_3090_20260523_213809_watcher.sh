#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/SERS-multicomponent-analysis
FULL_RUN_ID="full_dl_ablation_3090_20260523_171250"
RUN_ID="candidate_smoke_after_full_3090_20260523_213809"
OUT_ROOT="data/pure63_mainline/models/paper_main/dl_candidate_smoke"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/run_metadata"
LOG="$OUT_ROOT/logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Candidate smoke watcher started: $RUN_ID ==="
date
hostname
pwd
echo "Waiting for full ablation bundle before starting candidate smoke..."
while true; do
  if [ -f "${FULL_RUN_ID}.tar.gz" ] && ! pgrep -f 'scripts/analysis/run_augmentation_ablation.py' >/dev/null 2>&1; then
    break
  fi
  date
  ps -eo pid,etime,cmd | grep -E 'run_augmentation_ablation|full_dl_ablation' | grep -v grep || true
  sleep 300
done

echo "=== Full ablation appears complete; starting candidate smoke ==="
date
python3 -m py_compile src/models.py scripts/analysis/run_dl_candidate_smoke.py
python3 - <<'PY'
from src.models import MODEL_REGISTRY
print('candidate_registry_ok=', 'RamanNet-Lite' in MODEL_REGISTRY and 'PatchTransformer' in MODEL_REGISTRY)
PY
cat > "$OUT_ROOT/run_metadata/${RUN_ID}_command_manifest.txt" <<'EOF'
Experiment: DL candidate smoke after full DL augmentation ablation
Models: RamanNet-Lite PatchTransformer
Variants: p1 raw
Tasks: G2_mg_molar_grade G3_mba_molar_grade
Mode: composition_mixup
Output: data/pure63_mainline/models/paper_main/dl_candidate_smoke/composition_mixup/
EOF
python3 -u scripts/analysis/run_dl_candidate_smoke.py   --models RamanNet-Lite PatchTransformer   --variants p1 raw   --tasks G2_mg_molar_grade G3_mba_molar_grade   --mode composition_mixup

echo "=== Verify candidate outputs ==="
find "$OUT_ROOT" -maxdepth 3 -type f -printf '%p rows=' -exec sh -c 'case "$1" in *.csv) wc -l < "$1";; *) echo -;; esac' _ {} \; | sort

echo "=== Pack candidate bundle ==="
tar -czf "${RUN_ID}.tar.gz"   "$OUT_ROOT/composition_mixup"   "$OUT_ROOT/logs"   "$OUT_ROOT/run_metadata"
ls -lh "${RUN_ID}.tar.gz"
echo "RETURN_BUNDLE=${RUN_ID}.tar.gz"
date
echo "=== Done ==="
