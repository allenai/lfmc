#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-results/raw}"
WORKSPACE="${BEAKER_WORKSPACE:?Set BEAKER_WORKSPACE environment variable}"

mkdir -p "$OUTPUT_DIR"

echo "Listing lfmc_finetune experiments in $WORKSPACE..."
EXP_IDS=$(beaker workspace experiments "$WORKSPACE" --text lfmc_finetune --format json | \
  python3 -c "import json,sys; [print(e['id']) for e in json.load(sys.stdin)]")

TOTAL=$(echo "$EXP_IDS" | wc -l | tr -d ' ')
echo "Found $TOTAL experiments. Downloading results.json and experiment_config.json..."

COUNT=0
for exp_id in $EXP_IDS; do
  COUNT=$((COUNT + 1))
  echo "[$COUNT/$TOTAL] Fetching $exp_id..."
  beaker experiment results "$exp_id" --output "$OUTPUT_DIR/$exp_id" --prefix results.json 2>/dev/null || true
  beaker experiment results "$exp_id" --output "$OUTPUT_DIR/$exp_id" --prefix experiment_config.json 2>/dev/null || true

  # Inject completed_at from Beaker job metadata into each experiment_config.json
  FINALIZED=$(beaker experiment get "$exp_id" --format json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)[0]
jobs = data.get('jobs', [])
if jobs:
    print(jobs[-1].get('status', {}).get('finalized', ''))
" 2>/dev/null || true)

  if [ -n "$FINALIZED" ]; then
    find "$OUTPUT_DIR/$exp_id" -name experiment_config.json -exec python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    config = json.load(f)
config['completed_at'] = sys.argv[2]
with open(path, 'w') as f:
    json.dump(config, f)
" {} "$FINALIZED" \;
  fi
done

echo "Done. Results saved to $OUTPUT_DIR/"
echo "Run: PYENV_VERSION=lfmc collect-results --results-dir $OUTPUT_DIR"
