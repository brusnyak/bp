#!/bin/bash
# Piper training / HQ corpus status - one command, no tokens burned.
#   bash scripts/train_status.sh
# Exit codes: 0 = running, 10 = finished+exported, 20 = idle/crashed, 30 = never started
LOG=${1:-/tmp/overnight_hq.log}
REPO=/Users/yegor/Documents/STU/BP
CKPT_DIR=/tmp/piper_finetune_work_sk/out/lightning_logs/version_0/checkpoints
ONNX=$REPO/backend/tts/piper_models/me_omni_piper_sk.onnx

echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

# 1. is the script still running?
if pgrep -f "overnight_hq.sh|finetune_personal_voice" >/dev/null; then
  echo "STATE: RUNNING"
  STATE=0
else
  STATE=20
  echo "STATE: NOT RUNNING (script finished, crashed, or never started)"
fi

# 2. phase reached
echo "--- phases ---"
grep -E "^--- PHASE|exit:|milestone|^=== OVERNIGHT|^omni_hq_" "$LOG" 2>/dev/null | tail -10
[ -f "$LOG" ] || { echo "no log at $LOG"; exit 30; }

# 3. training progress
if [ -d "$CKPT_DIR" ]; then
  EPOCH=$(grep -oE "Epoch [0-9]+" "$LOG" | awk '{print $2}' | sort -n | tail -1)
  LAST=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -1)
  NSTEPS=$(grep -oE "max_steps [0-9]+" "$LOG" | head -1 | awk '{print $2}')
  echo "--- progress ---"
  echo "epoch: ${EPOCH:-?} / ~$(( ${NSTEPS:-2500} / 10 ))"
  if [ -n "$LAST" ]; then
    STEPS=$( cd "$REPO" && timeout 200 .venv-train/bin/python -c "
import pathlib, torch, torch.serialization
torch.serialization.add_safe_globals([pathlib.PosixPath])
ck = torch.load('$LAST', map_location='cpu', weights_only=False)
print(ck.get('global_step'))
" 2>/dev/null | tail -1 )
    echo "newest ckpt: $(basename "$LAST")  global_step=${STEPS:-?}"
    if [ -n "$NSTEPS" ] && [ -n "$STEPS" ]; then
      [ "$STEPS" -ge "$NSTEPS" ] && echo "  -> reached max_steps" || echo "  -> $((NSTEPS - STEPS)) steps short"
    fi
  fi
  echo "ckpt count: $(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l | tr -d ' ')"
fi

# 4. export
echo "--- export ---"
if [ -f "$ONNX" ]; then
  echo "ONNX: $(ls -lh "$ONNX" | awk '{print $5}')  $(date -r "$ONNX" '+%H:%M')"
  grep -q "Done. Personal voice ready" "$LOG" && { echo "VERDICT: DONE - voice exported"; [ "$STATE" = 20 ] && STATE=10; }
else
  echo "ONNX: not yet written"
fi

# 5. errors. NOTE: `grep -c` prints 0 AND exits 1 on no match, so a bare
# `|| echo 0` would append a second line and make the test below always true.
ERR=$(grep -cE "Traceback|out of memory|Killed" "$LOG" 2>/dev/null || true)
ERR=${ERR:-0}
if [ "$ERR" != "0" ]; then
  echo "ERRORS: $ERR found - check $LOG"
  STATE=20
fi

echo "--- last log line ---"
tail -c 200 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1
exit $STATE
