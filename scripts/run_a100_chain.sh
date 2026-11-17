#!/bin/bash
# Run configs/a100/<name>.yaml one after another on one GPU.
#   setsid scripts/run_a100_chain.sh                 # all runs, priority order below
#   setsid scripts/run_a100_chain.sh t15_s1 t30_s1   # only these
#   DEVICE=cuda:1 setsid scripts/run_a100_chain.sh ...
# Launch with setsid and stop with `kill -- -$(cat logs/a100_chain.pid)`:
# main_msn.py uses mp.spawn, so killing only the launcher leaves training running.
# Runs that already have a -best checkpoint are skipped (delete it to rerun).
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda:0}
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(t15_s0 t30_s0 t30_nomean p30_daily_raw p30_daily_cbrt p30_roll30 p10_monthly
        t60_b1024 t100_b1024 t15_s1 t15_s2 t30_s1 t30_s2)
fi
mkdir -p logs
echo $$ > logs/a100_chain.pid
for name in "${RUNS[@]}"; do
  cfg=configs/a100/$name.yaml
  [ -f "$cfg" ] || { echo "$(date) missing $cfg"; continue; }
  if [ -f checkpoint/a100-$name/a100-$name-best.pth.tar ]; then
    echo "$(date) skip $name (best checkpoint exists)"; continue
  fi
  echo "$(date) start $name"
  .venv/bin/python main_msn.py --fname "$cfg" --devices "$DEVICE" > logs/a100-$name.log 2>&1
  echo "$(date) end $name exit=$?"
done
