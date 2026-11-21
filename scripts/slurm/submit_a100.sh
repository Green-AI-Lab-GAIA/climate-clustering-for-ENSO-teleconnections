#!/bin/bash
# Submit the A100 experiment queue on the FGV HPC (run on hpchead, from the repo root).
#   scripts/slurm/submit_a100.sh                  # all runs, in priority order
#   scripts/slurm/submit_a100.sh t15_s1 t30_s1    # only these
#   DRY=1 scripts/slurm/submit_a100.sh            # print the sbatch lines only
# Jobs are chained with --dependency=afterany so they run in this order, one at a time
# (the node runs one job anyway); a failed run does not block the next one.
# Resubmitting a run is safe: configs use load_checkpoint: auto, so it resumes from -latest
# (a run that already finished all epochs exits after reloading, keeping its -best).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/slurm

# name  --time (generous: ~3x the expected duration; measured times go in docs/A100_RUNBOOK.md)
declare -A TIME=(
  [t15_s0]=05:00:00 [t15_s1]=05:00:00 [t15_s2]=05:00:00
  [t30_s0]=05:00:00 [t30_s1]=05:00:00 [t30_s2]=05:00:00
  [t30_nomean]=05:00:00 [t30_nomean_200ep]=09:00:00
  [p30_daily_raw]=05:00:00 [p30_daily_cbrt]=05:00:00 [p30_roll30]=05:00:00
  [p10_monthly]=03:00:00
  [t60_b1024]=09:00:00 [t100_b1024]=09:00:00
)
ORDER=(t15_s0 t30_s0 t30_nomean p30_daily_raw p30_daily_cbrt p30_roll30 p10_monthly
       t60_b1024 t100_b1024 t15_s1 t30_s1 t15_s2 t30_s2)
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=("${ORDER[@]}")

prev=""
for run in "${RUNS[@]}"; do
  [ -f "configs/a100/$run.yaml" ] || { echo "skip $run: no config"; continue; }
  t=${TIME[$run]:-05:00:00}
  dep=${prev:+--dependency=afterany:$prev}
  cmd=(sbatch --parsable --job-name="msn-$run" --time="$t" $dep --export=ALL,RUN="$run" scripts/slurm/train.sbatch)
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; prev=DRY; continue; fi
  prev=$("${cmd[@]}")
  echo "$prev  msn-$run  time=$t ${dep}"
done
squeue --me
