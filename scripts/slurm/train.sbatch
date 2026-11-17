#!/bin/bash
# One MSN training run on the FGV HPC A100 node. Submitted by scripts/slurm/submit_a100.sh:
#   sbatch --job-name=msn-<run> --time=HH:MM:SS --export=ALL,RUN=<run> scripts/slurm/train.sbatch
# FGV rules: --time is mandatory (<= 24h); never pass --mem or -c (the job gets the whole node).
# If the node reboots, Slurm requeues the job; configs use meta.load_checkpoint: auto,
# so it resumes from checkpoint/a100-<run>/a100-<run>-latest.pth.tar.
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${RUN:?set RUN=<config name in configs/a100>}"
CFG=configs/a100/$RUN.yaml
[ -f "$CFG" ] || { echo "missing $CFG"; exit 2; }

source .venv/bin/activate
echo "$(date -Is) job $SLURM_JOB_ID run=$RUN host=$(hostname) restarts=${SLURM_RESTART_COUNT:-0}"
echo "git $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted files"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

status=0
python main_msn.py --fname "$CFG" --devices cuda:0 || status=$?
echo "$(date -Is) end run=$RUN exit=$status"
exit $status
