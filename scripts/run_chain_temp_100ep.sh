#!/bin/bash
# Temperature-only runs (15 and 30 prototypes) on the same 100-epoch / val_metric=msn
# protocol as the temp+prep -100ep runs. Sequential on purpose: the A4000 is shared.
cd "$(dirname "$0")/.."
for K in 15 30; do
  tag=temperature-era-amazonia-$K-anom-100ep
  echo "$(date) start $tag"
  .venv/bin/python main_msn.py --fname configs/temp_era_anom_${K}_100ep.yaml --devices cuda:0 > logs/$tag.log 2>&1
  echo "$(date) end $tag exit=$?"
done
