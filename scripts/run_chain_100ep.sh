#!/bin/bash
# Runs experiment 2 (cube-root tp) and then its control (raw tp), both with the
# 100-epoch / val_metric=msn schedule. Sequential on purpose: the A4000 is shared.
cd "$(dirname "$0")/.."
for tag in temp-prep-cbrt-era-amazonia-30-anom-100ep temp-prep-era-amazonia-30-anom-100ep; do
  cfg=configs/$(echo $tag | sed -e 's/temp-prep-cbrt-era-amazonia-30-anom-100ep/temp_prep_cbrt_era_anom_30_100ep/' -e 's/temp-prep-era-amazonia-30-anom-100ep/temp_prep_era_anom_30_100ep/').yaml
  echo "$(date) start $tag ($cfg)"
  .venv/bin/python main_msn.py --fname $cfg --devices cuda:0 > logs/$tag.log 2>&1
  echo "$(date) end $tag exit=$?"
done
