"""Encode every day with the A100 runs' best checkpoints -> results/a100-<run>/{E,F,E_val,F_val}.pt

usage: .venv/bin/python scripts/review/infer_a100.py [runs...]   (default: every checkpoint/a100-*)
Uses the slim checkpoints brought back from the HPC (<tag>-best-slim.pth.tar: target_encoder +
prototypes only). Skips runs whose results already exist.
"""
import glob, os, sys
sys.path.insert(0, '.')
import src.inference as inf

runs = sys.argv[1:] or sorted(os.path.basename(d)[5:] for d in glob.glob('checkpoint/a100-*'))
for run in runs:
    out = f'results/a100-{run}'
    if os.path.exists(f'{out}/F_val.pt'):
        print('skip', run); continue
    params, dataset = inf.read_data(f'checkpoint/a100-{run}/params-a100-{run}.yaml', validation=True)
    enc, prot = inf.load_model(params, model_suffix='-best-slim.pth.tar')
    inf.get_model_results(read_path=out, params=params, dataset=dataset, encoder=enc, prototypes=prot)
    inf.get_model_results(read_path=out, params=params, dataset=dataset.validation_imgs,
                          encoder=enc, prototypes=prot, validation=True)
    print('done', run, flush=True)
