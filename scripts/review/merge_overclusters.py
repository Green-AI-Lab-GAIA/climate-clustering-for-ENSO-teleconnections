"""Merge over-clustered runs (K60/K100) into K=15/30 by agglomerative clustering of the prototypes.

Prototypes are grouped by average-linkage cosine distance; each day keeps its argmax prototype and
inherits that prototype's group. Writes results/a100-<run>-m<k>/{F,F_val}.pt as one-hot matrices so the
other review scripts read them like any run.
usage: .venv/bin/python scripts/review/merge_overclusters.py
"""
import os, torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

for run in ('t60_b1024', 't100_b1024'):
    ck = torch.load(f'checkpoint/a100-{run}/a100-{run}-best-slim.pth.tar', map_location='cpu', weights_only=False)
    P = torch.nn.functional.normalize(ck['prototypes'].float()).numpy()
    Z = linkage(P, method='average', metric='cosine')
    F = torch.load(f'results/a100-{run}/F.pt', weights_only=False).cpu()
    Fv = torch.load(f'results/a100-{run}/F_val.pt', weights_only=False).cpu()
    for k in (15, 30):
        g = fcluster(Z, k, criterion='maxclust') - 1
        assert len(set(g)) == k, (run, k, len(set(g)))
        out = f'results/a100-{run}-m{k}'; os.makedirs(out, exist_ok=True)
        for name, M in (('F', F), ('F_val', Fv)):
            lab = torch.as_tensor(g, dtype=torch.long)[M.argmax(1)]
            torch.save(torch.nn.functional.one_hot(lab, k).float(), f'{out}/{name}.pt')
        sizes = np.bincount(g[F.argmax(1).numpy()], minlength=k) / len(F) * 100
        print(run, k, 'protos/group', np.bincount(g).tolist(), 'day%% min %.1f max %.1f' % (sizes.min(), sizes.max()))
