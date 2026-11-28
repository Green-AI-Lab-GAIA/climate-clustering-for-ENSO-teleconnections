"""P2OT pseudo-labels (tp30_ppot_l{0p3,1,3}) vs Sinkhorn MSN (tp30_mask) vs PCA k-means (REVIEW §14.19).

Same field, metrics and ENSO test as mask_eval.py (its setup and metric functions are executed from that file,
so the numbers are directly comparable with §14.15). Extra: rare-type check. For every EOF k-means K = 30
cluster (seed 0), best pattern match (centred r of the mean z~ maps) among each MSN run's native clusters,
and the size of that match; plus how many native clusters are < 1.5 % of days, and prototypes never used.
usage: .venv/bin/python -I scripts/review/ppot_eval.py scripts/review <out_dir>
"""
import sys, os
_src = open(os.path.join(sys.argv[1], 'mask_eval.py')).read()
exec(_src[:_src.index('train_years = ')])          # field A, AN, clim, enso(), field_stats(), pca_scores(), load_run()

RUNS = {'tp30_mask': 'Sinkhorn + me-max (tp30_mask, ep57)',
        'tp30_ppot_l3': 'P2OT λ 3 (ep98)', 'tp30_ppot_l1': 'P2OT λ 1 (ep95)', 'tp30_ppot_l0p3': 'P2OT λ 0.3 (ep98)'}
Z = pca_scores(np.arange(ntr))
rows, labs = [], {}


def add(name, K, lab, seed=0, extra=None):
    st = field_stats(lab, K); en = enso(lab, K)
    sz = np.bincount(lab, minlength=K) / N * 100
    labs[(name, K, seed)] = (lab, st['means'])
    rows.append(dict(method=name, K=K, seed=seed, used=int((sz > 0).sum()), eta2=st['eta2'], coh=st['coh'],
                     coh05=st['coh05'], pat=st['pat'], min_size=sz[sz > 0].min(), max_size=sz.max(),
                     n_small=int(((sz > 0) & (sz < 1.5)).sum()), size_cv=sz.std() / sz.mean(),
                     oni_r2=en['r2'], nsig=en['nsig'], **(extra or {})))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)


for s in (0, 1, 2):
    add('PCA k-means', 30, KMeans(30, n_init=10, random_state=s).fit(Z[:ntr]).predict(Z), s)
add('PCA k-means', 10, KMeans(10, n_init=10, random_state=0).fit(Z[:ntr]).predict(Z))
for r, nm in RUNS.items():
    E, F = load_run(r)
    add(f'{nm} native', 30, F.argmax(1))
    add(f'{nm} k-means', 10, KMeans(10, n_init=10, random_state=0).fit(E[:ntr]).predict(E))

df = pd.DataFrame(rows); df.to_csv(f'{out}/ppot_eval.csv', index=False, float_format='%.4f')
g = df.groupby(['method', 'K'], sort=False).mean(numeric_only=True).reset_index()


def centred(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


# rare-type check against EOF k-means K = 30 (seed 0)
lab_e, M_e = labs[('PCA k-means', 30, 0)]
sz_e = np.bincount(lab_e, minlength=30) / N * 100
Tx_e = np.array([AN[torch.as_tensor(np.where(lab_e == j)[0]), 1].mean().item() for j in range(30)])
Tn_e = np.array([AN[torch.as_tensor(np.where(lab_e == j)[0]), 0].mean().item() for j in range(30)])
P_e = np.array([100 * AN[torch.as_tensor(np.where(lab_e == j)[0]), 2].sum().item()
                / clim[mon[lab_e == j], 2].sum().item() for j in range(30)])
rare = pd.DataFrame(dict(eof=np.arange(30), size=sz_e, Tmin=Tn_e, Tmax=Tx_e, P_pct=P_e))
for r, nm in RUNS.items():
    lab_m, M_m = labs[(f'{nm} native', 30, 0)]
    sz_m = np.bincount(lab_m, minlength=30) / N * 100
    C = centred(M_e) @ centred(M_m).T
    C[:, sz_m == 0] = -np.inf
    j = C.argmax(1)
    rare[f'r_{r}'] = C[np.arange(30), j]; rare[f'size_{r}'] = sz_m[j]
rare = rare.sort_values('size').reset_index(drop=True)
rare.to_csv(f'{out}/rare_types.csv', index=False, float_format='%.3f')

f2 = lambda v, f: '–' if pd.isna(v) else f.format(v)
L = ['# P2OT vs Sinkhorn MSN vs PCA k-means (detrended Tmin, Tmax, cbrt tp; full field)', '',
     'Metrics as in mask_eval.md. used = prototypes with ≥ 1 day; n<1.5% = clusters below 1.5 % of days; '
     'size CV = sd/mean of cluster sizes. PCA K = 30 averaged over 3 k-means seeds.', '',
     '| method | K | used | eta2 | coh | coh>0.5 | pat | min–max size % | n<1.5% | size CV | ONI R2 | n ONI |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|']
for _, x in g.iterrows():
    L.append(f'| {x.method} | {int(x.K)} | {x.used:.0f} | {x.eta2:.3f} | {x.coh:.2f} | {x.coh05:.0%} | {x.pat:.2f} | '
             f'{x.min_size:.1f}–{x.max_size:.1f} | {x.n_small:.0f} | {x.size_cv:.2f} | {x.oni_r2:.3f} | {x.nsig:.1f} |')
L += ['', '## Rare EOF types (EOF k-means K = 30, seed 0, 8 smallest): best-matching native cluster per run (pattern r, size %)', '',
      '| EOF size % | Tmin °C | Tmax °C | P % | ' + ' | '.join(RUNS.values()) + ' |',
      '|---|---|---|---|' + '---|' * len(RUNS)]
for _, x in rare.head(8).iterrows():
    L.append(f'| {x["size"]:.1f} | {x.Tmin:+.2f} | {x.Tmax:+.2f} | {x.P_pct:+.0f} | ' +
             ' | '.join(f'{x[f"r_{r}"]:.2f} ({x[f"size_{r}"]:.1f})' for r in RUNS) + ' |')
L += ['', 'median best-match r over all 30 EOF types: ' +
      ', '.join(f'{nm} {np.median(rare[f"r_{r}"]):.2f}' for r, nm in RUNS.items()),
      'median best-match r, 8 smallest EOF types: ' +
      ', '.join(f'{nm} {np.median(rare.head(8)[f"r_{r}"]):.2f}' for r, nm in RUNS.items())]
open(f'{out}/ppot_eval.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L), flush=True)

# size distributions figure
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for (name, K, s), (lab, _) in labs.items():
    if K != 30 or s != 0 or 'k-means' in name and 'PCA' not in name: continue
    sz = np.sort(np.bincount(lab, minlength=30) / N * 100)[::-1]
    ax.plot(np.arange(1, 31), sz, 'o-', ms=3, label=name)
ax.axhline(100 / 30, color='k', lw=0.6, ls=':'); ax.set_xlabel('cluster rank'); ax.set_ylabel('% of days')
ax.set_title('Cluster sizes, K = 30'); ax.legend(fontsize=7)
fig.savefig(f'{out}/sizes.png', dpi=110); plt.close(fig)
