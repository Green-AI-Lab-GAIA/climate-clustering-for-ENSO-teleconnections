"""Intra-cluster dispersion deep-dive for one partition (for understanding, not for the paper).

usage: dispersion_detail.py <scratch_dir> <out_dir> <run label in review_lib.RUNS>
Writes figures + per-cluster table to <out_dir>/<slug>/.
"""
import sys, os, re
sys.path.insert(0, sys.argv[1])
from review_lib import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

label = sys.argv[3]
slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
out = os.path.join(sys.argv[2], slug); os.makedirs(out, exist_ok=True)
rdir, _ = RUNS[label]
ctr, cv, k, F, Fv = run_assignments(rdir)
F = F.cpu()
E = torch.load(f'{rdir}/E.pt', weights_only=False).float().cpu()

X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t'))
off = len(t_tr) - len(ctr); assert off in (0, 366)
X, t_tr = X[off:], t_tr[off:]
n, V, H, W = X.shape
Tm = X.mean(1)                                   # (n,H,W) mean of Tmin/Tmax anomaly
level = Tm.mean((1, 2)).numpy()                  # daily domain-mean anomaly
cid = np.asarray(ctr)

# ENSO association (|shift|>=24 rotation null) to mark clusters in plots
m = monthly_from(ctr, cv, t_tr, t_v, k, oni_table())
r = exact_rotation_stats(m, k, min_shift=24)
enso = {j for j in range(k) if r['p_fwer'][j] < 0.05}

# --- per-cluster statistics -------------------------------------------------- #
flat = Tm.reshape(n, -1)
comp = torch.stack([flat[torch.as_tensor(cid == j)].mean(0) for j in range(k)])
a = flat - flat.mean(1, keepdim=True); c = comp - comp.mean(1, keepdim=True)
own = c[torch.as_tensor(cid)]
pc = ((a * own).sum(1) / (a.norm(dim=1) * own.norm(dim=1) + 1e-9)).numpy()
Fs = F.float()
top2 = Fs.topk(2, dim=1).values
margin = (top2[:, 0] - top2[:, 1]).numpy()

rows = []
for j in range(k):
    mj = cid == j
    xs = flat[torch.as_tensor(mj)]
    sd = xs.std(0)
    snr = (comp[j].abs() / sd.clamp_min(1e-6)).mean().item()
    lv = level[mj]
    rows.append(dict(cluster=j, size_pct=mj.mean() * 100, oni_r=r['corr'][j], enso_fwer=j in enso,
                     level_mean=lv.mean(), level_sd=lv.std(), level_iqr=np.subtract(*np.percentile(lv, [75, 25])),
                     level_skew=stats.skew(lv), within_sd=sd.mean().item(), comp_amp=comp[j].abs().mean().item(),
                     snr=snr, pc_med=np.median(pc[mj]), pc_p10=np.percentile(pc[mj], 10),
                     pc_neg_pct=(pc[mj] < 0).mean() * 100, margin_med=np.median(margin[mj])))
tab = pd.DataFrame(rows).set_index('cluster').sort_values('level_mean')
tab.to_csv(f'{out}/per_cluster.csv')
order = tab.index.tolist()

# level-only separability: how much do neighbouring clusters overlap in level?
overlap = []
for a_, b_ in zip(order[:-1], order[1:]):
    la, lb = level[cid == a_], level[cid == b_]
    overlap.append(min(1.0, (np.minimum(np.histogram(la, 60, (-4, 4), density=True)[0],
                                        np.histogram(lb, 60, (-4, 4), density=True)[0]).sum() * 8 / 60)))
# silhouette in embedding space (cosine), subsample
from sklearn.metrics import silhouette_score
rng = np.random.default_rng(0); sub = rng.choice(n, min(6000, n), replace=False)
sil_emb = silhouette_score(torch.nn.functional.normalize(E[sub], dim=1).numpy(), cid[sub], metric='cosine')
sil_fld = silhouette_score(flat[sub].numpy(), cid[sub], metric='euclidean')

with open(f'{out}/summary.txt', 'w') as fh:
    s = (f'{label}: K={k}, train days={n}\n'
         f'ENSO-FWER clusters (|shift|>=24): {sorted(enso)}\n'
         f'silhouette: embedding (cosine) {sil_emb:.3f} | field space (euclid) {sil_fld:.3f}\n'
         f'median pattern corr with own composite: {np.median(pc):.2f}; days with corr<0: {(pc < 0).mean():.1%}\n'
         f'median top1-top2 cosine margin: {np.median(margin):.3f}; days with margin<0.02: {(margin < 0.02).mean():.1%}\n'
         f'level overlap between level-adjacent clusters: median {np.median(overlap):.2f} (0=disjoint, 1=identical)\n'
         f'mean composite SNR (|mean|/within-sd, averaged over cells): {tab.snr.mean():.2f} '
         f'(range {tab.snr.min():.2f}-{tab.snr.max():.2f})\n\n')
    fh.write(s + tab.round(3).to_string()); print(s + tab.round(2).to_string())

# --- figures ------------------------------------------------------------------ #
cols = ['#c0392b' if j in enso and r['corr'][j] > 0 else '#2471a3' if j in enso else '#7f8c8d' for j in order]

fig, ax = plt.subplots(figsize=(max(8, .45 * k), 4.2))
parts = ax.violinplot([level[cid == j] for j in order], showmedians=True, widths=.85)
for b, cc in zip(parts['bodies'], cols):
    b.set_facecolor(cc); b.set_alpha(.55)
ax.axhline(0, color='k', lw=.6)
ax.set_xticks(range(1, k + 1)); ax.set_xticklabels([f'C{j}' for j in order], rotation=90, fontsize=8)
ax.set_ylabel('daily domain-mean T anomaly (σ)')
ax.set_title(f'{label}: level distribution per cluster (red/blue = ENSO-linked +/−, FWER)')
plt.tight_layout(); plt.savefig(f'{out}/level_violins.png', dpi=130); plt.close()

nc = 6 if k <= 15 else 8; nr = int(np.ceil(k / nc))
fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.2 * nr), squeeze=False)
for ax, j, cc in zip(axes.flat, order, cols):
    lv = level[cid == j]
    (osm, osr), (sl, ic, _) = stats.probplot((lv - lv.mean()) / lv.std(), dist='norm')
    ax.plot(osm, osr, '.', ms=1.5, color=cc); ax.plot(osm, sl * osm + ic, 'k-', lw=.7)
    ax.set_title(f'C{j}  skew {stats.skew(lv):+.2f}', fontsize=8); ax.tick_params(labelsize=6)
for ax in list(axes.flat)[k:]:
    ax.axis('off')
fig.suptitle(f'{label}: QQ (normal) of members\' domain-mean anomaly, standardised within cluster', fontsize=10)
plt.tight_layout(); plt.savefig(f'{out}/qq_level.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(max(8, .45 * k), 4))
bp = ax.boxplot([pc[cid == j] for j in order], whis=(5, 95), showfliers=False, patch_artist=True)
for b, cc in zip(bp['boxes'], cols):
    b.set_facecolor(cc); b.set_alpha(.55)
ax.axhline(0, color='k', lw=.6)
ax.set_xticks(range(1, k + 1)); ax.set_xticklabels([f'C{j}' for j in order], rotation=90, fontsize=8)
ax.set_ylabel('pattern corr. with own composite'); ax.set_ylim(-1, 1)
ax.set_title(f'{label}: how much each day resembles its cluster composite (whiskers 5–95%)')
plt.tight_layout(); plt.savefig(f'{out}/pattern_corr.png', dpi=130); plt.close()

# composite / within-sd / SNR maps for the ENSO-linked clusters (+ coolest & warmest)
show = list(dict.fromkeys([order[0], order[-1]] + [j for j in order if j in enso]))
fig, axes = plt.subplots(3, len(show), figsize=(2.6 * len(show), 6.4), squeeze=False)
for col, j in enumerate(show):
    xs = Tm[torch.as_tensor(cid == j)]
    mu, sd = xs.mean(0), xs.std(0)
    for row, (img, cm, vr, ttl) in enumerate([(mu, 'RdBu_r', (-1.5, 1.5), 'mean'),
                                                (sd, 'viridis', (0, 1.2), 'within-sd'),
                                                (mu.abs() / sd, 'magma', (0, 2), '|mean|/sd')]):
        im = axes[row, col].imshow(img, cmap=cm, vmin=vr[0], vmax=vr[1], extent=[-74, -44, -18, 5.5])
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if col == 0:
            axes[row, col].set_ylabel(ttl)
        if row == 0:
            axes[row, col].set_title(f'C{j} r={r["corr"][j]:+.2f}', fontsize=9)
        # the inference crop: only this box decides the assignment
        axes[row, col].add_patch(plt.Rectangle((-69.3, -16.6), 20.6, 20.7, fill=False, ec='w', lw=.8, ls='--'))
fig.suptitle(f'{label}: composite, within-cluster spread and signal-to-noise (dashed = inference crop)', fontsize=10)
plt.tight_layout(); plt.savefig(f'{out}/snr_maps.png', dpi=120); plt.close()

# random member samples for the ENSO-linked clusters
smp = [j for j in order if j in enso][:4] or order[-2:]
fig, axes = plt.subplots(len(smp), 9, figsize=(17, 2.0 * len(smp)), squeeze=False)
for row, j in enumerate(smp):
    idx = rng.choice(np.where(cid == j)[0], 8, replace=False)
    axes[row, 0].imshow(Tm[torch.as_tensor(cid == j)].mean(0), cmap='RdBu_r', vmin=-2, vmax=2)
    axes[row, 0].set_title(f'C{j} composite', fontsize=8)
    for c_, i in enumerate(idx, 1):
        axes[row, c_].imshow(Tm[i], cmap='RdBu_r', vmin=-2.5, vmax=2.5)
        axes[row, c_].set_title(f'{t_tr[i].date()} pc={pc[i]:+.2f}', fontsize=7)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'{label}: random members of ENSO-linked clusters (mean T anomaly, σ)', fontsize=10)
plt.tight_layout(); plt.savefig(f'{out}/random_samples.png', dpi=110); plt.close()
print('figures in', out)
