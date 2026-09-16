"""Generate publication figures + stats.json for the Nature-style paper.

Mirrors notebooks/analysis_significance.ipynb (same seeds/N_PERM/N_BOOT, so the
numbers are identical), but renders multi-panel figures sized for a
Springer-Nature article. Run from the repo root:

    .venv/bin/python paper/make_figures.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO)
import src.inference as inf                              # noqa: E402
from src.el_nino import read_enso_data                   # noqa: E402
import src.enso_stats as es                              # noqa: E402
from src.data_manager import Era5Dataset                 # noqa: E402

CONFIG = 'checkpoint/temperature-era-amazonia-15-anom/params-temperature-era-amazonia-15-anom.yaml'
SAVE = 'results/temperature-era-amazonia-15-anom'
OUT = 'paper/figures'
N_PERM, N_BOOT = 5000, 2000

# ---------------------------------------------------------------- style ----
mpl.rcParams.update({
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
# significance encoding: one hue family, monotone lightness (CVD-safe), plus
# text markers so colour is never the only channel
C_FWER, C_RAW, C_NS = '#99000d', '#fb6a4a', '#c6c6c6'
C_POS, C_NEG, C_NEU = '#b2182b', '#2166ac', '#bdbdbd'   # RdBu poles + neutral
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM                               # Nature col widths


def sig_color(p_fwer, p_raw):
    return [C_FWER if pf < .05 else (C_RAW if pr < .05 else C_NS)
            for pf, pr in zip(p_fwer, p_raw)]


def sig_mark(ax, x, y, p_fwer, p_raw, dy):
    for xi, yi, pf, pr in zip(x, y, p_fwer, p_raw):
        m = '**' if pf < .05 else ('*' if pr < .05 else '')
        if m:
            ax.text(xi, yi + dy * (1 if yi >= 0 else -1), m, ha='center',
                    va='bottom' if yi >= 0 else 'top', fontsize=7)


def panel_label(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='right')


# ----------------------------------------------------------------- data ----
print('loading data …')
params, dataset = inf.read_data(CONFIG, validation=True)
N = params['criterion']['num_proto']
F = inf.get_model_results(read_path=SAVE, params=params, dataset=dataset)[1]
F_v = inf.get_model_results(read_path=SAVE, params=params,
                            dataset=dataset.validation_imgs, validation=True)[1]

cid = torch.cat([F.argmax(1).cpu(), F_v.argmax(1).cpu()]).numpy()
dates = pd.to_datetime(list(dataset.time) + list(dataset.val_time))
df = pd.DataFrame({'cluster_id': cid, 'date': dates})

oni = read_enso_data(); oni.index.name = 'date_period'
monthly = es.build_monthly(df, oni, N)

imgs = dataset.imgs
cid_tr = F.argmax(1).cpu().numpy()
vars_names = params['data']['surf_vars']
VAR_LABEL = {'mn2t': 'T$_{min}$', 'mx2t': 'T$_{max}$', 'tp': 'Precip.'}
V, H, W = imgs.shape[1:]
lat_lim = params['data']['lat_limit']; lon_lim = params['data']['lon_limit']
EXTENT = [min(lon_lim), max(lon_lim), min(lat_lim), max(lat_lim)]

df_day = df.sort_values('date').reset_index(drop=True)
seq = df_day['cluster_id'].values
gap = (df_day['date'].diff().dt.days.values == 1)

tr = pd.DataFrame({'cluster_id': cid_tr, 'date': pd.to_datetime(dataset.time)})
tr['mp'] = tr['date'].dt.to_period('M')
tr = tr.merge(oni[['Label', 'ONI']], left_on='mp', right_index=True, how='left')


def composite(mask):
    return imgs[np.asarray(mask, bool)].mean(0).numpy()


dommean = np.array([imgs[cid_tr == k].mean().item() for k in range(N)])
freq_pct = np.array([(cid_tr == k).mean() * 100 for k in range(N)])
order = list(np.argsort(dommean))

# regime stats (durations, trends)
runs = {k: [] for k in range(N)}; cur, L = seq[0], 1
for i in range(1, len(seq)):
    if seq[i] == cur and gap[i]:
        L += 1
    else:
        runs[cur].append(L); cur, L = seq[i], 1
runs[cur].append(L)
dur = {k: float(np.mean(runs[k])) for k in range(N)}

yf = monthly.groupby('year')[[f'f{k}' for k in range(N)]].mean() * 100
yrs = yf.index.values.astype(float)
trend = {k: float(np.polyfit(yrs, yf[f'f{k}'].values, 1)[0] * 10) for k in range(N)}

mfreq = monthly.groupby('month')[[f'f{k}' for k in range(N)]].mean() * 100
seas_span = float((mfreq.max() - mfreq.min()).mean())

# ------------------------------------------------------------ statistics ----
print('running significance machinery …')
omni_stat, omni_p = es.omnibus_test(monthly, N, n_perm=N_PERM)
resp = es.oni_response(monthly, N, n_perm=N_PERM)
lo, hi = es.bootstrap_ci_oni_corr(monthly, N, n_boot=N_BOOT)
resp['ci_lo'], resp['ci_hi'] = lo, hi
an = es.elnino_anomaly(monthly, N, n_perm=N_PERM, n_boot=N_BOOT)
lags, cc, pf = es.lag_crosscorr(monthly, N, max_lag=12, n_perm=N_PERM)
sh, split_r, split_mid = es.split_half(monthly, N)
robust = [k for k in range(N) if resp.p_fwer[k] < .05]

# ENSO fingerprint + similarity
el_map = composite(tr.Label.values == 'El Niño')
ln_map = composite(tr.Label.values == 'La Niña')


def spatial_corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


sim = np.array([spatial_corr(composite(cid_tr == k), el_map) for k in range(N)])
phase_order = ['La Niña', 'Neutro', 'El Niño']
phase_freq = (monthly.groupby('Label')[[f'f{k}' for k in range(N)]].mean() * 100
              ).reindex(phase_order)

# raw-temperature sanity check
raw_ds = Era5Dataset(transform=None, surf_vars=vars_names, lat_lim=lat_lim,
                     lon_lim=lon_lim, split_val=True,
                     val_years=params['data']['val_years'],
                     deseason_standardize=False)
dm_raw = raw_ds.imgs.mean((2, 3)).numpy()
month_tr = tr['date'].dt.month.values
clim_m = np.stack([dm_raw[month_tr == m].mean(0) for m in range(1, 13)])


def month_departure(k):
    d = np.full((12, V), np.nan)
    for m in range(1, 13):
        s = (month_tr == m) & (cid_tr == k)
        if s.sum():
            d[m - 1] = dm_raw[s].mean(0) - clim_m[m - 1]
    return d


# =================================================================== FIG 1 ==
print('fig 1 …')
vabs = np.percentile(np.abs(np.stack([composite(cid_tr == k) for k in range(N)])), 99)
fig, axes = plt.subplots(3, 5, figsize=(W2, 105 * MM))
for ax, k in zip(axes.flat, order):
    m = composite(cid_tr == k).mean(0)
    im = ax.imshow(m, cmap='RdBu_r', vmin=-vabs, vmax=vabs, extent=EXTENT,
                   origin='upper', aspect='auto')
    star = r' $\bf{*}$' if k in robust else ''
    ax.set_title(f'C{k}{star}   {freq_pct[k]:.1f}%', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
fig.subplots_adjust(left=.01, right=.90, top=.94, bottom=.02, wspace=.06, hspace=.18)
cax = fig.add_axes([.92, .15, .015, .65])
cb = fig.colorbar(im, cax=cax)
cb.set_label('temperature anomaly (s.d.)', fontsize=7)
fig.savefig(f'{OUT}/fig1_regimes.pdf'); fig.savefig(f'{OUT}/fig1_regimes.png')
plt.close(fig)

# =================================================================== FIG 2 ==
print('fig 2 …')
fig = plt.figure(figsize=(W2, 95 * MM))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1, 1.1], hspace=.55, wspace=.45,
              left=.07, right=.98, top=.92, bottom=.10)

axa = fig.add_subplot(gs[:, 0])
im = axa.imshow(mfreq.T.values, aspect='auto', cmap='Blues',
                extent=[.5, 12.5, N - .5, -.5])
axa.set_yticks(range(N)); axa.set_yticklabels([f'C{k}' for k in range(N)])
axa.set_xticks(range(1, 13)); axa.set_xlabel('calendar month'); axa.set_ylabel('regime')
axa.set_title('Monthly occurrence frequency')
cb = fig.colorbar(im, ax=axa, shrink=.8, pad=.02); cb.set_label('%', fontsize=6.5)
panel_label(axa, 'a')

axb = fig.add_subplot(gs[:, 1])
ts = sorted(range(N), key=lambda k: trend[k])
axb.barh(range(N), [trend[k] for k in ts],
         color=[C_NEG if trend[k] < 0 else C_POS for k in ts], height=.72)
axb.set_yticks(range(N)); axb.set_yticklabels([f'C{k}' for k in ts])
axb.axvline(0, color='k', lw=.6)
axb.set_xlabel('frequency trend (% per decade)')
axb.set_title('Long-term trend, 1981–2025')
panel_label(axb, 'b')

axc = fig.add_subplot(gs[0, 2])
axc.bar(range(N), [dur[k] for k in range(N)], color='#4d4d4d', width=.72)
axc.set_xticks(range(N))
axc.set_xticklabels([f'{k}' for k in range(N)], fontsize=5.5)
axc.set_ylabel('mean duration (days)'); axc.set_xlabel('regime')
axc.set_title('Persistence')
panel_label(axc, 'c')

axd = fig.add_subplot(gs[1, 2])
sanity = [order[-1], order[-2], order[1], order[0]]
x = np.arange(len(sanity)); w = .34
dep = np.array([[np.nanmean(month_departure(k)[:, v]) for v in range(V)]
                for k in sanity])
axd.bar(x - w / 2, dep[:, 0], w, label=VAR_LABEL[vars_names[0]], color='#7f7f7f')
axd.bar(x + w / 2, dep[:, 1], w, label=VAR_LABEL[vars_names[1]], color='#252525')
axd.axhline(0, color='k', lw=.6)
axd.set_xticks(x); axd.set_xticklabels([f'C{k}' for k in sanity])
axd.set_ylabel('departure from monthly\nclimatology (K)')
axd.set_title('Raw-temperature check')
axd.legend(frameon=False, ncol=2, loc='upper right', borderpad=0)
panel_label(axd, 'd')
fig.savefig(f'{OUT}/fig2_characterization.pdf')
fig.savefig(f'{OUT}/fig2_characterization.png')
plt.close(fig)

# =================================================================== FIG 3 ==
print('fig 3 …')
fig = plt.figure(figsize=(W2, 110 * MM))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1], hspace=.5, wspace=.3,
              left=.07, right=.98, top=.93, bottom=.09)

axa = fig.add_subplot(gs[0, 0])
c = sig_color(resp.p_fwer.values, resp.p_raw.values)
axa.bar(resp.index, resp['corr'], color=c, width=.72)
axa.errorbar(resp.index, resp['corr'],
             yerr=[resp['corr'] - resp.ci_lo, resp.ci_hi - resp['corr']],
             fmt='none', ecolor='k', capsize=1.5, lw=.7)
sig_mark(axa, resp.index, np.where(resp['corr'] >= 0, resp.ci_hi, resp.ci_lo),
         resp.p_fwer.values, resp.p_raw.values, .015)
axa.axhline(0, color='k', lw=.6)
axa.set_xticks(range(N)); axa.set_xlabel('regime')
axa.set_ylabel('corr(monthly frequency, ONI)')
axa.set_title('Continuous-ONI response')
panel_label(axa, 'a')

axb = fig.add_subplot(gs[0, 1])
axb.bar(an.index, an['anom'], color=sig_color(an.p_fwer.values, an.p_raw.values),
        width=.72)
axb.errorbar(an.index, an['anom'], yerr=[an['anom'] - an.ci_lo, an.ci_hi - an['anom']],
             fmt='none', ecolor='k', capsize=1.5, lw=.7)
axb.axhline(0, color='k', lw=.6)
axb.set_xticks(range(N)); axb.set_xlabel('regime')
axb.set_ylabel(r'$P(k\,|\,$El Niño$) - P(k)$  (pp)')
axb.set_title('El Niño occurrence anomaly')
panel_label(axb, 'b')

axc = fig.add_subplot(gs[1, 0])
vm = np.abs(cc).max()
im = axc.imshow(cc, aspect='auto', cmap='RdBu_r', vmin=-vm, vmax=vm,
                extent=[lags[0] - .5, lags[-1] + .5, N - .5, -.5])
for k in range(N):
    for j, Lg in enumerate(lags):
        if pf[k, j] < .05:
            axc.add_patch(plt.Rectangle((Lg - .5, k - .5), 1, 1, fill=False,
                                        edgecolor='k', lw=1.2))
axc.set_xlabel('lag (months; ONI leads for lag > 0)')
axc.set_yticks(range(N)); axc.set_yticklabels([f'C{k}' for k in range(N)])
axc.set_ylabel('regime')
axc.set_title('Lagged cross-correlation with ONI')
cb = fig.colorbar(im, ax=axc, shrink=.85, pad=.02); cb.set_label('r', fontsize=6.5)
panel_label(axc, 'c')

axd = fig.add_subplot(gs[1, 1])
axd.axhline(0, color='k', lw=.5); axd.axvline(0, color='k', lw=.5)
lim = np.abs(sh.values).max() * 1.2
axd.plot([-lim, lim], [-lim, lim], '--', color='grey', lw=.7)
cols = sig_color(resp.p_fwer.values, resp.p_raw.values)
for k in range(N):
    axd.scatter(sh.corr_firsthalf[k], sh.corr_secondhalf[k], color=cols[k],
                s=26, zorder=3, edgecolor='k', linewidth=.3)
    axd.annotate(str(k), (sh.corr_firsthalf[k], sh.corr_secondhalf[k]),
                 fontsize=5, ha='center', va='center', zorder=4)
axd.set_xlim(-lim, lim); axd.set_ylim(-lim, lim)
axd.set_xlabel(f'ONI corr., 1981–{split_mid - 1}')
axd.set_ylabel(f'ONI corr., {split_mid}–2025')
axd.set_title(f'Split-half stability (r = {split_r:.2f})')
panel_label(axd, 'd')
fig.savefig(f'{OUT}/fig3_enso_stats.pdf'); fig.savefig(f'{OUT}/fig3_enso_stats.png')
plt.close(fig)

# =================================================================== FIG 4 ==
print('fig 4 …')
fig = plt.figure(figsize=(W2, 125 * MM))
gs_top = GridSpec(2, 4, figure=fig, hspace=.18, wspace=.12,
                  left=.075, right=.90, top=.93, bottom=.42)
gs_bot = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=.35,
                  left=.075, right=.98, top=.30, bottom=.09)

map_sets = [(f'C{k}', composite(cid_tr == k)) for k in robust]
fp_sets = [('El Niño', el_map), ('La Niña', ln_map)]
vmax4 = np.percentile(np.abs(np.stack([m for _, m in map_sets + fp_sets])), 99)

# rows 0-1: robust regime composites (cols 0-2) + fingerprint (col 3)
for j, (name, m) in enumerate(map_sets):
    for v in range(V):
        ax = fig.add_subplot(gs_top[v, j])
        im = ax.imshow(m[v], cmap='RdBu_r', vmin=-vmax4, vmax=vmax4,
                       extent=EXTENT, origin='upper', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5)
        if v == 0:
            k = robust[j]
            ax.set_title(f'{name}  (r = {resp["corr"][k]:+.2f})', fontsize=7)
        if j == 0:
            ax.set_ylabel(VAR_LABEL[vars_names[v]], fontsize=7)
        if j == 0 and v == 0:
            panel_label(ax, 'a', dx=-.18)
for v in range(V):
    ax = fig.add_subplot(gs_top[v, 3])
    m = el_map - ln_map
    ax.imshow(m[v], cmap='RdBu_r', vmin=-vmax4, vmax=vmax4,
              extent=EXTENT, origin='upper', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    if v == 0:
        ax.set_title('El Niño − La Niña', fontsize=7)
        panel_label(ax, 'b', dx=-.08)
cax = fig.add_axes([.925, .47, .013, .41])
cb = fig.colorbar(im, cax=cax)
cb.set_label('temperature anomaly (s.d.)', fontsize=6.5)

# bottom row: similarity bars + phase occurrence
axc = fig.add_subplot(gs_bot[0, 0])
axc.bar(range(N), sim, color=sig_color(resp.p_fwer.values, resp.p_raw.values),
        width=.72)
axc.axhline(0, color='k', lw=.6)
axc.set_xticks(range(N)); axc.set_xlabel('regime', labelpad=1)
axc.set_ylabel('spatial corr. with\nEl Niño composite')
panel_label(axc, 'c', dx=-.14)

axd = fig.add_subplot(gs_bot[0, 1])
x = np.arange(len(robust)); w = .26
for j, (ph, col, lab) in enumerate(zip(phase_order, [C_NEG, C_NEU, C_POS],
                                       ['La Niña', 'Neutral', 'El Niño'])):
    axd.bar(x + (j - 1) * w, [phase_freq.loc[ph, f'f{k}'] for k in robust], w,
            label=lab, color=col)
axd.set_xticks(x); axd.set_xticklabels([f'C{k}' for k in robust])
axd.set_xlabel('robust regime', labelpad=1)
axd.set_ylabel('mean monthly\nfrequency (%)')
axd.legend(frameon=False, ncol=3, loc='upper left', columnspacing=.9,
           handlelength=1.2)
panel_label(axd, 'd', dx=-.18)
fig.savefig(f'{OUT}/fig4_physical.pdf'); fig.savefig(f'{OUT}/fig4_physical.png')
plt.close(fig)

# ============================================================ ED FIGURES ==
print('extended data …')
# ED1: transition matrix
Tm = np.zeros((N, N))
for i in range(1, len(seq)):
    if gap[i]:
        Tm[seq[i - 1], seq[i]] += 1
Tn = Tm / Tm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(W1, 75 * MM))
im = ax.imshow(Tn, cmap='Blues', vmin=0, vmax=np.percentile(Tn, 98))
ax.set_xlabel('to regime'); ax.set_ylabel('from regime')
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_title('Day-to-day transition probability')
fig.colorbar(im, label='P(to | from)', shrink=.8)
fig.tight_layout()
fig.savefig(f'{OUT}/ed1_transitions.pdf'); fig.savefig(f'{OUT}/ed1_transitions.png')
plt.close(fig)

# ED2: rolling frequency vs ONI
mon = monthly.copy(); mon['t'] = mon.index.to_timestamp()
fig, axes = plt.subplots(len(robust), 1, figsize=(W2, 40 * MM * len(robust)),
                         sharex=True, squeeze=False)
for i, k in enumerate(robust):
    ax = axes[i][0]
    fr = mon[f'f{k}'].rolling(12, center=True).mean() * 100
    ax.fill_between(mon['t'], 0, 1, where=(mon['Label'] == 'El Niño').values,
                    color=C_POS, alpha=.12, transform=ax.get_xaxis_transform())
    ax.fill_between(mon['t'], 0, 1, where=(mon['Label'] == 'La Niña').values,
                    color=C_NEG, alpha=.12, transform=ax.get_xaxis_transform())
    ax.plot(mon['t'], fr, color='#1a1a1a', lw=1.2)
    ax.set_ylabel(f'C{k} frequency (%)')
    ax.set_title(f'Regime C{k}  (ONI corr. {resp["corr"][k]:+.2f})', fontsize=7)
axes[-1][0].set_xlabel('year')
fig.tight_layout()
fig.savefig(f'{OUT}/ed2_timeseries.pdf'); fig.savefig(f'{OUT}/ed2_timeseries.png')
plt.close(fig)

# ED3: monthly raw-temperature departures for the four extreme regimes
fig, axes = plt.subplots(len(sanity), V, figsize=(W2, 26 * MM * len(sanity)),
                         sharex=True, squeeze=False)
for i, k in enumerate(sanity):
    d = month_departure(k)
    for v in range(V):
        ax = axes[i][v]
        ax.bar(range(1, 13), d[:, v],
               color=[C_POS if xx > 0 else C_NEG for xx in d[:, v]], width=.7)
        ax.axhline(0, color='k', lw=.6)
        if i == 0:
            ax.set_title(VAR_LABEL[vars_names[v]])
        if v == 0:
            ax.set_ylabel(f'C{k}\n(K)')
for v in range(V):
    axes[-1][v].set_xlabel('calendar month')
    axes[-1][v].set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(f'{OUT}/ed3_sanity.pdf'); fig.savefig(f'{OUT}/ed3_sanity.png')
plt.close(fig)

# ================================================================= STATS ==
consistency = {}
for k in sanity:
    d = month_departure(k)
    consistency[int(k)] = {
        vars_names[v]: {
            'mean_K': float(np.nanmean(d[:, v])),
            'sign_consistency_%': float(
                np.nanmean(np.sign(d[:, v]) == np.sign(np.nanmean(d[:, v]))) * 100)}
        for v in range(V)}

stats = {
    'months': int(len(monthly)), 'years': int(monthly.year.nunique()),
    'episodes': int(monthly.episode.nunique()),
    'oni_range': [float(monthly.ONI.min()), float(monthly.ONI.max())],
    'elnino_months': int((monthly.Label == 'El Niño').sum()),
    'lanina_months': int((monthly.Label == 'La Niña').sum()),
    'neutral_months': int((monthly.Label == 'Neutro').sum()),
    'n_proto': int(N), 'grid_HW': [int(H), int(W)],
    'freq_range_pct': [float(freq_pct.min()), float(freq_pct.max())],
    'duration_range_d': [float(min(dur.values())), float(max(dur.values()))],
    'seasonal_span_pct': seas_span,
    'trend_pct_dec': {int(k): round(trend[k], 2) for k in range(N)},
    'omnibus': {'stat': float(omni_stat), 'p': float(omni_p)},
    'resp': resp.round(4).to_dict(orient='index'),
    'n_raw_sig': int((resp.p_raw < .05).sum()),
    'n_fwer_sig': int((resp.p_fwer < .05).sum()),
    'robust': [int(k) for k in robust],
    'elnino_anomaly': an.round(3).to_dict(orient='index'),
    'lag_cells_sig': int((pf < .05).sum()), 'lag_cells_total': int(pf.size),
    'split_half_r': float(split_r), 'split_mid': int(split_mid),
    'spatial_sim': {int(k): round(float(sim[k]), 3) for k in range(N)},
    'phase_freq': phase_freq.round(2).to_dict(orient='index'),
    'raw_temp_check': consistency,
}
with open('paper/stats.json', 'w') as f:
    json.dump(stats, f, indent=1, default=str)
print('done — figures in', OUT, '+ paper/stats.json')
