"""Significance-aware ENSO analysis helpers.

Everything is built on the MONTHLY cluster-frequency series so that inference
respects the real unit of replication (months / ENSO episodes), not individual
days. Significance comes from an autocorrelation-preserving null (circular
rotation of the ONI series) and family-wise error is controlled with a
max-statistic permutation test. Confidence intervals use a block (year)
bootstrap. ONI is used as a continuous, signed predictor so La Nina information
and event magnitude both contribute.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def build_monthly(df, oni_index, n_proto):
    """From a daily frame (needs 'cluster_id','date') + ONI table, build a
    monthly table: one row per calendar month with cluster frequencies f0..fK,
    the ONI value, ENSO Label, year, month and an ENSO-episode id.
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d['mp'] = d['date'].dt.to_period('M')
    # monthly cluster frequency (rows sum to 1)
    freq = (pd.crosstab(d['mp'], d['cluster_id'], normalize='index')
            .reindex(columns=range(n_proto), fill_value=0.0))
    freq.columns = [f'f{k}' for k in range(n_proto)]
    oni = oni_index.copy()
    oni.index = oni.index if oni.index.name == 'date_period' else oni.index
    m = freq.join(oni[['ONI', 'Label']], how='inner').dropna(subset=['ONI'])
    m = m.sort_index()
    m['year'] = m.index.year
    m['month'] = m.index.month
    # episode id: contiguous runs of identical Label
    lab = m['Label'].values
    epi = np.zeros(len(lab), int)
    for i in range(1, len(lab)):
        epi[i] = epi[i - 1] + (lab[i] != lab[i - 1])
    m['episode'] = epi
    return m


def _freq_matrix(monthly, n_proto):
    return monthly[[f'f{k}' for k in range(n_proto)]].values  # (T, K)


def _corr(a, b):
    """Column-wise Pearson corr between matrix a (T,K) and vector b (T,)."""
    a = a - a.mean(0); b = b - b.mean()
    num = a.T @ b
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum()) + 1e-12
    return num / den


# --------------------------------------------------------------------------- #
# Core effect: continuous-ONI response per cluster  (corr of monthly freq ~ ONI)
# --------------------------------------------------------------------------- #
def oni_response(monthly, n_proto, n_perm=5000, seed=0):
    """Per-cluster correlation between monthly cluster frequency and ONI, with:
      - raw p  (circular-rotation null, preserves autocorrelation)
      - fwer p (max-|corr| across clusters per rotation -> controls 15 tests)
    Returns a DataFrame indexed by cluster.
    """
    rng = np.random.default_rng(seed)
    F = _freq_matrix(monthly, n_proto)
    o = monthly['ONI'].values.astype(float)
    obs = _corr(F, o)                              # (K,)
    T = len(o)
    null = np.empty((n_perm, n_proto))
    for i in range(n_perm):
        shift = rng.integers(1, T)
        null[i] = _corr(F, np.roll(o, shift))
    raw_p = (np.abs(null) >= np.abs(obs)).mean(0)
    maxnull = np.abs(null).max(1)                  # (n_perm,) family-wise
    fwer_p = (maxnull[:, None] >= np.abs(obs)).mean(0)
    return pd.DataFrame({'corr': obs, 'p_raw': raw_p, 'p_fwer': fwer_p},
                        index=range(n_proto)).rename_axis('cluster')


def omnibus_test(monthly, n_proto, n_perm=5000, seed=0):
    """Single global test: does ONI relate to cluster occupancy at all?
    Statistic = sum_k corr(f_k, ONI)^2. Circular-rotation null.
    """
    rng = np.random.default_rng(seed)
    F = _freq_matrix(monthly, n_proto)
    o = monthly['ONI'].values.astype(float)
    stat = (_corr(F, o) ** 2).sum()
    T = len(o)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = (_corr(F, np.roll(o, rng.integers(1, T))) ** 2).sum()
    p = (null >= stat).mean()
    return stat, p


# --------------------------------------------------------------------------- #
# Block (year) bootstrap CIs on a per-cluster statistic
# --------------------------------------------------------------------------- #
def bootstrap_ci_oni_corr(monthly, n_proto, n_boot=2000, seed=0, alpha=0.05):
    """Resample whole YEARS with replacement (blocks) -> CI on corr(f_k, ONI)."""
    rng = np.random.default_rng(seed)
    years = monthly['year'].unique()
    by_year = {y: monthly[monthly['year'] == y] for y in years}
    boot = np.empty((n_boot, n_proto))
    for b in range(n_boot):
        pick = rng.choice(years, len(years), replace=True)
        sample = pd.concat([by_year[y] for y in pick])
        F = _freq_matrix(sample, n_proto)
        boot[b] = _corr(F, sample['ONI'].values.astype(float))
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# --------------------------------------------------------------------------- #
# El Nino occurrence anomaly with honest null + bootstrap CI (same plot as the
# original notebook, but with error bands and autocorrelation-aware p-values)
# --------------------------------------------------------------------------- #
def elnino_anomaly(monthly, n_proto, n_perm=5000, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    F = _freq_matrix(monthly, n_proto)
    is_el = (monthly['Label'] == 'El Niño').values
    base = F.mean(0)                                   # P(k) climatology
    obs = F[is_el].mean(0) - base                      # anomaly per cluster (fraction)
    # circular-rotation null on the El Nino indicator (preserves ENSO blocks)
    T = len(is_el)
    null = np.empty((n_perm, n_proto))
    for i in range(n_perm):
        idx = np.roll(is_el, rng.integers(1, T))
        null[i] = F[idx].mean(0) - base
    raw_p = (np.abs(null) >= np.abs(obs)).mean(0)
    fwer_p = (np.abs(null).max(1)[:, None] >= np.abs(obs)).mean(0)
    # year-block bootstrap CI
    years = monthly['year'].unique()
    by_year = {y: monthly[monthly['year'] == y] for y in years}
    boot = np.empty((n_boot, n_proto))
    for b in range(n_boot):
        s = pd.concat([by_year[y] for y in rng.choice(years, len(years), replace=True)])
        Fb = _freq_matrix(s, n_proto); el = (s['Label'] == 'El Niño').values
        boot[b] = Fb[el].mean(0) - Fb.mean(0)
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return pd.DataFrame({'anom': obs * 100, 'ci_lo': lo * 100, 'ci_hi': hi * 100,
                         'p_raw': raw_p, 'p_fwer': fwer_p}, index=range(n_proto)).rename_axis('cluster')


# --------------------------------------------------------------------------- #
# Lagged cross-correlation (replaces the per-cell heatmaps)
# --------------------------------------------------------------------------- #
def lag_crosscorr(monthly, n_proto, max_lag=12, n_perm=5000, seed=0):
    """corr(f_k(t), ONI(t-lag)) for lag in [-max_lag, max_lag].
    Significance per (cluster,lag) via circular rotation, with a single
    max-|corr| null over ALL clusters and lags (family-wise across the map).
    Returns (lags, corr[K,L], p_fwer[K,L]).
    """
    rng = np.random.default_rng(seed)
    lags = np.arange(-max_lag, max_lag + 1)
    F = _freq_matrix(monthly, n_proto)
    o = monthly['ONI'].values.astype(float)
    T = len(o)

    def cc(ovec):
        out = np.empty((n_proto, len(lags)))
        for j, L in enumerate(lags):
            out[:, j] = _corr(F, np.roll(ovec, L))
        return out

    obs = cc(o)
    maxnull = np.empty(n_perm)
    for i in range(n_perm):
        maxnull[i] = np.abs(cc(np.roll(o, rng.integers(1, T)))).max()
    p_fwer = (maxnull[None, None, :] >= np.abs(obs)[..., None]).mean(2)
    return lags, obs, p_fwer


# --------------------------------------------------------------------------- #
# Temporal split-half robustness
# --------------------------------------------------------------------------- #
def split_half(monthly, n_proto):
    yrs = np.sort(monthly['year'].unique())
    mid = yrs[len(yrs) // 2]
    a = monthly[monthly['year'] < mid]; b = monthly[monthly['year'] >= mid]
    ca = _corr(_freq_matrix(a, n_proto), a['ONI'].values.astype(float))
    cb = _corr(_freq_matrix(b, n_proto), b['ONI'].values.astype(float))
    r = np.corrcoef(ca, cb)[0, 1]
    return pd.DataFrame({'corr_firsthalf': ca, 'corr_secondhalf': cb},
                        index=range(n_proto)).rename_axis('cluster'), r, int(mid)
