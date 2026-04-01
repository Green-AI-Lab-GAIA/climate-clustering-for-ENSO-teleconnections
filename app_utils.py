"""
Shared utilities for the Streamlit climate analysis app.
All heavy data is loaded once via @st.cache_resource.
Plot functions return matplotlib Figure objects.
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make sure the project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.el_nino import read_enso_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DATA = os.path.join(PROJECT_ROOT, "app_data")

PALETTE30 = list(sns.color_palette("tab20", 20)) + list(sns.color_palette("bright", 10))
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def _to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

PALETTE30_HEX = [_to_hex(c) for c in PALETTE30]

# Dash styles cycling per variable (Plotly dash names)
_VAR_DASH = ["solid", "dash", "dot", "dashdot"]

_ENSO_COLS = ["Label", "ONI", "label_color"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_season(date):
    m, d = date.month, date.day
    if (m == 12 and d >= 21) or m in (1, 2) or (m == 3 and d <= 20):
        return "Summer"
    elif (m == 3 and d >= 21) or m in (4, 5) or (m == 6 and d <= 20):
        return "Autumn"
    elif (m == 6 and d >= 21) or m in (7, 8) or (m == 9 and d <= 22):
        return "Winter"
    else:
        return "Spring"


def compute_seasonal_groups(df):
    df = df.copy()
    df["season"] = df["date"].apply(get_season)
    seasonal_dist = pd.crosstab(df["cluster_id"], df["season"], normalize="index")
    seasonal_dist["dominant_season"] = seasonal_dist.idxmax(axis=1)
    grupos = (
        seasonal_dist.reset_index()
        .groupby("dominant_season")
        .apply(lambda x: x["cluster_id"].tolist(), include_groups=False)
        .to_dict()
    )
    return df, grupos


def compute_anomaly(data, comparison_mode):
    """Return per-cluster probability anomaly (%) Series."""
    prob = pd.crosstab(data["Label"], data["cluster_id"], normalize="index") * 100
    nino_label = next((l for l in ["El Niño", "El Nino"] if l in prob.index), None)
    if nino_label is None:
        raise ValueError("No El Niño rows found")
    p_nino = prob.loc[nino_label]
    if comparison_mode == "climatology":
        baseline = data["cluster_id"].value_counts(normalize=True) * 100
        baseline = baseline.reindex(p_nino.index, fill_value=0)
    else:
        neutro_label = next((l for l in ["Neutro", "Neutral"] if l in prob.index), None)
        if neutro_label is None:
            baseline = pd.Series(0.0, index=p_nino.index)
        else:
            baseline = prob.loc[neutro_label]
    return p_nino - baseline


def _build_lagged_df(df_base, oni_index, lag, year_start=None, year_end=None):
    """Re-merge df with a lagged ONI index; adds 'month' column."""
    df = df_base.drop(columns=[c for c in _ENSO_COLS if c in df_base.columns], errors="ignore").copy()
    if year_start:
        df = df[df["date"].dt.year >= year_start]
    if year_end:
        df = df[df["date"].dt.year <= year_end]
    df["month"] = df["date"].dt.month
    return df.merge(oni_index.shift(lag).dropna(), left_on="date_period", right_index=True, how="left")


def _make_pivot(climate_anon, row_name, col_name="Lag"):
    anom_df = pd.DataFrame.from_dict(climate_anon, orient="index", columns=["Anomaly"])
    anom_df.index = pd.MultiIndex.from_tuples(anom_df.index, names=[row_name, col_name])
    return anom_df.reset_index().pivot(index=row_name, columns=col_name, values="Anomaly")


def get_lagged_anomaly(df_base, oni_index, comparison_mode="neutral",
                       year_start=None, year_end=None):
    """Anomaly (%) DataFrame: rows=clusters, cols=lags."""
    anomalies = {}
    for lag in range(-12, 13):
        merged = _build_lagged_df(df_base, oni_index, lag, year_start, year_end)
        try:
            anomalies[lag] = compute_anomaly(merged, comparison_mode)
        except (ValueError, KeyError):
            continue
    anom = pd.DataFrame(anomalies).sort_index(axis=1)
    anom.columns.name = "Lag"
    return anom


# ---------------------------------------------------------------------------
# Data loading (cached once per server session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading data…")
def load_app_data():
    """Load pre-computed app_data/ produced by scripts/precompute_app_data.py.

    No NetCDF files, no model weights — just small pre-computed artefacts.
    Run scripts/precompute_app_data.py once locally to generate app_data/.
    """
    def _p(*parts):
        return os.path.join(APP_DATA, *parts)

    # -- cluster assignments + dates --
    df = pd.read_parquet(_p("df_clusters.parquet"))
    df["date"] = pd.to_datetime(df["date"])

    # -- prototype images (top-N per cluster) and metadata --
    top_data   = torch.load(_p("top_samples.pt"), weights_only=False)
    var_stats  = top_data["var_stats"]
    vars_names = top_data["vars_names"]
    nvars      = len(vars_names)

    # -- pre-computed quantile arrays for Q-Q plots --
    quantiles = torch.load(_p("quantiles.pt"), weights_only=False)

    # -- params (for N_PROTO) --
    import yaml
    with open(_p("params.yaml")) as f:
        params = yaml.safe_load(f)
    N_PROTO = params["criterion"]["num_proto"]

    # -- ENSO index --
    oni_index = read_enso_data(path=_p("oni_index.xlsx"))

    # -- seasonal grouping --
    df, grupos = compute_seasonal_groups(df)

    df["date_period"] = df["date"].dt.to_period("M")
    df_el_nino = df.merge(oni_index, left_on="date_period", right_index=True, how="left")

    return {
        "df":          df,
        "df_el_nino":  df_el_nino,
        "oni_index":   oni_index,
        "top_data":    top_data,    # {images: {cid: tensor}, var_stats, vars_names, top_n}
        "quantiles":   quantiles,   # {q, global: {var: array}, clusters: {cid: {var: array}}}
        "grupos":      grupos,
        "var_stats":   var_stats,
        "vars_names":  vars_names,
        "N_PROTO":     N_PROTO,
        "nvars":       nvars,
    }


# ---------------------------------------------------------------------------
# Page 1 – Interactive Plotly helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Rendering monthly frequency…")
def fig_monthly_frequency_plotly(selected_clusters: tuple, title: str = "Monthly Occurrence Frequency"):
    """Plotly interactive line chart: cluster frequency by calendar month."""
    d = load_app_data()
    df = d["df"].copy()
    df["month"] = df["date"].dt.month
    freq = pd.crosstab(df["month"], df["cluster_id"], normalize="index") * 100

    fig = go.Figure()
    for cid in selected_clusters:
        if cid not in freq.columns:
            continue
        fig.add_trace(go.Scatter(
            x=MONTH_NAMES,
            y=freq[cid].values,
            name=f"C{cid}",
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color=PALETTE30_HEX[cid % 30], width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Frequency within month (%)",
        legend_title="Cluster",
        hovermode="x unified",
        height=420,
        margin=dict(t=50, b=40),
    )
    return fig


@st.cache_data(show_spinner="Rendering Q-Q plot…")
def fig_qq_plotly_split(selected_clusters: tuple, **_):
    """Plotly Q-Q deviation with one subplot per variable (Tmin | Tmax).
    Uses pre-computed quantiles from app_data/quantiles.pt.
    """
    d = load_app_data()
    quant  = d["quantiles"]
    vars_names = d["vars_names"]
    nvars  = d["nvars"]
    q      = quant["q"]

    fig = make_subplots(rows=1, cols=nvars, subplot_titles=vars_names, shared_yaxes=True)
    for var in range(nvars):
        global_q = quant["global"][var]
        for cid in selected_clusters:
            cq = quant["clusters"].get(cid, {}).get(var)
            if cq is None or np.all(np.isnan(cq)):
                continue
            delta = cq - global_q
            fig.add_trace(
                go.Scatter(
                    x=q, y=delta,
                    name=f"C{cid}",
                    mode="lines",
                    line=dict(color=PALETTE30_HEX[cid % 30], width=1.8),
                    legendgroup=f"C{cid}",
                    showlegend=(var == 0),
                    hovertemplate=f"C{cid}<br>q=%{{x:.2f}}<br>Δ=%{{y:.3f}}°C<extra></extra>",
                ),
                row=1, col=var + 1,
            )
        fig.update_xaxes(title_text="Quantile", row=1, col=var + 1)

    fig.update_yaxes(title_text="Δ Quantile vs Global (°C)", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.5)
    fig.update_layout(
        title="Q-Q Deviation from Global Distribution (by variable)",
        legend_title="Cluster", height=460, hovermode="x", margin=dict(t=60, b=40),
    )
    return fig


@st.cache_data(show_spinner="Rendering Q-Q plot…")
def fig_qq_plotly_combined(selected_clusters: tuple, subtitle: str = "", **_):
    """Plotly Q-Q deviation with Tmin + Tmax in one plot (different dash per variable).
    Uses pre-computed quantiles from app_data/quantiles.pt.
    """
    d = load_app_data()
    quant      = d["quantiles"]
    vars_names = d["vars_names"]
    nvars      = d["nvars"]
    q          = quant["q"]

    fig = go.Figure()
    for var in range(nvars):
        global_q = quant["global"][var]
        dash = _VAR_DASH[var % len(_VAR_DASH)]
        for cid in selected_clusters:
            cq = quant["clusters"].get(cid, {}).get(var)
            if cq is None or np.all(np.isnan(cq)):
                continue
            delta = cq - global_q
            fig.add_trace(go.Scatter(
                x=q, y=delta,
                name=f"C{cid} – {vars_names[var]}",
                mode="lines",
                line=dict(color=PALETTE30_HEX[cid % 30], width=1.8, dash=dash),
                legendgroup=f"C{cid}",
                hovertemplate=f"C{cid} {vars_names[var]}<br>q=%{{x:.2f}}<br>Δ=%{{y:.3f}}°C<extra></extra>",
            ))

    title = "Q-Q Deviation from Global Distribution"
    if subtitle:
        title += f" — {subtitle}"
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.5)
    fig.update_layout(
        title=title, xaxis_title="Quantile", yaxis_title="Δ Quantile vs Global (°C)",
        legend_title="Cluster – Variable", height=460, hovermode="x",
        margin=dict(t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1 – Climate Regimes plots
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Rendering cluster prototypes…")
def fig_cluster_prototypes(selected_clusters: tuple, top: int = 5):
    """Exact layout match to plot_cluster_prototypes() in notebooks/analysis.ipynb.
    Images come from app_data/top_samples.pt (pre-computed, no NetCDF needed).
    """
    d          = load_app_data()
    top_data   = d["top_data"]
    var_stats  = d["var_stats"]
    vars_names = d["vars_names"]
    nvars      = d["nvars"]
    images     = top_data["images"]   # {cid: tensor(k, nvars, H, W)}

    clusters = list(selected_clusters)
    n_cols   = len(clusters)
    n_rows   = top * nvars

    fig_w = max(6, 40 * n_cols / 30)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for col_idx, cid in enumerate(clusters):
        imgs = images.get(cid)
        if imgs is None:
            continue
        n_show = min(top, len(imgs))
        for s_idx in range(n_show):
            img = imgs[s_idx]   # tensor(nvars, H, W)
            for var in range(nvars):
                ax = axes[s_idx * nvars + var, col_idx]
                vmin, vmax = var_stats[var]
                ax.imshow(img[var].numpy(), cmap="coolwarm", vmin=vmin, vmax=vmax,
                          aspect="auto")
                if s_idx * nvars + var == 0:
                    ax.set_title(f"Cluster {cid}", fontsize=10, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        f"S{s_idx + 1}\n{vars_names[var]}",
                        rotation=0, labelpad=25, va="center", fontsize=8,
                    )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for r in range(nvars, n_rows, nvars):
        upper_pos = axes[r - 1, 0].get_position()
        lower_pos = axes[r,     0].get_position()
        y_mid = (upper_pos.y0 + lower_pos.y1) / 2
        fig.add_artist(plt.Line2D(
            [0.1, 0.9], [y_mid, y_mid],
            transform=fig.transFigure,
            color="black", linewidth=1.5, linestyle="--",
        ))

    return fig


@st.cache_data(show_spinner="Rendering monthly frequency…")
def fig_monthly_frequency(selected_clusters: tuple):
    d = load_app_data()
    df = d["df"].copy()
    df["month"] = df["date"].dt.month
    freq = pd.crosstab(df["month"], df["cluster_id"], normalize="index") * 100
    freq = freq[[c for c in selected_clusters if c in freq.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE30[c % 30] for c in selected_clusters]
    freq.plot(ax=ax, color=colors, alpha=0.8, lw=1.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Monthly Occurrence Frequency by Cluster")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Rendering Q-Q plot…")
def fig_qq_plot(selected_clusters: tuple, n_quantiles: int = 200):
    d = load_app_data()
    df = d["df"]
    combined = d["combined_dataset"]
    vars_names = d["vars_names"]
    nvars = d["nvars"]

    q = np.linspace(0.01, 0.99, n_quantiles)
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    # combined_dataset is train only; restrict to train samples
    df_train = df[df["sample_type"] == "train"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for var in range(nvars):
        global_q = np.quantile(combined[:, var].ravel(), q)
        for cid in selected_clusters:
            idx = df_train[df_train["cluster_id"] == cid]["combined_idx"].values
            if len(idx) == 0:
                continue
            data = combined[idx, var].ravel()
            ax.plot(
                q,
                np.quantile(data, q) - global_q,
                alpha=0.7,
                color=PALETTE30[cid % 30],
                linestyle=linestyles[var % len(linestyles)],
                label=f"C{cid} – {vars_names[var]}",
            )

    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Δ Quantile vs Global (°C)")
    ax.set_title("Q-Q Deviation from Global Distribution")
    var_handles = [
        plt.Line2D([0], [0], color="gray", lw=2, linestyle=linestyles[v % len(linestyles)],
                   label=vars_names[v])
        for v in range(nvars)
    ]
    legend1 = ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.add_artist(legend1)
    ax.legend(handles=var_handles, title="Variable", bbox_to_anchor=(1.01, 0.4), loc="upper left")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 2 – Global ENSO Analysis plots
# ---------------------------------------------------------------------------

def _lag_formula(comparison_mode: str, lag: int) -> tuple[str, str]:
    """Return (lag_description, LaTeX formula string) for a given lag and mode.

    Lag convention (pandas shift):
      shift(+L): at time t we see ONI from t-L  → ENSO leads cluster by L months
      shift(-L): at time t we see ONI from t+L  → cluster leads ENSO by L months
    """
    if lag == 0:
        lag_desc = "Simultaneous  (Lag = 0)"
        enso_sub = "_t"
    elif lag > 0:
        s = "s" if lag > 1 else ""
        lag_desc = f"ENSO leads cluster by {lag} month{s}  (Lag = +{lag})"
        enso_sub = r"_{t-" + str(lag) + r"}"
    else:
        s = "s" if abs(lag) > 1 else ""
        lag_desc = f"Cluster leads ENSO by {abs(lag)} month{s}  (Lag = {lag})"
        enso_sub = r"_{t+" + str(abs(lag)) + r"}"

    if comparison_mode == "climatology":
        formula = (
            r"$\Delta P_k = P(k_t \mid \mathrm{El\ Niño}" + enso_sub + r") - P(k)$"
        )
    else:
        formula = (
            r"$\Delta P_k = P(k_t \mid \mathrm{El\ Niño}" + enso_sub
            + r") - P(k_t \mid \mathrm{Neutral})$"
        )
    return lag_desc, formula


@st.cache_resource(show_spinner="Computing ENSO anomaly bar chart…")
def fig_anomaly_at_lag(comparison_mode: str, year_start: int, year_end: int,
                       selected_clusters: tuple, lag: int = 0):
    d = load_app_data()
    df_base = d["df"]
    oni_index = d["oni_index"]

    merged = _build_lagged_df(df_base, oni_index, lag, year_start, year_end)
    try:
        anomaly = compute_anomaly(merged, comparison_mode)
    except ValueError:
        return None

    anomaly = anomaly.reindex(range(d["N_PROTO"]), fill_value=0.0)
    selected = [c for c in selected_clusters if c in anomaly.index]
    anomaly = anomaly.loc[selected]

    lag_desc, formula = _lag_formula(comparison_mode, lag)
    colors = ["steelblue" if v >= 0 else "tomato" for v in anomaly]
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.35 + 2), 4))
    ax.bar(anomaly.index.astype(str), anomaly.values, color=colors)
    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.7)
    ax.set_title(
        f"Cluster Anomalies  |  {lag_desc}  |  {year_start}–{year_end}\n{formula}",
        fontsize=11,
    )
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Anomaly (%)")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Computing lagged ENSO response…")
def fig_anomaly_all_lags_plotly(comparison_mode: str, year_start: int, year_end: int,
                                 selected_clusters: tuple, subtitle: str = ""):
    """Plotly interactive line chart: anomaly vs ENSO lag for each cluster."""
    d = load_app_data()
    df_base = d["df"]
    oni_index = d["oni_index"]

    anom = get_lagged_anomaly(df_base, oni_index, comparison_mode, year_start, year_end)
    selected = [c for c in selected_clusters if c in anom.index]
    anom = anom.loc[selected]

    _, formula_lag0 = _lag_formula(comparison_mode, 0)
    # Strip LaTeX delimiters for Plotly title (plain text fallback)
    mode_label = "P(k|El Niño) − P(k)" if comparison_mode == "climatology" \
                 else "P(k|El Niño) − P(k|Neutral)"

    fig = go.Figure()
    for cid in selected:
        fig.add_trace(go.Scatter(
            x=list(anom.columns),
            y=anom.loc[cid].values,
            name=f"C{cid}",
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color=PALETTE30_HEX[cid % 30], width=2),
            hovertemplate=f"C{cid}<br>Lag=%{{x}}m<br>Δ=%{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.4)
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, opacity=0.4)

    title = f"Anomaly vs ENSO Lag  |  {year_start}–{year_end}  |  {mode_label}"
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"

    fig.update_layout(
        title=title,
        xaxis=dict(title="ENSO Lag (months)",
                   tickmode="linear", dtick=2,
                   zeroline=False),
        yaxis_title="Anomaly (%)",
        legend_title="Cluster",
        hovermode="x unified",
        height=480,
        margin=dict(t=70, b=50),
    )
    return fig


@st.cache_data(show_spinner="Computing monthly heatmap…")
def fig_global_heatmap_for_month(comparison_mode: str, year_start: int, year_end: int,
                                  selected_clusters: tuple, month: int):
    """Heatmap: (selected_clusters × lag) for a fixed calendar month."""
    d = load_app_data()
    df_base = d["df"]
    oni_index = d["oni_index"]

    data = {}
    for lag in range(-12, 13):
        merged = _build_lagged_df(df_base, oni_index, lag, year_start, year_end)
        merged_m = merged[merged["month"] == month]
        nino_m = merged_m[merged_m["Label"] == "El Niño"]
        if comparison_mode == "neutral":
            base_m = merged_m[merged_m["Label"] == "Neutro"]
        else:
            base_m = merged_m

        for cid in selected_clusters:
            n_nino = len(nino_m)
            n_base = len(base_m)
            if n_nino < 5 or n_base < 5:
                data[(cid, lag)] = np.nan
            else:
                p_nino = (nino_m["cluster_id"] == cid).sum() / n_nino * 100
                p_base = (base_m["cluster_id"] == cid).sum() / n_base * 100
                data[(cid, lag)] = p_nino - p_base

    pivot = _make_pivot(data, row_name="Cluster")
    pivot.index = [f"C{i}" for i in pivot.index]

    vabs = np.nanmax(np.abs(pivot.values)) if not np.all(np.isnan(pivot.values)) else 1
    fig, ax = plt.subplots(figsize=(16, max(3, len(selected_clusters) * 0.5 + 2)))
    sns.heatmap(
        pivot, cmap="coolwarm", center=0, vmin=-vabs, vmax=vabs,
        annot=True, fmt=".1f", linewidths=0.4, ax=ax,
        cbar_kws={"label": "Anomaly (%)"},
    )
    ax.set_title(
        f"Cluster vs Lag Heatmap  |  Month: {MONTH_NAMES[month-1]}  |  {year_start}–{year_end}\n"
        f"Mode: {comparison_mode}"
    )
    ax.set_xlabel("ENSO Lag (months)")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 3 – Specific Regimes Analysis plots
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing per-cluster month×lag heatmap…")
def fig_cluster_heatmap(cluster_id: int, comparison_mode: str,
                        year_start: int, year_end: int):
    """Month × Lag heatmap for a single cluster."""
    d = load_app_data()
    df_base = d["df"]
    oni_index = d["oni_index"]

    climate_anon = {}
    for lag in range(-12, 13):
        merged = _build_lagged_df(df_base, oni_index, lag, year_start, year_end)
        for month in range(1, 13):
            mask_m = merged["month"] == month
            nino_m = merged[mask_m & (merged["Label"] == "El Niño")]
            if comparison_mode == "neutral":
                base_m = merged[mask_m & (merged["Label"] == "Neutro")]
            else:
                base_m = merged[mask_m]

            n_nino, n_base = len(nino_m), len(base_m)
            if n_nino < 5 or n_base < 5:
                climate_anon[(month, lag)] = np.nan
            else:
                p_nino = (nino_m["cluster_id"] == cluster_id).sum() / n_nino * 100
                p_base = (base_m["cluster_id"] == cluster_id).sum() / n_base * 100
                climate_anon[(month, lag)] = p_nino - p_base

    pivot = _make_pivot(climate_anon, row_name="Month")
    pivot.index = [MONTH_NAMES[i - 1] for i in pivot.index]

    vabs = np.nanmax(np.abs(pivot.values)) if not np.all(np.isnan(pivot.values)) else 1
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot, cmap="coolwarm", center=0, vmin=-vabs, vmax=vabs,
        annot=True, fmt=".1f", linewidths=0.4, ax=ax,
        cbar_kws={"label": "Anomaly (%)"},
    )
    ax.set_title(
        f"Month × Lag Heatmap  |  Cluster {cluster_id}  |  {year_start}–{year_end}\n"
        f"Mode: {comparison_mode}"
    )
    ax.set_xlabel("ENSO Lag (months)")
    ax.set_ylabel("Month")
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Rendering time series…")
def fig_temporal_evolution_separate(cluster_id: int, windows: tuple, year_start: int, year_end: int):
    """Time series for a single cluster with multiple moving-average windows."""
    d = load_app_data()
    df_el_nino = d["df_el_nino"].copy()
    df_el_nino = df_el_nino[
        (df_el_nino["date"].dt.year >= year_start) &
        (df_el_nino["date"].dt.year <= year_end)
    ]
    df_el_nino["date"] = pd.to_datetime(df_el_nino["date"])
    df_el_nino = df_el_nino.sort_values("date").set_index("date")
    df_el_nino["is_target"] = (df_el_nino["cluster_id"] == cluster_id).astype(int)

    monthly = df_el_nino.resample("MS").agg({"is_target": "mean", "Label": "first"})

    window_styles = {6: ("lightblue", 1.2, "--"), 12: ("royalblue", 2, "-"), 24: ("navy", 2.5, "-")}
    default_styles = [("tab:blue", 1.2, "--"), ("royalblue", 2, "-"), ("navy", 2.5, "-")]

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, w in enumerate(windows):
        col, lw, ls = window_styles.get(w, default_styles[i % len(default_styles)])
        ax.plot(
            monthly.index, monthly["is_target"].rolling(w).mean(),
            label=f"{w}m MA", color=col, lw=lw, linestyle=ls,
        )

    is_nino = monthly["Label"] == "El Niño"
    ax.fill_between(monthly.index, 0, 1, where=is_nino, color="red", alpha=0.12,
                    transform=ax.get_xaxis_transform(), label="El Niño")

    ax.set_title(f"Cluster {cluster_id} – Temporal Evolution")
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Rendering aggregated time series…")
def fig_temporal_evolution_aggregate(selected_clusters: tuple, windows: tuple,
                                     year_start: int, year_end: int):
    """Aggregated time series for all selected clusters combined."""
    d = load_app_data()
    df_el_nino = d["df_el_nino"].copy()
    df_el_nino = df_el_nino[
        (df_el_nino["date"].dt.year >= year_start) &
        (df_el_nino["date"].dt.year <= year_end)
    ]
    df_el_nino["date"] = pd.to_datetime(df_el_nino["date"])
    df_el_nino = df_el_nino.sort_values("date").set_index("date")
    df_el_nino["is_target"] = df_el_nino["cluster_id"].isin(selected_clusters).astype(int)

    monthly = df_el_nino.resample("MS").agg({"is_target": "mean", "Label": "first"})

    window_styles = {6: ("lightblue", 1.2, "--"), 12: ("royalblue", 2, "-"), 24: ("navy", 2.5, "-")}
    default_styles = [("tab:blue", 1.2, "--"), ("royalblue", 2, "-"), ("navy", 2.5, "-")]

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, w in enumerate(windows):
        col, lw, ls = window_styles.get(w, default_styles[i % len(default_styles)])
        ax.plot(
            monthly.index, monthly["is_target"].rolling(w).mean(),
            label=f"{w}m MA", color=col, lw=lw, linestyle=ls,
        )

    is_nino = monthly["Label"] == "El Niño"
    ax.fill_between(monthly.index, 0, 1, where=is_nino, color="red", alpha=0.12,
                    transform=ax.get_xaxis_transform(), label="El Niño")

    cluster_str = ", ".join(str(c) for c in selected_clusters)
    ax.set_title(f"Clusters [{cluster_str}] – Aggregated Temporal Evolution")
    ax.set_ylabel("Combined Frequency")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig
