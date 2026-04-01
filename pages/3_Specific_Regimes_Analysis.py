"""
Page 3 – Specific Regimes Analysis
Deep-dives into selected clusters: month×lag heatmaps and temporal evolution.
Two display modes:
  • Aggregate – all selected clusters combined into one plot
  • Separate  – one plot per cluster
"""
import streamlit as st
from app_utils import (
    load_app_data,
    fig_cluster_heatmap,
    fig_temporal_evolution_separate,
    fig_temporal_evolution_aggregate,
    MONTH_NAMES,
)

st.set_page_config(page_title="Specific Regimes Analysis", layout="wide")
st.title("Specific Regimes Analysis")

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
data = load_app_data()
N_PROTO = data["N_PROTO"]
df_el = data["df_el_nino"]
all_clusters = list(range(N_PROTO))

year_min = int(df_el["date"].dt.year.min())
year_max = int(df_el["date"].dt.year.max())

# ---------------------------------------------------------------------------
# Sidebar – filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    selected_clusters = st.multiselect(
        "Clusters",
        options=all_clusters,
        default=[2, 22],
        help="Select the clusters to analyse. Default: 2 and 22.",
    )

    comparison_mode = st.radio(
        "Comparison mode",
        options=["climatology", "neutral"],
        index=1,
        help=(
            "**climatology**: ΔP = P(k|El Niño) – P(k)\n\n"
            "**neutral**: ΔP = P(k|El Niño) – P(k|Neutral)"
        ),
    )

    year_range = st.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    year_start, year_end = year_range

    st.divider()

    display_mode = st.radio(
        "Display mode",
        options=["Separate (one plot per cluster)", "Aggregate (combined)"],
        index=0,
        help=(
            "**Separate**: renders one plot for each selected cluster.\n\n"
            "**Aggregate**: combines all selected clusters into a single plot."
        ),
    )
    aggregate = display_mode.startswith("Aggregate")

if not selected_clusters:
    st.warning("Select at least one cluster in the sidebar.")
    st.stop()

selected_tuple = tuple(sorted(selected_clusters))

# ---------------------------------------------------------------------------
# Section 1 – Month × Lag heatmaps
# ---------------------------------------------------------------------------
st.subheader("Month × Lag Heatmaps")
st.caption(
    "Rows = calendar month, Columns = ENSO lag (months). "
    "Each cell shows the cluster probability anomaly for that month–lag combination."
)

if aggregate:
    # Aggregate: treat all selected clusters as one combined regime
    st.info(
        "Aggregate mode: the selected clusters are treated as a **single regime**. "
        "The anomaly is ΔP = P(cluster ∈ selected | El Niño) − baseline, "
        "summing their counts together rather than averaging individual anomalies."
    )
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from app_utils import _build_lagged_df, _make_pivot, load_app_data

    @st.cache_data(show_spinner="Computing aggregated heatmap…")
    def fig_aggregate_heatmap(selected_clusters_: tuple, comparison_mode_: str,
                              year_start_: int, year_end_: int):
        d = load_app_data()
        df_base = d["df"]
        oni_index = d["oni_index"]

        climate_anon = {}
        for lag in range(-12, 13):
            merged = _build_lagged_df(df_base, oni_index, lag, year_start_, year_end_)
            for month in range(1, 13):
                mask_m = merged["month"] == month
                nino_m = merged[mask_m & (merged["Label"] == "El Niño")]
                if comparison_mode_ == "neutral":
                    base_m = merged[mask_m & (merged["Label"] == "Neutro")]
                else:
                    base_m = merged[mask_m]

                n_nino, n_base = len(nino_m), len(base_m)
                if n_nino < 5 or n_base < 5:
                    climate_anon[(month, lag)] = np.nan
                    continue
                # Combined count: any day where cluster_id is in the selected set
                p_nino = nino_m["cluster_id"].isin(selected_clusters_).sum() / n_nino * 100
                p_base = base_m["cluster_id"].isin(selected_clusters_).sum() / n_base * 100
                climate_anon[(month, lag)] = p_nino - p_base

        pivot = _make_pivot(climate_anon, row_name="Month")
        pivot.index = [MONTH_NAMES[i - 1] for i in pivot.index]

        vabs = np.nanmax(np.abs(pivot.values)) if not np.all(np.isnan(pivot.values)) else 1
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            pivot, cmap="coolwarm", center=0, vmin=-vabs, vmax=vabs,
            annot=True, fmt=".1f", linewidths=0.4, ax=ax,
            cbar_kws={"label": "Mean Anomaly (%)"},
        )
        cluster_str = ", ".join(str(c) for c in selected_clusters_)
        ax.set_title(
            f"Month × Lag  |  Clusters [{cluster_str}] (mean)  |  {year_start_}–{year_end_}\n"
            f"Mode: {comparison_mode_}"
        )
        ax.set_xlabel("ENSO Lag (months)")
        ax.set_ylabel("Month")
        plt.tight_layout()
        return fig

    agg_heat_fig = fig_aggregate_heatmap(
        selected_tuple, comparison_mode, year_start, year_end
    )
    st.pyplot(agg_heat_fig, use_container_width=True)

else:
    # Separate heatmap per cluster
    for cid in selected_tuple:
        st.markdown(f"#### Cluster {cid}")
        heat_fig = fig_cluster_heatmap(cid, comparison_mode, year_start, year_end)
        st.pyplot(heat_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 – Temporal evolution (time series)
# ---------------------------------------------------------------------------
st.subheader("Temporal Evolution")
st.caption(
    "Moving-average frequency of the selected cluster(s) over time. "
    "Red shading marks El Niño periods."
)

available_windows = [3, 6, 12, 24, 36]
selected_windows = st.multiselect(
    "Moving-average windows (months)",
    options=available_windows,
    default=[6, 12, 24],
    help="Select one or more window sizes for the rolling mean.",
)
if not selected_windows:
    selected_windows = [12]

windows_tuple = tuple(sorted(selected_windows))

if aggregate:
    ts_fig = fig_temporal_evolution_aggregate(selected_tuple, windows_tuple, year_start, year_end)
    st.pyplot(ts_fig, use_container_width=True)
else:
    for cid in selected_tuple:
        st.markdown(f"#### Cluster {cid}")
        ts_fig = fig_temporal_evolution_separate(cid, windows_tuple, year_start, year_end)
        st.pyplot(ts_fig, use_container_width=True)
