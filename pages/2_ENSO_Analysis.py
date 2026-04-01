"""
Page 2 – Global ENSO Analysis
Explores how cluster occurrence probabilities shift during El Niño events,
at different lags and across calendar months.
"""
import streamlit as st
from app_utils import (
    load_app_data,
    fig_anomaly_at_lag,
    fig_anomaly_all_lags_plotly,
    fig_global_heatmap_for_month,
    MONTH_NAMES,
)

st.set_page_config(page_title="ENSO Analysis", layout="wide")
st.title("ENSO Analysis – Global")

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
data = load_app_data()
N_PROTO = data["N_PROTO"]
df_el = data["df_el_nino"]
grupos = data["grupos"]   # {season: [cluster_ids]}
all_clusters = list(range(N_PROTO))

year_min = int(df_el["date"].dt.year.min())
year_max = int(df_el["date"].dt.year.max())

# ---------------------------------------------------------------------------
# Sidebar – shared filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

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

    selected_clusters = st.multiselect(
        "Clusters to display",
        options=all_clusters,
        default=all_clusters,
        help="Filter which clusters appear in the plots.",
    )

if not selected_clusters:
    st.warning("Select at least one cluster in the sidebar.")
    st.stop()

selected_tuple = tuple(sorted(selected_clusters))

# Season groups filtered to the selected clusters (used in Part 2 tabs)
season_groups = {
    s: [c for c in cs if c in selected_clusters]
    for s, cs in grupos.items()
}
season_groups = {s: cs for s, cs in season_groups.items() if cs}

# ---------------------------------------------------------------------------
# Part 1 – Anomalies at a chosen lag
# ---------------------------------------------------------------------------
st.subheader("Part 1 · Anomalies at a Chosen Lag")
st.caption(
    "The formula in the title updates with the lag direction: "
    "positive lag → ENSO leads the cluster; negative lag → cluster leads ENSO."
)

lag = st.slider(
    "ENSO lag (months)", min_value=-12, max_value=12, value=0,
    help=(
        "**Positive**: ONI shifted back → ENSO precedes the cluster response.\n\n"
        "**Negative**: ONI shifted forward → cluster occurs before the ENSO event."
    ),
)
fig1 = fig_anomaly_at_lag(comparison_mode, year_start, year_end, selected_tuple, lag)
if fig1 is None:
    st.warning("Not enough El Niño samples in the selected period / lag to compute anomalies.")
else:
    st.pyplot(fig1, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Part 2 – Anomalies across all lags  (Plotly, split by season)
# ---------------------------------------------------------------------------
st.subheader("Part 2 · Anomalies Across All Lags (−12 to +12 months)")
st.caption(
    "Each line shows how a cluster's probability anomaly varies as the ENSO index "
    "is shifted relative to the observation date. "
    "Tabs group clusters by their dominant season."
)

if season_groups:
    tabs = st.tabs(list(season_groups.keys()))
    for tab, (season, clusters) in zip(tabs, season_groups.items()):
        with tab:
            st.plotly_chart(
                fig_anomaly_all_lags_plotly(
                    comparison_mode,
                    year_start,
                    year_end,
                    tuple(sorted(clusters)),
                    subtitle=f"{season} clusters · {sorted(clusters)}",
                ),
                use_container_width=True,
            )
else:
    st.plotly_chart(
        fig_anomaly_all_lags_plotly(comparison_mode, year_start, year_end, selected_tuple),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Part 3 – Month × Lag heatmap
# ---------------------------------------------------------------------------
st.subheader("Part 3 · Cluster × Lag Heatmap for a Selected Month")
st.caption(
    "For the chosen calendar month, shows the anomaly of each cluster (rows) "
    "across different ENSO lags (columns)."
)

month_idx = st.selectbox(
    "Select month",
    options=list(range(1, 13)),
    format_func=lambda m: MONTH_NAMES[m - 1],
    index=11,  # default: December
)
fig3 = fig_global_heatmap_for_month(comparison_mode, year_start, year_end, selected_tuple, month_idx)
st.pyplot(fig3, use_container_width=True)
