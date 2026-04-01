"""
Page 1 – Climate Regimes
Shows spatial patterns and temporal characteristics of each cluster.
"""
import streamlit as st
from app_utils import (
    load_app_data,
    fig_cluster_prototypes,
    fig_monthly_frequency_plotly,
    fig_qq_plotly_split,
    fig_qq_plotly_combined,
)

st.set_page_config(page_title="Climate Regimes", layout="wide")
st.title("Climate Regimes")

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
data = load_app_data()
N_PROTO = data["N_PROTO"]
grupos = data["grupos"]   # {season: [cluster_ids]}
all_clusters = list(range(N_PROTO))

# ---------------------------------------------------------------------------
# Sidebar – filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    selected_clusters = st.multiselect(
        "Clusters to display",
        options=all_clusters,
        default=all_clusters,
    )
    top_n = st.slider("Top N samples per cluster", min_value=1, max_value=10, value=5)
    st.divider()
    by_season = st.toggle(
        "Break plots by dominant season",
        value=True,
    )

if not selected_clusters:
    st.warning("Select at least one cluster in the sidebar.")
    st.stop()

selected_tuple = tuple(sorted(selected_clusters))

# Clusters filtered per season (only selected ones)
season_groups = {
    s: [c for c in cs if c in selected_clusters]
    for s, cs in grupos.items()
}
season_groups = {s: cs for s, cs in season_groups.items() if cs}

# ---------------------------------------------------------------------------
# Season info banner
# ---------------------------------------------------------------------------
if by_season:
    st.info(
        "**Season grouping** assigns each cluster to its *dominant* season — "
        "the season with the highest historical occurrence frequency for that cluster "
        "(Southern Hemisphere: **Summer** = Dec–Feb, **Autumn** = Mar–May, "
        "**Winter** = Jun–Aug, **Spring** = Sep–Nov). "
        "All three plots are split into one tab per season, showing only the clusters "
        "most characteristic of that season. "
        "In the Q-Q view, **Tmin** and **Tmax** are shown together using "
        "different line styles (solid = Tmin, dashed = Tmax).",
        icon="ℹ️",
    )
    if not season_groups:
        st.warning("None of the selected clusters could be assigned to a season.")
        st.stop()

# ---------------------------------------------------------------------------
# Helper: render one section with or without season tabs
# ---------------------------------------------------------------------------
def _render_section(render_fn_flat, render_fn_season, section_caption_flat):
    """Call render_fn_flat() or render_fn_season(season, clusters) per tab."""
    if not by_season:
        st.caption(section_caption_flat)
        render_fn_flat()
    else:
        tabs = st.tabs(list(season_groups.keys()))
        for tab, (season, clusters) in zip(tabs, season_groups.items()):
            with tab:
                render_fn_season(season, clusters)


# ---------------------------------------------------------------------------
# Section 1 – Prototype images
# ---------------------------------------------------------------------------
st.subheader("Top Samples per Cluster")


def _proto_flat():
    with st.spinner("Rendering prototypes…"):
        st.pyplot(fig_cluster_prototypes(selected_tuple, top=top_n),
                  use_container_width=True)


def _proto_season(season, clusters):
    st.caption(
        f"**{season}** — clusters {sorted(clusters)} "
        f"(most frequent in {season})"
    )
    with st.spinner("Rendering prototypes…"):
        st.pyplot(
            fig_cluster_prototypes(tuple(sorted(clusters)), top=top_n),
            use_container_width=True,
        )


_render_section(
    _proto_flat,
    _proto_season,
    "Each column shows the top-N training samples with highest assignment probability. "
    "Rows alternate between climate variables (Tmin / Tmax). "
    "Dashed lines separate samples; title only on first cell per cluster.",
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 – Monthly frequency (Plotly)
# ---------------------------------------------------------------------------
st.subheader("Monthly Occurrence Frequency")


def _monthly_flat():
    st.plotly_chart(
        fig_monthly_frequency_plotly(selected_tuple),
        use_container_width=True,
    )


def _monthly_season(season, clusters):
    st.plotly_chart(
        fig_monthly_frequency_plotly(
            tuple(sorted(clusters)),
            title=f"Monthly Frequency — {season}  {sorted(clusters)}",
        ),
        use_container_width=True,
    )


_render_section(
    _monthly_flat,
    _monthly_season,
    "Share of days assigned to each cluster per calendar month "
    "(row-normalised: values across all clusters sum to 100 % per month).",
)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 – Q-Q deviation (Plotly)
# ---------------------------------------------------------------------------
st.subheader("Q-Q Deviation from Global Distribution")


def _qq_flat():
    st.caption(
        "Left subplot = **Tmin**, right subplot = **Tmax**. "
        "Y-axis: quantile of the cluster minus quantile of the global distribution (°C)."
    )
    st.plotly_chart(fig_qq_plotly_split(selected_tuple), use_container_width=True)


def _qq_season(season, clusters):
    st.caption(
        f"**{season}** clusters {sorted(clusters)}. "
        "**Solid** = Tmin, **dashed** = Tmax. Color identifies the cluster."
    )
    st.plotly_chart(
        fig_qq_plotly_combined(
            tuple(sorted(clusters)),
            subtitle=f"{season}  {sorted(clusters)}",
        ),
        use_container_width=True,
    )


_render_section(_qq_flat, _qq_season, "")
