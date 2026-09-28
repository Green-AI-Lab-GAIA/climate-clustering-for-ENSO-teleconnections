from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import src.inference as inf
from src.el_nino import read_enso_data


st.set_page_config(page_title="Climate Clustering Analysis", layout="wide")


def get_season(date: pd.Timestamp) -> str:
    month, day = date.month, date.day
    if (month == 12 and day >= 21) or month in (1, 2) or (month == 3 and day <= 20):
        return "Summer"
    if (month == 3 and day >= 21) or month in (4, 5) or (month == 6 and day <= 20):
        return "Autumn"
    if (month == 6 and day >= 21) or month in (7, 8) or (month == 9 and day <= 22):
        return "Winter"
    return "Spring"


def compute_anomaly(data: pd.DataFrame, comparison_mode: str) -> pd.Series:
    prob = pd.crosstab(data["Label"], data["cluster_id"], normalize="index") * 100
    if "El Niño" not in prob.index and "El Nino" not in prob.index:
        raise ValueError("No El Niño rows found in selected period")

    nino_label = "El Niño" if "El Niño" in prob.index else "El Nino"
    p_nino = prob.loc[nino_label]

    if comparison_mode == "climatology":
        baseline = data["cluster_id"].value_counts(normalize=True) * 100
        baseline = baseline.reindex(p_nino.index, fill_value=0)
    else:
        neutral_label = "Neutro" if "Neutro" in prob.index else "Neutral"
        if neutral_label not in prob.index:
            raise ValueError("No Neutral rows found in selected period")
        baseline = prob.loc[neutral_label]

    return p_nino - baseline


def rolling_cluster_frequency(
    df_el_nino: pd.DataFrame,
    target_clusters: list[int],
    aggregate: bool,
    windows: tuple[int, int, int],
) -> pd.DataFrame:
    cur = df_el_nino.copy()
    cur["date"] = pd.to_datetime(cur["date"])
    cur = cur.sort_values("date").set_index("date")

    rows = []
    if aggregate and len(target_clusters) > 1:
        cur["is_target"] = cur["cluster_id"].isin(target_clusters).astype(float)
        monthly = cur.resample("MS").agg({"is_target": "mean", "Label": "first"})
        for window in windows:
            rows.append(
                pd.DataFrame(
                    {
                        "date": monthly.index,
                        "frequency": monthly["is_target"].rolling(window).mean(),
                        "cluster": f"Agg[{', '.join(map(str, target_clusters))}]",
                        "window": f"{window}m",
                        "Label": monthly["Label"],
                    }
                )
            )
    else:
        for cluster_id in target_clusters:
            tmp = cur.copy()
            tmp["is_target"] = (tmp["cluster_id"] == cluster_id).astype(float)
            monthly = tmp.resample("MS").agg({"is_target": "mean", "Label": "first"})
            for window in windows:
                rows.append(
                    pd.DataFrame(
                        {
                            "date": monthly.index,
                            "frequency": monthly["is_target"].rolling(window).mean(),
                            "cluster": f"C{cluster_id}",
                            "window": f"{window}m",
                            "Label": monthly["Label"],
                        }
                    )
                )

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["frequency"])
    return out


@st.cache_data(show_spinner=False)
def _load_tensor_if_exists(path: str):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


@st.cache_data(show_spinner=True)
def load_state(
    validation: bool,
    config_file: str,
    save_path: str,
    model_suffix: str | None,
):
    params, dataset = inf.read_data(config_file, validation=validation)

    e_file = os.path.join(save_path, "E.pt")
    f_file = os.path.join(save_path, "F.pt")
    e_val_file = os.path.join(save_path, "E_val.pt")
    f_val_file = os.path.join(save_path, "F_val.pt")

    tsne_e_file = os.path.join(save_path, "tsne_E.pt")
    tsne_prot_file = os.path.join(save_path, "tsne_prot.pt")
    tsne_eval_file = os.path.join(save_path, "tsne_Eval.pt")

    E = _load_tensor_if_exists(e_file)
    F = _load_tensor_if_exists(f_file)
    E_val = _load_tensor_if_exists(e_val_file) if validation else None
    F_val = _load_tensor_if_exists(f_val_file) if validation else None
    tsne_E = _load_tensor_if_exists(tsne_e_file)
    tsne_prot = _load_tensor_if_exists(tsne_prot_file)
    tsne_Eval = _load_tensor_if_exists(tsne_eval_file) if validation else None

    if E is None or F is None or tsne_E is None or tsne_prot is None or (validation and (E_val is None or F_val is None or tsne_Eval is None)):
        target_encoder, prot = inf.load_model(params, model_suffix=model_suffix)

        E, F = inf.get_model_results(
            read_path=save_path,
            params=params,
            dataset=dataset,
            encoder=target_encoder,
            prototypes=prot,
        )

        if validation:
            E_val, F_val = inf.get_model_results(
                read_path=save_path,
                params=params,
                dataset=dataset.validation_imgs,
                encoder=target_encoder,
                prototypes=prot,
                validation=True,
            )

        tsne_E, tsne_prot, tsne_Eval = inf.get_TSNE(
            save_path,
            E=E.cpu(),
            prot=prot.cpu(),
            E_val=E_val.cpu() if validation and E_val is not None else None,
            validation=validation,
        )

    cluster_prob, cluster_id = torch.max(F, dim=1)
    df = pd.DataFrame(
        {
            "cluster_id": cluster_id.cpu().numpy(),
            "cluster_prob": cluster_prob.cpu().numpy(),
            "sample_type": "train",
            "date": pd.to_datetime(dataset.time),
        }
    )

    if validation and F_val is not None:
        val_cluster_prob, val_cluster_id = torch.max(F_val, dim=1)
        df_val = pd.DataFrame(
            {
                "cluster_id": val_cluster_id.cpu().numpy(),
                "cluster_prob": val_cluster_prob.cpu().numpy(),
                "sample_type": "val",
                "date": pd.to_datetime(dataset.val_time),
            }
        )
        df = pd.concat([df, df_val]).sort_values("date").reset_index(drop=True)

        combined_dataset = torch.cat((dataset[:][0], dataset.validation_imgs))[df.index]
        tsne = np.concatenate((np.asarray(tsne_E), np.asarray(tsne_Eval)))[df.index]

        mean = torch.tensor(params["data"]["norm_means"]).view(len(params["data"]["surf_vars"]), 1, 1)
        std = torch.tensor(params["data"]["norm_stds"]).view(len(params["data"]["surf_vars"]), 1, 1)
        combined_dataset = (combined_dataset * std) + mean
    else:
        combined_dataset = dataset[:][0]
        tsne = np.asarray(tsne_E)

    nvars = combined_dataset.shape[1]
    vars_names = params["data"]["surf_vars"]
    var_stats = {
        var_idx: (
            float(combined_dataset[:, var_idx].min()),
            float(combined_dataset[:, var_idx].max()),
        )
        for var_idx in range(nvars)
    }

    oni_index = read_enso_data()
    df["date_period"] = df["date"].dt.to_period("M")
    df_el_nino = df.merge(oni_index, left_on="date_period", right_index=True, how="left")
    df_el_nino["Average Tmin"] = combined_dataset[:, 0].mean(dim=(1, 2)).cpu().numpy()
    if nvars > 1:
        df_el_nino["Average Tmax"] = combined_dataset[:, 1].mean(dim=(1, 2)).cpu().numpy()
    else:
        df_el_nino["Average Tmax"] = np.nan
    df_el_nino["season"] = df_el_nino["date"].apply(get_season)

    return {
        "params": params,
        "dataset_tensor": combined_dataset,
        "df": df,
        "df_el_nino": df_el_nino,
        "tsne": tsne,
        "tsne_prot": np.asarray(tsne_prot),
        "nvars": nvars,
        "vars_names": vars_names,
        "var_stats": var_stats,
        "oni_index": oni_index,
        "n_proto": int(params["criterion"]["num_proto"]),
    }


def render_embedding_space(state):
    st.subheader("1) Embedding Space (t-SNE)")
    df_plot = state["df_el_nino"].copy()
    df_plot["x"] = state["tsne"][:, 0]
    df_plot["y"] = state["tsne"][:, 1]

    color_metric = st.selectbox(
        "Color by",
        ["cluster_id", "Average Tmin", "Average Tmax", "sample_type", "Label"],
        index=0,
        key="tsne_color_metric",
    )
    point_size = st.slider("Point size", 1, 8, 3, key="tsne_point_size")

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=color_metric,
        symbol="sample_type",
        opacity=0.7,
        hover_data=["date", "cluster_id", "cluster_prob", "Label"],
        title="Latent Space Projection",
    )
    fig.update_traces(marker={"size": point_size})

    prot = state["tsne_prot"]
    fig.add_trace(
        go.Scatter(
            x=prot[:, 0],
            y=prot[:, 1],
            mode="markers+text",
            marker={"size": 10, "symbol": "x", "color": "black"},
            text=[f"P{i}" for i in range(len(prot))],
            textposition="top center",
            name="Prototypes",
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cluster_prototypes(state):
    st.subheader("2) Cluster Prototypes")

    df = state["df"]
    combined_dataset = state["dataset_tensor"]
    nvars = state["nvars"]
    vars_names = state["vars_names"]
    var_stats = state["var_stats"]

    col1, col2, col3 = st.columns(3)
    with col1:
        cluster_id = st.selectbox(
            "Cluster", sorted(df["cluster_id"].unique().tolist()), key="proto_cluster"
        )
    with col2:
        top_k = st.slider("Top-k by assignment prob", 1, 20, 5, key="proto_topk")
    with col3:
        sample_rank = st.slider("Sample rank", 1, top_k, 1, key="proto_rank")

    top_samples = (
        df[(df["cluster_id"] == cluster_id) & (df["sample_type"] == "train")]
        .nlargest(top_k, "cluster_prob")
        .reset_index()
    )
    if top_samples.empty:
        st.info("No training samples found for this cluster.")
        return

    chosen = top_samples.iloc[sample_rank - 1]
    image_data = combined_dataset[int(chosen["index"])]

    st.caption(
        f"Cluster {cluster_id} • sample #{sample_rank}/{top_k} • prob={chosen['cluster_prob']:.3f} • date={chosen['date'].date()}"
    )

    cols = st.columns(nvars)
    for var_idx in range(nvars):
        with cols[var_idx]:
            vmin, vmax = var_stats[var_idx]
            fig = go.Figure(
                data=go.Heatmap(
                    z=image_data[var_idx].cpu().numpy(),
                    colorscale="RdBu_r",
                    zmin=vmin,
                    zmax=vmax,
                    colorbar={"title": vars_names[var_idx]},
                )
            )
            fig.update_layout(title=vars_names[var_idx], margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(fig, use_container_width=True)


def render_seasonal(state):
    st.subheader("3) Seasonal Analysis")
    df = state["df"].copy()
    df["season"] = df["date"].apply(get_season)

    seasonal_dist = pd.crosstab(df["cluster_id"], df["season"], normalize="index") * 100
    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        if season not in seasonal_dist.columns:
            seasonal_dist[season] = 0.0
    seasonal_dist = seasonal_dist[["Spring", "Summer", "Autumn", "Winter"]].reset_index()

    fig = px.bar(
        seasonal_dist,
        x="cluster_id",
        y=["Spring", "Summer", "Autumn", "Winter"],
        title="Seasonal Composition by Cluster",
        labels={"value": "Historical Frequency (%)", "cluster_id": "Cluster ID"},
    )
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    month_freq = pd.crosstab(df["date"].dt.month, df["cluster_id"], normalize="index") * 100
    sel_clusters = st.multiselect(
        "Clusters to inspect monthly frequency",
        options=sorted(df["cluster_id"].unique().tolist()),
        default=sorted(df["cluster_id"].unique().tolist())[:4],
        key="season_month_clusters",
    )
    if sel_clusters:
        monthly_long = month_freq[sel_clusters].reset_index().melt(
            id_vars="date", var_name="cluster_id", value_name="freq"
        )
        monthly_long = monthly_long.rename(columns={"date": "month"})
        fig2 = px.line(
            monthly_long,
            x="month",
            y="freq",
            color="cluster_id",
            markers=True,
            title="Monthly Frequency by Cluster",
            labels={"freq": "Frequency (%)", "month": "Month"},
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_enso(state, comparison_mode: str):
    st.subheader("4) ENSO Analysis")
    df_el_nino = state["df_el_nino"].copy()

    years = df_el_nino["date"].dt.year
    min_year, max_year = int(years.min()), int(years.max())

    c1, c2, c3 = st.columns(3)
    with c1:
        period_a = st.slider("Period A", min_year, max_year, (min_year, max_year), key="enso_p1")
    with c2:
        compare_two = st.checkbox("Compare with Period B", value=True, key="enso_compare_two")
    with c3:
        period_b = st.slider("Period B", min_year, max_year, (1994, max_year), key="enso_p2")

    periods = [period_a]
    if compare_two:
        periods.append(period_b)

    cols = st.columns(len(periods))
    for idx, (start, end) in enumerate(periods):
        cur = df_el_nino[df_el_nino["date"].dt.year.between(start, end)]
        try:
            anomaly = compute_anomaly(cur, comparison_mode).sort_index()
        except ValueError as exc:
            cols[idx].warning(str(exc))
            continue

        anom_df = anomaly.rename_axis("cluster_id").reset_index(name="anomaly")
        anom_df["sign"] = np.where(anom_df["anomaly"] >= 0, "Positive", "Negative")
        fig = px.bar(
            anom_df,
            x="cluster_id",
            y="anomaly",
            color="sign",
            color_discrete_map={"Positive": "darkblue", "Negative": "darkred"},
            title=f"Anomaly ({start}-{end})",
            labels={"anomaly": "Anomaly (%)", "cluster_id": "Cluster ID"},
        )
        fig.add_hline(y=0, line_dash="dash")
        cols[idx].plotly_chart(fig, use_container_width=True)

    st.markdown("### Sliding Windows")
    window_size = st.slider("Window size (years)", 5, 20, 11, key="enso_window_size")
    all_vals, labels = [], []
    for start_year in range(min_year, max_year, window_size):
        end_year = min(start_year + window_size - 1, max_year)
        cur = df_el_nino[df_el_nino["date"].dt.year.between(start_year, end_year)]
        if len(cur) == 0:
            continue
        try:
            anomaly = compute_anomaly(cur, comparison_mode)
        except ValueError:
            continue
        all_vals.append(anomaly)
        labels.append(f"{start_year}-{end_year}")

    if all_vals:
        df_anom = pd.DataFrame(all_vals, index=labels).reset_index(names="period")
        long_anom = df_anom.melt(id_vars="period", var_name="cluster_id", value_name="anomaly")
        fig = px.bar(
            long_anom,
            x="cluster_id",
            y="anomaly",
            facet_col="period",
            facet_col_wrap=3,
            color=np.where(long_anom["anomaly"] >= 0, "Positive", "Negative"),
            color_discrete_map={"Positive": "darkblue", "Negative": "darkred"},
            title="Cluster Anomalies Across Time Windows",
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)


_ENSO_COLS = ["Label", "ONI", "is_target", "label_color"]


def _build_lagged_df(df_el_nino: pd.DataFrame, oni_index: pd.DataFrame, lag: int, year_start: int | None):
    df = df_el_nino.copy()
    if year_start is not None:
        df = df[df["date"].dt.year >= year_start]
    df = df.drop(columns=_ENSO_COLS, errors="ignore")
    df["month"] = df["date"].dt.month
    return df.merge(oni_index.shift(lag).dropna(), left_on="date_period", right_index=True, how="left")


def compute_month_lag_pivot(
    df_el_nino: pd.DataFrame,
    oni_index: pd.DataFrame,
    comparison_mode: str,
    selected_clusters: list[int],
    aggregate: bool,
    year_start: int | None,
):
    lags = list(range(-12, 13))
    months = list(range(1, 13))
    out = {}

    if aggregate and len(selected_clusters) > 1:
        for month in months:
            for lag in lags:
                merged = _build_lagged_df(df_el_nino, oni_index, lag, year_start)
                mask_month = merged["month"] == month
                nino = merged[mask_month & (merged["Label"] == "El Niño")]
                baseline = (
                    merged[mask_month & (merged["Label"] == "Neutro")]
                    if comparison_mode == "neutral"
                    else merged[mask_month]
                )
                p_nino = nino["cluster_id"].isin(selected_clusters).mean() * 100 if len(nino) > 0 else 0.0
                p_base = baseline["cluster_id"].isin(selected_clusters).mean() * 100 if len(baseline) > 0 else 0.0
                out[(month, lag)] = p_nino - p_base
        pivot = pd.DataFrame.from_dict(out, orient="index", columns=["anomaly"])
        pivot.index = pd.MultiIndex.from_tuples(pivot.index, names=["month", "lag"])
        return {"Aggregated": pivot.reset_index().pivot(index="month", columns="lag", values="anomaly")}

    pivots = {}
    for cluster_id in selected_clusters:
        out = {}
        for month in months:
            for lag in lags:
                merged = _build_lagged_df(df_el_nino, oni_index, lag, year_start)
                mask_month = merged["month"] == month
                nino = merged[mask_month & (merged["Label"] == "El Niño")]
                baseline = (
                    merged[mask_month & (merged["Label"] == "Neutro")]
                    if comparison_mode == "neutral"
                    else merged[mask_month]
                )
                p_nino = ((nino["cluster_id"] == cluster_id).sum() / len(nino) * 100) if len(nino) > 0 else 0.0
                p_base = ((baseline["cluster_id"] == cluster_id).sum() / len(baseline) * 100) if len(baseline) > 0 else 0.0
                out[(month, lag)] = p_nino - p_base
        pivot = pd.DataFrame.from_dict(out, orient="index", columns=["anomaly"])
        pivot.index = pd.MultiIndex.from_tuples(pivot.index, names=["month", "lag"])
        pivots[f"C{cluster_id}"] = pivot.reset_index().pivot(index="month", columns="lag", values="anomaly")
    return pivots


def render_lagged_heatmaps(state, comparison_mode: str):
    st.markdown("### Monthly Lagged Heatmaps")
    df_el_nino = state["df_el_nino"]
    oni_index = state["oni_index"]
    all_clusters = sorted(df_el_nino["cluster_id"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_clusters = st.multiselect(
            "Clusters",
            all_clusters,
            default=[c for c in [2, 22] if c in all_clusters] or all_clusters[:2],
            key="lagged_clusters",
        )
    with c2:
        aggregate = st.checkbox(
            "Aggregate selected clusters as one",
            value=True,
            help="When enabled, occurrences of selected clusters are summed into one combined signal.",
            key="lagged_aggregate",
        )
    with c3:
        min_year = int(df_el_nino["date"].dt.year.min())
        max_year = int(df_el_nino["date"].dt.year.max())
        year_start = st.slider("Start year", min_year, max_year, max(min_year, 1994), key="lagged_year_start")

    if not selected_clusters:
        st.info("Select at least one cluster.")
        return

    pivots = compute_month_lag_pivot(
        df_el_nino,
        oni_index,
        comparison_mode,
        selected_clusters,
        aggregate,
        year_start,
    )

    for title, pivot in pivots.items():
        fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            origin="lower",
            aspect="auto",
            labels={"x": "ENSO Lag (months)", "y": "Month", "color": "Anomaly (%)"},
            title=f"{title} • Lagged ENSO Response",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


def render_temporal_evolution(state):
    st.subheader("6) Temporal Evolution of Clusters")
    df_el_nino = state["df_el_nino"]
    all_clusters = sorted(df_el_nino["cluster_id"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_clusters = st.multiselect(
            "Target clusters",
            options=all_clusters,
            default=[c for c in [2, 22, 25, 0] if c in all_clusters] or all_clusters[:2],
            key="temp_clusters",
        )
    with c2:
        aggregate = st.checkbox(
            "Aggregate selected clusters as one",
            value=False,
            help="If enabled, selected clusters are merged into one target frequency.",
            key="temp_aggregate",
        )
    with c3:
        windows = st.multiselect("Moving windows (months)", [3, 6, 12, 18, 24, 36], default=[6, 12, 24], key="temp_windows")

    if not selected_clusters:
        st.info("Select at least one cluster.")
        return
    if not windows:
        st.info("Select at least one moving window.")
        return

    freq_df = rolling_cluster_frequency(
        df_el_nino,
        target_clusters=selected_clusters,
        aggregate=aggregate,
        windows=tuple(sorted(windows)),
    )

    fig = px.line(
        freq_df,
        x="date",
        y="frequency",
        color="cluster",
        line_dash="window",
        title="Moving-Average Cluster Frequency",
        labels={"frequency": "Relative Frequency", "date": "Year"},
    )

    monthly = (
        df_el_nino.copy()
        .assign(date=pd.to_datetime(df_el_nino["date"]))
        .sort_values("date")
        .set_index("date")
        .resample("MS")
        .agg({"Label": "first"})
        .reset_index()
    )
    nino_mask = monthly["Label"] == "El Niño"
    if nino_mask.any():
        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=nino_mask.astype(float) * freq_df["frequency"].max(),
                mode="lines",
                line={"color": "rgba(220,20,60,0.25)", "width": 2},
                name="El Niño periods (indicator)",
                hovertemplate="%{x|%Y-%m}<extra></extra>",
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Climate Clustering Analysis (Interactive)")

    with st.sidebar:
        st.header("Configuration")
        validation = st.checkbox("Use validation set", value=True)
        config_file = st.text_input(
            "Config file",
            value="checkpoint/temperature-run2/params-temperature-exp-2-c30.yaml",
        )
        save_path = st.text_input("Results path", value="results/temperature-exp-2-c30")
        model_suffix = st.text_input("Model suffix (optional)", value="")
        comparison_mode = st.selectbox("ENSO comparison", ["climatology", "neutral"], index=0)

    with st.spinner("Loading data and cached results..."):
        state = load_state(
            validation=validation,
            config_file=config_file,
            save_path=save_path,
            model_suffix=(model_suffix if model_suffix.strip() else None),
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Embedding",
            "Prototypes",
            "Seasonal",
            "ENSO",
            "Lagged Heatmaps",
            "Temporal Evolution",
        ]
    )

    with tab1:
        render_embedding_space(state)
    with tab2:
        render_cluster_prototypes(state)
    with tab3:
        render_seasonal(state)
    with tab4:
        render_enso(state, comparison_mode=comparison_mode)
    with tab5:
        render_lagged_heatmaps(state, comparison_mode=comparison_mode)
    with tab6:
        render_temporal_evolution(state)


if __name__ == "__main__":
    main()
