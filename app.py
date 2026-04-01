"""
Climate Clustering for ENSO Teleconnections – Dashboard
Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Climate Clustering Dashboard",
    page_icon="🌡",
    layout="wide",
)

st.title("Climate Clustering for ENSO Teleconnections")
st.markdown("""
This dashboard explores climate regime clustering and their relationship to
El Niño–Southern Oscillation (ENSO) events over Brazil.

---

### Pages

| Page | Description |
|------|-------------|
| **1 · Climate Regimes** | Top-5 samples per cluster, monthly frequency, Q-Q plot |
| **2 · ENSO Analysis** | Global anomaly at a chosen lag, anomaly vs all lags, month-lag heatmap |
| **3 · Specific Regimes Analysis** | Month × lag heatmaps and temporal evolution per cluster |

Navigate using the sidebar on the left.

---
""")
