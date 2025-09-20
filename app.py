import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# ---------------- Page setup ----------------
st.set_page_config(page_title="COVID-19 Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Tighter base spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Card look for control panels */
.control-card {
  padding: 1rem 1rem 0.8rem 1rem;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  background: #fbfbfc;
}

/* Smaller help text */
small, .help-text { color: #6c757d; }

/* Section headers */
h2 { margin-top: 0.2rem; }

/* KPI metric spacing */
.kpi .stMetric { text-align: center; }
.kpi .stMetric > div { justify-content: center; }

/* Chart container padding */
.chart-card { padding: .25rem .5rem; }

/* Expander tweak */
.streamlit-expanderHeader { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# loading the csv
@st.cache_data(show_spinner=False)
def load():
    df = pd.read_csv("health.csv")
    df.columns = df.columns.str.strip()
    if "Nb of Covid-19 cases" in df.columns:
        df["Nb of Covid-19 cases"] = pd.to_numeric(df["Nb of Covid-19 cases"], errors="coerce").fillna(0).astype(int)
    for c in [
        "Existence of chronic diseases - Hypertension",
        "Existence of chronic diseases - Cardiovascular disease",
        "Existence of chronic diseases - Diabetes",
        "Existence of chronic diseases - does not exist",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

raw = load()
area_col = "refArea" if "refArea" in raw.columns else st.stop()

# title
st.title("COVID-19 Data — Lebanon")
st.caption("Explore regional data and how chronic disease prevalence relates to COVID-19 cases. Use the controls next to each chart, then click **Apply** to update.")

st.divider()

# Section A: Bar chart — Totals by Governorate/District
st.header("Bar Chart: Total COVID-19 Cases by Governorate/District")

ctrl, viz = st.columns([1.05, 2.95], gap="large")

with ctrl:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Controls")
    with st.form("bar_form", clear_on_submit=False):
        metric_mode = st.radio(
            "Metric",
            options=["Total cases", "Percent of national total"],
            index=0,
            help="Switch between absolute totals and share of national cases."
        )
        query = st.text_input("Search area (contains…)", value="")
        all_areas = sorted(raw[area_col].dropna().unique().tolist())
        max_n = min(25, len(all_areas)) if len(all_areas) > 0 else 10
        top_n = st.slider(
            "Show Top-N areas",
            min_value=5, max_value=max(5, max_n),
            value=min(15, max_n), step=1
        )
        descending = st.toggle("Sort descending", value=True)
        bar_submit = st.form_submit_button("Apply filters")
    st.markdown("</div>", unsafe_allow_html=True)


agg = (
    raw.groupby(area_col, dropna=False)["Nb of Covid-19 cases"]
    .sum()
    .reset_index()
    .rename(columns={"Nb of Covid-19 cases": "total_cases"})
)
total_sum = agg["total_cases"].sum()
agg["pct_of_total"] = (agg["total_cases"] / total_sum * 100).round(2) if total_sum > 0 else 0.0

if query:
    q = query.strip().lower()
    agg = agg[agg[area_col].str.lower().str.contains(q, na=False)]

y_col = "total_cases" if metric_mode == "Total cases" else "pct_of_total"
y_label = "Total Covid-19 cases" if y_col == "total_cases" else "% of national total"

agg = agg.sort_values(y_col, ascending=not descending).head(top_n)

with viz:
    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        st.metric("Areas shown", len(agg))
    with k2:
        shown_sum = agg["total_cases"].sum()
        st.metric("Cases in view", f"{shown_sum:,}")
    with k3:
        share = (shown_sum / total_sum * 100) if total_sum else 0
        st.metric("Share of national total", f"{share:.1f}%")

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    if agg.empty:
        st.info("No areas match your filters. Try clearing the search or increasing Top-N.")
    else:
        fig = px.bar(
            agg, x=area_col, y=y_col, text=y_col,
            labels={area_col: "Governorate / District", y_col: y_label}
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        fig.update_layout(
            xaxis_tickangle=-45, yaxis_title=y_label, xaxis_title="Governorate / District",
            margin=dict(t=10, r=10, l=10, b=60), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander(" Notes & Quick Insights (bar)"):
    if not agg.empty:
        top_row = agg.iloc[0]
        st.markdown(
            f"- **Top hotspot:** `{top_row[area_col]}` leads with **{top_row['total_cases']:,}** cases "
            f"(**{top_row['pct_of_total']:.2f}%** of the national total)."
        )
        st.markdown("- The distribution is highly uneven: a few governorates account for most cases.")
        st.markdown("- Switching to *percent of total* highlights smaller areas with notable relative share.")
        st.markdown("- Adjust **Top-N** and **Search** to see how rankings shift across regions.")

st.divider()

# Section B: Scatter — Chronic disease % vs Total cases
st.header("Scatter: % with ≥1 Chronic Disease vs Total COVID-19 Cases")

sctrl, sviz = st.columns([1.05, 2.95], gap="large")

with sctrl:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Controls")
    with st.form("scatter_form", clear_on_submit=False):
        pct_min, pct_max = st.slider(
            "Chronic % range", 0.0, 100.0, (60.0, 100.0), 1.0,
            help="Show governorates whose % of records with ≥1 chronic disease falls in this range."
        )
        min_cases = st.number_input(
            "Minimum total cases", min_value=0, value=0, step=1000,
            help="Hide governorates below this total-cases threshold."
        )
        size_mode = st.radio("Bubble size", ["By records", "Uniform"], index=0)
        log_x = st.checkbox("Log-scale x-axis (cases)", value=False)
        show_trend = st.checkbox("Show linear trendline", value=False)
        scat_submit = st.form_submit_button("Apply filters")
    st.markdown("</div>", unsafe_allow_html=True)

hy   = "Existence of chronic diseases - Hypertension"
card = "Existence of chronic diseases - Cardiovascular disease"
dia  = "Existence of chronic diseases - Diabetes"

tmp = raw.copy()
tmp["any_chronic"] = (
    (tmp.get(hy, 0).fillna(0).astype(int) == 1) |
    (tmp.get(card, 0).fillna(0).astype(int) == 1) |
    (tmp.get(dia, 0).fillna(0).astype(int) == 1)
).astype(int)

gov_df = (
    tmp.groupby(area_col, dropna=False)
       .agg(total_cases=("Nb of Covid-19 cases","sum"),
            records=("Nb of Covid-19 cases","size"),
            chronic_count=("any_chronic","sum"))
       .reset_index()
)
gov_df["chronic_pct"] = (gov_df["chronic_count"] / gov_df["records"] * 100).round(2)

mask = (
    (gov_df["chronic_pct"].between(pct_min, pct_max, inclusive="both")) &
    (gov_df["total_cases"] >= min_cases)
)
gov_filtered = gov_df[mask].sort_values("total_cases", ascending=False)

with sviz:
    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        st.metric("Governorates shown", len(gov_filtered))
    with k2:
        st.metric("Min cases threshold", f"{min_cases:,}")
    with k3:
        rng = f"{pct_min:.0f}%–{pct_max:.0f}%"
        st.metric("Chronic % range", rng)

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    if gov_filtered.empty:
        st.info("No governorates match the scatter filters. Loosen the ranges to see points.")
    else:
        fig_sc = px.scatter(
            gov_filtered, x="total_cases", y="chronic_pct",
            size=("records" if size_mode == "By records" else None),
            hover_name=area_col,
            labels={"total_cases":"Total Covid-19 cases",
                    "chronic_pct":"% records with ≥1 chronic disease"},
        )
        if show_trend and len(gov_filtered) >= 2:
            x = gov_filtered["total_cases"].to_numpy()
            y = gov_filtered["chronic_pct"].to_numpy()
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100); yy = m*xx + b
            fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Trendline"))
        fig_sc.update_layout(
            xaxis_title="Total Covid-19 cases",
            yaxis_title="% records with ≥1 chronic disease",
            margin=dict(t=10, r=10, l=10, b=60),
        )
        if log_x:
            fig_sc.update_xaxes(type="log")
        st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander(" Notes & Quick Insights (scatter)"):
    if not gov_filtered.empty:
        corr = gov_filtered["chronic_pct"].corr(gov_filtered["total_cases"])
        st.markdown(
            f"- **Coverage:** {len(gov_filtered)} governorates shown after filters "
            f"(min cases ≥ {min_cases:,}, chronic % in [{pct_min:.0f}–{pct_max:.0f}])."
        )
        st.markdown("- **Bubble size** reflects number of records; switch to Uniform choice for position-only comparison.")
        st.markdown("- Tighten the **chronic % range** to focus on outliers or consistent clusters.")
