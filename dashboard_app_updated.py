import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ · Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  :root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #c8f135;
    --accent2:   #5b8def;
    --accent3:   #f13566;
    --text:      #e8e8f0;
    --muted:     #6b6b8a;
    --card-bg:   #13131d;
  }

  /* Global reset */
  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Main content area */
  .main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
  }
  [data-testid="metric-container"]:hover { border-color: var(--accent); }

  [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--accent) !important;
    white-space: nowrap;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted) !important;
  }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

  /* Section headers */
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  /* Dividers */
  hr { border-color: var(--border) !important; margin: 1.5rem 0; }

  /* Plotly chart containers */
  .stPlotlyChart { border: 1px solid var(--border); border-radius: 4px; }

  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
  }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .logo-accent { color: #c8f135; }

  .section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 1rem;
    display: block;
  }
  .tag {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 2px;
    padding: 2px 8px;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c8f135;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Data generation (based on UCI Online Retail dataset structure) ────────────

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('Online Retail.csv')
    
    # Basic Cleaning
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)] # Remove cancellations
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Add Core Columns
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['Month'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    df['Quarter'] = df['InvoiceDate'].apply(lambda d: f"Q{((d.month-1)//3)+1} {d.year}")
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    
    # Infer Categories (Simple mapping for demo)
    categories = ["Home & Garden", "Gifts", "Fashion", "Beauty", "Kids", "Accessories"]
    np.random.seed(42)
    unique_stock = df['StockCode'].unique()
    stock_to_cat = {code: np.random.choice(categories) for code in unique_stock}
    df['Category'] = df['StockCode'].map(stock_to_cat)
    
    # RFM Scoring & Segmentation
    latest_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Scores (1-4)
    rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['RFM_Total'] = rfm['R'].astype(int) + rfm['F'].astype(int) + rfm['M'].astype(int)
    
    def get_segment(score):
        if score >= 10: return "Champions"
        if score >= 8:  return "Loyal Customers"
        if score >= 6:  return "Potential Loyalists"
        if score >= 4:  return "At Risk"
        return "Lost"
        
    rfm['Segment'] = rfm['RFM_Total'].apply(get_segment)
    
    # Merge Segment back into Main DF
    df = df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID', how='left')
    
    # Churn Prediction (Heuristic: Recency > 90 days)
    df['Churned'] = df['CustomerID'].map(dict(zip(rfm['CustomerID'], (rfm['Recency'] > 90).astype(int))))

    return df, rfm


df, rfm = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">◈ Retail<span class="logo-accent">IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;letter-spacing:0.1em;color:#6b6b8a;margin-bottom:1.5rem;">CUSTOMER INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<span class="section-label">📅 Time Range</span>', unsafe_allow_html=True)
    months = sorted(df["Month"].unique())
    month_range = st.select_slider("", options=months, value=(months[0], months[-1]), label_visibility="collapsed")

    st.divider()
    st.markdown('<span class="section-label">🌍 Country</span>', unsafe_allow_html=True)
    countries_avail = ["All"] + sorted(df["Country"].unique())
    selected_country = st.selectbox("", countries_avail, label_visibility="collapsed")

    st.divider()
    st.markdown('<span class="section-label">🏷 Segment</span>', unsafe_allow_html=True)
    segments_avail = ["All"] + list(df["Segment"].unique())
    selected_seg = st.selectbox("", segments_avail, label_visibility="collapsed")

    st.divider()
    st.markdown('<span class="section-label">📦 Category</span>', unsafe_allow_html=True)
    cats_avail = ["All"] + sorted(df["Category"].unique())
    selected_cat = st.selectbox("", cats_avail, label_visibility="collapsed")

    st.divider()
    st.caption(f"Dataset: {len(df):,} transactions · {df['Month'].min()} – {df['Month'].max()}")
    st.caption("Source: UCI Online Retail")


# ── Filter data ───────────────────────────────────────────────────────────────
fdf = df[df["Month"].between(month_range[0], month_range[1])]
if selected_country != "All":
    fdf = fdf[fdf["Country"] == selected_country]
if selected_seg != "All":
    fdf = fdf[fdf["Segment"] == selected_seg]
if selected_cat != "All":
    fdf = fdf[fdf["Category"] == selected_cat]

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#c8f135", "#5b8def", "#f13566", "#f5a623", "#a78bfa", "#34d399", "#fb923c"]
SEGMENT_COLORS = {
    "Champions":          "#c8f135",
    "Loyal Customers":    "#5b8def",
    "Potential Loyalists":"#f5a623",
    "At Risk":            "#f13566",
    "Lost":               "#6b6b8a",
}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#e8e8f0", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
)


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<span class="tag">Executive Dashboard · v2.1</span>', unsafe_allow_html=True)
    st.markdown("# UCI Customer Analytics")
    st.markdown(f'<span style="color:#6b6b8a;font-size:0.8rem;">{month_range[0]} → {month_range[1]} &nbsp;·&nbsp; {len(fdf):,} transactions filtered</span>', unsafe_allow_html=True)
with col_h2:
    now = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f'<div style="text-align:right;color:#6b6b8a;font-size:0.72rem;margin-top:2rem;">{now}</div>', unsafe_allow_html=True)

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total_rev      = fdf["Revenue"].sum()
unique_custs   = fdf["CustomerID"].nunique()
avg_order_val  = fdf["Revenue"].mean()
churn_rate     = fdf["Churned"].mean() * 100
total_orders   = fdf["InvoiceNo"].nunique()
rev_per_cust   = total_rev / unique_custs if unique_custs else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Revenue",     f"£{total_rev/1e6:.2f}M")
k2.metric("Unique Customers",  f"{unique_custs:,}")
k3.metric("Avg. Order Value",  f"£{avg_order_val:.0f}")
k4.metric("Churn Rate",        f"{churn_rate:.1f}%")
k5.metric("Total Orders",      f"{total_orders:,}")
k6.metric("Rev / Customer",    f"£{rev_per_cust:.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈  Revenue & Trends", "👥  Customer Segments", "🌍  Geo & Category", "🔬  RFM Analysis"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Revenue & Trends
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<span class="section-label">Monthly Revenue & Orders</span>', unsafe_allow_html=True)
        monthly = fdf.groupby("Month").agg(Revenue=("Revenue","sum"), Orders=("InvoiceNo","nunique")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Revenue"],
            name="Revenue", marker_color="#c8f135", opacity=0.85,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Orders"],
            name="Orders", line=dict(color="#5b8def", width=2),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig.update_layout(**CHART_LAYOUT, height=320, legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(title_text="Revenue (£)", secondary_y=False, tickprefix="£", gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="Orders", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<span class="section-label">Revenue by Day of Week</span>', unsafe_allow_html=True)
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = fdf.groupby("DayOfWeek")["Revenue"].sum().reindex(dow_order).reset_index()
        fig2 = px.bar(dow, x="Revenue", y="DayOfWeek", orientation="h",
                      color="Revenue", color_continuous_scale=["#1e1e2e","#c8f135"])
        layout = {
            **CHART_LAYOUT,
            "height": 340,
            "xaxis": {**CHART_LAYOUT.get("xaxis", {}), "tickformat": ".0%", "title": "Churn Risk"},
            "yaxis": {**CHART_LAYOUT.get("yaxis", {}), "tickprefix": "£", "title": "Total Revenue"},
            "showlegend": False}
        fig2.update_layout(**layout)
        fig2.update_layout(**CHART_LAYOUT, height=340, showlegend=False)

        fig2.update_xaxes(tickformat=".0%", title="Churn Risk")
        fig2.update_yaxes(tickprefix="£", title="Total Revenue")
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<span class="section-label">Quarterly Cohort Revenue</span>', unsafe_allow_html=True)
        qtr = fdf.groupby(["Quarter","Segment"])["Revenue"].sum().reset_index()
        fig3 = px.bar(qtr, x="Quarter", y="Revenue", color="Segment",
                      color_discrete_map=SEGMENT_COLORS, barmode="stack")
        fig3.update_layout(**CHART_LAYOUT, height=300, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<span class="section-label">Revenue Distribution</span>', unsafe_allow_html=True)
        fig4 = px.histogram(fdf[fdf["Revenue"] < fdf["Revenue"].quantile(0.97)],
                            x="Revenue", nbins=60, color_discrete_sequence=["#5b8def"])
        fig4.update_layout(**CHART_LAYOUT, height=300)
        fig4.update_traces(marker_line_width=0, opacity=0.8)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Customer Segments
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<span class="section-label">Segment Distribution</span>', unsafe_allow_html=True)
        seg_data = fdf.groupby("Segment").agg(
            Customers=("CustomerID","nunique"),
            Revenue  =("Revenue","sum"),
            Churn    =("Churned","mean"),
        ).reset_index()
        fig = px.pie(seg_data, names="Segment", values="Customers",
                     color="Segment", color_discrete_map=SEGMENT_COLORS,
                     hole=0.55)
        fig.update_layout(**CHART_LAYOUT, height=340,
                          legend=dict(orientation="h", y=-0.1))
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0a0a0f", width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<span class="section-label">Revenue vs Churn by Segment</span>', unsafe_allow_html=True)
        fig2.update_layout(**{
            **CHART_LAYOUT,
            "height": 340,
            "xaxis": dict(tickformat=".0%", title="Churn Risk", gridcolor="#1e1e2e"),
            "yaxis": dict(tickprefix="£", title="Total Revenue", gridcolor="#1e1e2e"),
            "showlegend": False,})
        fig2 = px.scatter(seg_data, x="Churn", y="Revenue",
                          size="Customers", color="Segment",
                          color_discrete_map=SEGMENT_COLORS,
                          text="Segment", size_max=60)
        fig2.update_traces(textposition="top center", marker_line_width=0)
        fig2.update_layout(**CHART_LAYOUT, height=340, showlegend=False)
        fig2.update_xaxes(tickformat=".0%", title="Churn Risk")
        fig2.update_yaxes(tickprefix="£", title="Total Revenue")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<span class="section-label">Segment Performance Table</span>', unsafe_allow_html=True)
    seg_table = fdf.groupby("Segment").agg(
        Customers   =("CustomerID","nunique"),
        Orders      =("Revenue","count"),
        Revenue     =("Revenue","sum"),
        Avg_Order   =("Revenue","mean"),
        Churn_Risk  =("Churned","mean"),
    ).reset_index()
    seg_table["Revenue"]    = seg_table["Revenue"].apply(lambda x: f"£{x:,.0f}")
    seg_table["Avg_Order"]  = seg_table["Avg_Order"].apply(lambda x: f"£{x:.2f}")
    seg_table["Churn_Risk"] = seg_table["Churn_Risk"].apply(lambda x: f"{x*100:.1f}%")
    seg_table.columns       = ["Segment","Customers","Orders","Revenue","Avg Order","Churn Risk"]
    st.dataframe(seg_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Geo & Category
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<span class="section-label">Revenue by Country</span>', unsafe_allow_html=True)
        geo = fdf.groupby("Country")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        fig = px.bar(geo.head(12), x="Country", y="Revenue",
                     color="Revenue", color_continuous_scale=["#1e2a3a","#5b8def","#c8f135"])
        fig.update_layout(**{
            **CHART_LAYOUT,
            "height": 340,
            "xaxis": dict(tickformat=".0%", title="Country", gridcolor="#1e1e2e"),
            "yaxis": dict(tickprefix="£", title="Revenue", gridcolor="#1e1e2e"),
            "showlegend": False,})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<span class="section-label">Category Revenue Share</span>', unsafe_allow_html=True)
        cat = fdf.groupby("Category")["Revenue"].sum().reset_index()
        fig2 = px.pie(cat, names="Category", values="Revenue",
                      color_discrete_sequence=PALETTE, hole=0.4)
        fig2.update_layout(**CHART_LAYOUT, height=340, legend=dict(orientation="h", y=-0.15))
        fig2.update_traces(marker=dict(line=dict(color="#0a0a0f", width=2)))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<span class="section-label">Category × Segment Heatmap</span>', unsafe_allow_html=True)
        heat = fdf.groupby(["Category","Segment"])["Revenue"].sum().reset_index()
        heat_pivot = heat.pivot(index="Category", columns="Segment", values="Revenue").fillna(0)
        fig3 = px.imshow(heat_pivot, color_continuous_scale=["#0a0a0f","#1e2a3a","#c8f135"],
                         aspect="auto", text_auto=".2s")
        fig3.update_layout(**CHART_LAYOUT, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<span class="section-label">Avg Order Value by Category</span>', unsafe_allow_html=True)
        aov = fdf.groupby("Category")["Revenue"].mean().reset_index().sort_values("Revenue")
        fig4 = px.bar(aov, x="Revenue", y="Category", orientation="h",
                      color="Revenue", color_continuous_scale=["#1e1e2e","#f13566"])
        fig4.update_layout(**{
            **CHART_LAYOUT,
            "height": 340,
            "xaxis": dict(tickformat=".0%", title="Revenue", gridcolor="#1e1e2e"),
            "yaxis": dict(tickprefix="£", title="Categories", gridcolor="#1e1e2e"),
            "showlegend": False,})
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RFM Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-label">RFM — Recency · Frequency · Monetary Analysis</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<span class="section-label">Customer RFM Scatter (Frequency vs Monetary)</span>', unsafe_allow_html=True)
        # Use RFM dataframe and sample
        seg_lookup = rfm[["CustomerID", "Segment", "Recency"]]
        seg_lookup = fdf.groupby("CustomerID")["Segment"].first().reset_index()
        rfm_plot = rfm.merge(seg_lookup, on="CustomerID", how="left", suffixes=("", "_drop"))
        rfm_plot = rfm_plot.drop(columns=[col for col in rfm_plot.columns if "_drop" in col])
        rfm_sample = seg_lookup.sample(min(1500, len(seg_lookup)), random_state=1)
        rfm_sample = rfm_sample.merge(rfm[['CustomerID', 'Frequency', 'Monetary', 'Recency']], on='CustomerID')
        fig = px.scatter(
            rfm_sample,
            x="Frequency",
            y="Monetary",
            color="Segment",
            opacity=0.7,
            hover_data={"CustomerID": True, "Recency": True})
        fig.update_layout(**{**CHART_LAYOUT, "yaxis": {**CHART_LAYOUT.get("yaxis", {}), "tickprefix": "£"}},
                            height=380,
                            legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<span class="section-label">Recency Distribution</span>', unsafe_allow_html=True)
        fig2 = px.histogram(rfm, x="Recency", nbins=40, color_discrete_sequence=["#f13566"])
        fig2.update_layout(**{
            **CHART_LAYOUT,
            "height": 340,
            "xaxis": dict(tickformat=".0%", title="Recency", gridcolor="#1e1e2e"),
            "yaxis": dict(tickprefix="£", title="Percentage", gridcolor="#1e1e2e"),
            "showlegend": False,})
        fig2.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<span class="section-label">Monetary Distribution</span>', unsafe_allow_html=True)
        rfm_clipped = rfm[rfm["Monetary"] < rfm["Monetary"].quantile(0.95)]
        fig3 = px.histogram(rfm_clipped, x="Monetary", nbins=40, color_discrete_sequence=["#c8f135"])
        fig3.update_layout(**{**CHART_LAYOUT, "xaxis": dict(tickprefix="£")},
                            height=180)
        fig3.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<span class="section-label">Avg Recency by Segment (days since last purchase)</span>', unsafe_allow_html=True)
        rfm_seg = rfm_plot.groupby("Segment").agg(
            Avg_Recency  =("Recency","mean"),
            Avg_Frequency=("Frequency","mean"),
            Avg_Monetary =("Monetary","mean"),
        ).reset_index()
        fig4 = px.bar(rfm_seg.sort_values("Avg_Recency"), x="Avg_Recency", y="Segment",
                      orientation="h", color="Segment", color_discrete_map=SEGMENT_COLORS)
        fig4.update_layout(
            height=280,
            showlegend=False,
            xaxis=dict(title="Days since last purchase"),
            **{k: v for k, v in CHART_LAYOUT.items() if k != "xaxis"}
        )
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown('<span class="section-label">RFM Radar — Segment Profiles</span>', unsafe_allow_html=True)
        # Normalise to 0-1 for radar
        metrics = ["Avg_Recency","Avg_Frequency","Avg_Monetary"]
        for m in metrics:
            rfm_seg[m+"_n"] = 1 - (rfm_seg[m] - rfm_seg[m].min()) / (rfm_seg[m].max() - rfm_seg[m].min() + 1e-9)
        rfm_seg["Avg_Frequency_n"] = (rfm_seg["Avg_Frequency"] - rfm_seg["Avg_Frequency"].min()) / (rfm_seg["Avg_Frequency"].max() - rfm_seg["Avg_Frequency"].min() + 1e-9)
        rfm_seg["Avg_Monetary_n"]  = (rfm_seg["Avg_Monetary"]  - rfm_seg["Avg_Monetary"].min())  / (rfm_seg["Avg_Monetary"].max()  - rfm_seg["Avg_Monetary"].min()  + 1e-9)

        categories_radar = ["Recency Score","Frequency","Monetary"]
        fig5 = go.Figure()
        for _, row in rfm_seg.iterrows():
            vals = [row["Avg_Recency_n"], row["Avg_Frequency_n"], row["Avg_Monetary_n"]]
            vals += vals[:1]
            fig5.add_trace(go.Scatterpolar(
                r=vals, theta=categories_radar+[categories_radar[0]],
                name=row["Segment"], fill="toself", opacity=0.5,
                line=dict(color=SEGMENT_COLORS.get(row["Segment"],"#c8f135"), width=2),
            ))
        fig5.update_layout(**CHART_LAYOUT, height=280,
                           polar=dict(
                               bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e1e2e", linecolor="#1e1e2e"),
                               angularaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
                           ),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig5, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#2a2a4e;font-size:0.65rem;letter-spacing:0.15em;">'
    'RETAILIQ CUSTOMER INTELLIGENCE · CONFIDENTIAL · FOR EXECUTIVE USE ONLY'
    '</div>',
    unsafe_allow_html=True
)
