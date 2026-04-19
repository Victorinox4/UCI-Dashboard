# RetailIQ — Customer Intelligence Dashboard

An executive-grade **Streamlit** dashboard for e-commerce customer analytics, built around the UCI Online Retail dataset structure.

## Features

| Section | What's inside |
|---|---|
| **KPI Row** | Revenue, unique customers, AOV, churn rate, orders, revenue/customer |
| **Revenue & Trends** | Monthly revenue + orders dual-axis, day-of-week breakdown, quarterly cohort stacking, order-value distribution |
| **Customer Segments** | RFM-based segmentation donut, churn vs revenue scatter, performance table |
| **Geo & Category** | Country revenue bar, category pie, segment×category heatmap, AOV by category |
| **RFM Analysis** | Scatter plot (frequency × monetary), recency/monetary histograms, segment radar chart |

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Launch
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Stack
- **Streamlit** — UI framework
- **Plotly** — interactive charts
- **Pandas / NumPy** — data wrangling
- Data is generated synthetically from UCI Online Retail schema (18 000 transactions, Jan 2023 – Jun 2024)

## Customisation
To swap in real data, replace the `load_data()` function in `app.py` with a CSV/database loader:

```python
@st.cache_data
def load_data():
    df = pd.read_csv("your_data.csv", parse_dates=["InvoiceDate"])
    # ensure columns: CustomerID, Country, Segment, Category,
    #                 UnitPrice, Quantity, Revenue, Churned
    ...
    return df, rfm
```
