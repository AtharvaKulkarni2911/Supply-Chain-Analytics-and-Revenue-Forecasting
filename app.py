"""
Supply Chain Analytics & Revenue Forecasting
Capstone 2 — Streamlit Web App
Author: Atharva
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
        padding: 12px 0 8px 0;
        border-bottom: 2px solid #2563eb;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .insight-box {
        background: #0f172a;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#7C3AED", "#0891B2"]

def fmt_inr(val):
    return f"R${val:,.0f}"

def safe_read(fname):
    """Read CSV if it exists, else return empty DataFrame."""
    if os.path.exists(fname):
        return pd.read_csv(fname)
    return pd.DataFrame()

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_all_data():
    # Try loading cleaned files first; fall back to final_ files
    def try_load(names):
        for n in names:
            df = safe_read(n)
            if not df.empty:
                return df
        return pd.DataFrame()

    orders    = try_load(["cleaned_orders.csv",     "final_orders.csv"])
    items     = try_load(["cleaned_order_items.csv","final_order_items.csv"])
    customers = try_load(["cleaned_customers.csv",  "final_customers.csv"])
    payments  = try_load(["cleaned_payments.csv",   "final_payments.csv"])
    products  = try_load(["cleaned_products.csv",   "final_products.csv"])
    monthly   = try_load(["monthly_revenue.csv"])
    master    = try_load(["master_dataset.csv"])

    # ── Date parsing ─────────────────────────────────────────────
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_timestamp", "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    if "ds" in monthly.columns:
        monthly["ds"] = pd.to_datetime(monthly["ds"], errors="coerce")

    # ── If master is missing, rebuild it ────────────────────────
    if master.empty and not orders.empty and not items.empty:
        st.info("master_dataset.csv not found — rebuilding from component files...")
        pay_agg = pd.DataFrame()
        if not payments.empty:
            pay_agg = (
                payments.groupby("order_id")
                .agg(
                    payment_value=("payment_value", "sum"),
                    payment_type=("payment_type", lambda x: x.mode()[0] if len(x) else "unknown"),
                    payment_installments=("payment_installments", "max"),
                )
                .reset_index()
            )

        master = orders.copy()
        if not customers.empty:
            master = master.merge(customers, on="customer_id", how="left")
        if not items.empty:
            master = master.merge(items, on="order_id", how="left")
        if not products.empty and "product_id" in master.columns:
            master = master.merge(products, on="product_id", how="left")
        if not pay_agg.empty:
            master = master.merge(pay_agg, on="order_id", how="left")

        # Re-parse dates after merge
        for col in date_cols:
            if col in master.columns:
                master[col] = pd.to_datetime(master[col], errors="coerce")

        # Derived columns
        if "order_purchase_timestamp" in master.columns:
            master["order_year"]        = master["order_purchase_timestamp"].dt.year
            master["order_month"]       = master["order_purchase_timestamp"].dt.month
            master["order_quarter"]     = master["order_purchase_timestamp"].dt.quarter
            master["order_day_of_week"] = master["order_purchase_timestamp"].dt.dayofweek

        if "order_delivered_timestamp" in master.columns and "order_purchase_timestamp" in master.columns:
            master["delivery_days"] = (
                (master["order_delivered_timestamp"] - master["order_purchase_timestamp"])
                .dt.total_seconds() / 86400
            ).round(1)
            master["on_time_delivery"] = (
                master["order_delivered_timestamp"] <= master["order_estimated_delivery_date"]
            ).astype("Int64")

        if "order_purchase_timestamp" in master.columns:
            master["approval_lag_hrs"] = (
                (master["order_approved_at"] - master["order_purchase_timestamp"])
                .dt.total_seconds() / 3600
            ).round(2)

        if "order_status" in master.columns:
            master["is_delivered"] = (master["order_status"] == "delivered").astype(int)
            master["is_canceled"]  = (master["order_status"] == "canceled").astype(int)

        if "price" in master.columns and "shipping_charges" in master.columns:
            master["item_revenue"] = master["price"] + master["shipping_charges"]

    # ── Rebuild monthly_revenue if missing ───────────────────────
    if monthly.empty and not master.empty:
        delivered_mask = master.get("order_status", pd.Series()) == "delivered"
        tmp = master[delivered_mask].copy()
        if "order_year" in tmp.columns and "order_month" in tmp.columns and "item_revenue" in tmp.columns:
            monthly = (
                tmp.groupby(["order_year", "order_month"])
                .agg(
                    y             =("item_revenue",  "sum"),
                    total_orders  =("order_id",      "nunique"),
                    avg_order_value=("item_revenue", "mean"),
                    total_customers=("customer_id",  "nunique"),
                )
                .reset_index()
            )
            monthly["ds"] = pd.to_datetime(
                monthly["order_year"].astype(str) + "-" +
                monthly["order_month"].astype(str).str.zfill(2) + "-01"
            )
            monthly = monthly.rename(columns={"y": "y"}).sort_values("ds").reset_index(drop=True)

    return orders, items, customers, payments, products, monthly, master


orders, items, customers, payments, products, monthly_rev, master = load_all_data()

# Guard: stop if no data at all
if master.empty and orders.empty:
    st.error(
        "❌ No data files found. Please ensure at least `cleaned_orders.csv`, "
        "`cleaned_order_items.csv`, `cleaned_customers.csv`, `cleaned_payments.csv`, "
        "and `cleaned_products.csv` are in the same directory as `app.py`."
    )
    st.stop()

df = master if not master.empty else orders

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Supply Chain Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Revenue Trends", "🗂️ Category Analysis",
         "🚚 Delivery Analysis", "🔮 Revenue Forecast", "🤖 ML Metrics"],
        index=0,
    )

    st.markdown("---")

    # Year filter
    if "order_year" in df.columns:
        years = sorted(df["order_year"].dropna().unique().astype(int))
        sel_years = st.multiselect("Filter by Year", years, default=years)
        if sel_years:
            df = df[df["order_year"].isin(sel_years)]
            if not monthly_rev.empty and "ds" in monthly_rev.columns:
                monthly_rev = monthly_rev[monthly_rev["ds"].dt.year.isin(sel_years)]

    st.markdown("---")
    st.caption("Capstone 2 — Supply Chain Analytics & Revenue Forecasting")

# ─────────────────────────────────────────────
# ══ PAGE 1 — OVERVIEW ══════════════════════
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📦 Supply Chain Analytics Dashboard")
    st.markdown("End-to-end analytics across orders, customers, products, and revenue.")
    st.markdown("---")

    delivered = df[df.get("order_status", pd.Series()) == "delivered"] if "order_status" in df.columns else df

    # KPI cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpi_data = [
        (col1, "Total Orders",    f"{df['order_id'].nunique():,}"              if "order_id"      in df.columns else "N/A", "📦"),
        (col2, "Total Revenue",   f"R${df['item_revenue'].sum():,.0f}"         if "item_revenue"  in df.columns else "N/A", "💰"),
        (col3, "Avg Order Value", f"R${df['item_revenue'].mean():,.2f}"        if "item_revenue"  in df.columns else "N/A", "🛒"),
        (col4, "Total Customers", f"{df['customer_id'].nunique():,}"           if "customer_id"   in df.columns else "N/A", "👥"),
        (col5, "Delivery Rate",   f"{delivered.shape[0]/max(df.shape[0],1)*100:.1f}%" if not delivered.empty else "N/A", "✅"),
        (col6, "Avg Delivery",    f"{delivered['delivery_days'].mean():.1f}d"  if "delivery_days" in delivered.columns else "N/A", "🚚"),
    ]
    for col, label, value, icon in kpi_data:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{icon}<br>{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Order status distribution
    with col_a:
        st.markdown('<div class="section-header">Order Status Distribution</div>', unsafe_allow_html=True)
        if "order_status" in df.columns:
            status_counts = df["order_status"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            wedges, texts, autotexts = ax.pie(
                status_counts.values,
                labels=status_counts.index,
                autopct="%1.1f%%",
                colors=COLORS[:len(status_counts)],
                startangle=90,
            )
            for t in texts + autotexts:
                t.set_color("white")
                t.set_fontsize(9)
            ax.set_title("Order Status", color="white", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

    # Payment type distribution
    with col_b:
        st.markdown('<div class="section-header">Payment Type Distribution</div>', unsafe_allow_html=True)
        pay_col = "payment_type" if "payment_type" in df.columns else None
        if pay_col:
            pay_counts = df[pay_col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            bars = ax.barh(pay_counts.index, pay_counts.values, color=COLORS[:len(pay_counts)])
            ax.set_title("Payment Types", color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#334155")
            ax.spines["left"].set_color("#334155")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar in bars:
                ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                        f"{int(bar.get_width()):,}", va="center", color="white", fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

    # Key insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insights = []
    if "order_status" in df.columns:
        top_status = df["order_status"].value_counts().index[0]
        top_pct    = df["order_status"].value_counts(normalize=True).iloc[0] * 100
        insights.append(f"🟢 <b>{top_status.title()}</b> is the most common order status at <b>{top_pct:.1f}%</b>")
    if "product_category_name" in df.columns:
        top_cat = df["product_category_name"].value_counts().index[0]
        insights.append(f"🛍️ Top product category: <b>{top_cat.replace('_',' ').title()}</b>")
    if "customer_state" in df.columns:
        top_state = df["customer_state"].value_counts().index[0]
        insights.append(f"📍 Most orders from state: <b>{top_state}</b>")
    if "delivery_days" in delivered.columns:
        avg_del = delivered["delivery_days"].mean()
        insights.append(f"🚚 Average delivery time: <b>{avg_del:.1f} days</b>")

    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ══ PAGE 2 — REVENUE TRENDS ════════════════
# ─────────────────────────────────────────────
elif page == "📈 Revenue Trends":
    st.title("📈 Revenue Trends")
    st.markdown("---")

    if monthly_rev.empty or "ds" not in monthly_rev.columns:
        st.warning("Monthly revenue data not available.")
        st.stop()

    # Monthly revenue trend
    st.markdown('<div class="section-header">Monthly Revenue Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(monthly_rev["ds"], monthly_rev["y"], color=COLORS[0], linewidth=2, marker="o", markersize=3)
    ax.fill_between(monthly_rev["ds"], monthly_rev["y"], alpha=0.15, color=COLORS[0])
    ax.set_title("Monthly Revenue Trend", color="white", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    # Yearly summary
    st.markdown('<div class="section-header">Yearly Revenue Summary</div>', unsafe_allow_html=True)
    if "order_year" in df.columns and "item_revenue" in df.columns:
        delivered_df = df[df.get("order_status", pd.Series("delivered")) == "delivered"] if "order_status" in df.columns else df
        yearly = (
            delivered_df.groupby("order_year")
            .agg(total_revenue=("item_revenue","sum"), total_orders=("order_id","nunique"))
            .reset_index()
        )
        yearly["revenue_yoy_pct"] = yearly["total_revenue"].pct_change() * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.bar(yearly["order_year"].astype(str), yearly["total_revenue"], color=COLORS[0])
            ax.set_title("Revenue by Year", color="white", fontsize=11, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.0f}M"))
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_color("#334155")
            st.pyplot(fig); plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            yoy = yearly.dropna(subset=["revenue_yoy_pct"])
            bar_colors = [COLORS[1] if v >= 0 else COLORS[2] for v in yoy["revenue_yoy_pct"]]
            ax.bar(yoy["order_year"].astype(str), yoy["revenue_yoy_pct"], color=bar_colors)
            ax.set_title("YoY Growth %", color="white", fontsize=11, fontweight="bold")
            ax.axhline(0, color="white", linewidth=0.8)
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_color("#334155")
            st.pyplot(fig); plt.close(fig)

        st.dataframe(
            yearly.style.format({
                "total_revenue":"R${:,.0f}",
                "total_orders":"{:,}",
                "revenue_yoy_pct":"{:.1f}%"
            }),
            use_container_width=True
        )

    # Seasonality
    st.markdown('<div class="section-header">Monthly Seasonality</div>', unsafe_allow_html=True)
    if "order_month" in df.columns and "item_revenue" in df.columns:
        delivered_df = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df
        monthly_avg = delivered_df.groupby("order_month")["item_revenue"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_avg["month_name"] = monthly_avg["order_month"].apply(lambda x: month_names[x-1])

        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.bar(monthly_avg["month_name"], monthly_avg["item_revenue"], color=COLORS[0])
        ax.set_title("Average Revenue by Month (Seasonality)", color="white", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────
# ══ PAGE 3 — CATEGORY ANALYSIS ═════════════
# ─────────────────────────────────────────────
elif page == "🗂️ Category Analysis":
    st.title("🗂️ Category Analysis")
    st.markdown("---")

    if "product_category_name" not in df.columns or "item_revenue" not in df.columns:
        st.warning("Category or revenue data not available.")
        st.stop()

    delivered_df = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df
    cat_rev = (
        delivered_df.groupby("product_category_name")
        .agg(total_revenue=("item_revenue","sum"), total_orders=("order_id","nunique"), avg_price=("price","mean"))
        .reset_index().sort_values("total_revenue", ascending=False).head(10)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top 10 by Revenue</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.barh(cat_rev["product_category_name"][::-1], cat_rev["total_revenue"][::-1], color=COLORS[0])
        ax.set_title("Revenue by Category", color="white", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.0f}M"))
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Top 10 by Orders</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.barh(cat_rev["product_category_name"][::-1], cat_rev["total_orders"][::-1], color=COLORS[1])
        ax.set_title("Orders by Category", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">Category Revenue Table</div>', unsafe_allow_html=True)
    st.dataframe(
        cat_rev.rename(columns={
            "product_category_name":"Category",
            "total_revenue":"Revenue (R$)",
            "total_orders":"Orders",
            "avg_price":"Avg Price (R$)"
        }).style.format({"Revenue (R$)":"R${:,.0f}", "Orders":"{:,}", "Avg Price (R$)":"R${:,.2f)"}),
        use_container_width=True
    )


# ─────────────────────────────────────────────
# ══ PAGE 4 — DELIVERY ANALYSIS ═════════════
# ─────────────────────────────────────────────
elif page == "🚚 Delivery Analysis":
    st.title("🚚 Delivery Analysis")
    st.markdown("---")

    if "delivery_days" not in df.columns:
        st.warning("Delivery data not available.")
        st.stop()

    delivered = df[df["order_status"] == "delivered"].copy() if "order_status" in df.columns else df.copy()
    delivered = delivered.dropna(subset=["delivery_days"])

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Avg Delivery Days",   f"{delivered['delivery_days'].mean():.1f} days"),
        (col2, "Min Delivery Days",   f"{delivered['delivery_days'].min():.0f} days"),
        (col3, "Max Delivery Days",   f"{delivered['delivery_days'].max():.0f} days"),
        (col4, "On-Time Rate",
         f"{delivered['on_time_delivery'].mean()*100:.1f}%" if "on_time_delivery" in delivered.columns else "N/A"),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Delivery Days Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        delivered["delivery_days"].hist(bins=30, ax=ax, color=COLORS[0], edgecolor="#0f172a")
        ax.axvline(delivered["delivery_days"].mean(), color=COLORS[2], linestyle="--",
                   label=f"Mean: {delivered['delivery_days'].mean():.1f}d")
        ax.set_title("Delivery Days Distribution", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Slowest States (Avg Days)</div>', unsafe_allow_html=True)
        if "customer_state" in delivered.columns:
            state_del = (
                delivered.groupby("customer_state")["delivery_days"]
                .mean().sort_values(ascending=False).head(10).reset_index()
            )
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.barh(state_del["customer_state"][::-1], state_del["delivery_days"][::-1], color=COLORS[3])
            ax.set_title("Avg Delivery Days by State", color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_color("#334155")
            st.pyplot(fig); plt.close(fig)

    # On-time by category
    if "product_category_name" in delivered.columns and "on_time_delivery" in delivered.columns:
        st.markdown('<div class="section-header">On-Time Delivery by Category</div>', unsafe_allow_html=True)
        cat_otd = (
            delivered.groupby("product_category_name")["on_time_delivery"]
            .mean().sort_values(ascending=False).head(10).reset_index()
        )
        cat_otd["on_time_pct"] = cat_otd["on_time_delivery"] * 100
        fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        bar_colors = [COLORS[1] if v >= 80 else COLORS[3] if v >= 60 else COLORS[2]
                      for v in cat_otd["on_time_pct"]]
        ax.bar(cat_otd["product_category_name"], cat_otd["on_time_pct"], color=bar_colors)
        ax.set_title("On-Time Delivery % by Category", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("On-Time %", color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.axhline(80, color=COLORS[2], linestyle="--", linewidth=1, label="80% target")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────
# ══ PAGE 5 — REVENUE FORECAST ══════════════
# ─────────────────────────────────────────────
elif page == "🔮 Revenue Forecast":
    st.title("🔮 Revenue Forecast — Next 12 Months")
    st.markdown("Facebook Prophet time-series forecasting model.")
    st.markdown("---")

    # Try loading pre-computed forecast
    forecast_file = "forecast_next12months.csv"
    eval_file     = "forecast_evaluation.csv"
    metrics_file  = "model_metrics.csv"

    pre_forecast = safe_read(forecast_file)
    pre_eval     = safe_read(eval_file)
    pre_metrics  = safe_read(metrics_file)

    if not pre_forecast.empty:
        # Show pre-computed results
        if "ds" in pre_forecast.columns:
            pre_forecast["ds"] = pd.to_datetime(pre_forecast["ds"])

        st.markdown('<div class="section-header">Next 12 Months Forecast</div>', unsafe_allow_html=True)

        if not pre_metrics.empty:
            m = pre_metrics.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in [
                (c1, "MAE",       f"R${m.get('MAE',0):,.0f}"),
                (c2, "RMSE",      f"R${m.get('RMSE',0):,.0f}"),
                (c3, "R² Score",  f"{m.get('R2',0):.4f}"),
                (c4, "Accuracy",  f"{m.get('Accuracy',0):.2f}%"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True
                    )
            st.markdown("<br>", unsafe_allow_html=True)

        # Forecast chart
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        if not monthly_rev.empty and "ds" in monthly_rev.columns:
            ax.plot(monthly_rev["ds"], monthly_rev["y"],
                    color=COLORS[0], linewidth=2, label="Historical Revenue")

        ax.plot(pre_forecast["ds"], pre_forecast["yhat"],
                color=COLORS[2], linewidth=2.5, linestyle="--", label="Forecast", zorder=3)
        if "yhat_lower" in pre_forecast.columns:
            ax.fill_between(pre_forecast["ds"], pre_forecast["yhat_lower"], pre_forecast["yhat_upper"],
                            alpha=0.2, color=COLORS[2], label="95% CI")

        if not monthly_rev.empty:
            ax.axvline(monthly_rev["ds"].max(), color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")

        ax.set_title("Revenue Forecast — Next 12 Months", color="white", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close(fig)

        # Forecast table
        st.markdown('<div class="section-header">Monthly Predictions</div>', unsafe_allow_html=True)
        disp = pre_forecast.copy()
        disp["Month"] = disp["ds"].dt.strftime("%b %Y")
        disp = disp.rename(columns={"yhat":"Predicted","yhat_lower":"Lower Bound","yhat_upper":"Upper Bound"})
        st.dataframe(
            disp[["Month","Predicted","Lower Bound","Upper Bound"]]
            .style.format({"Predicted":"R${:,.0f}","Lower Bound":"R${:,.0f}","Upper Bound":"R${:,.0f}"}),
            use_container_width=True
        )

        total_pred = pre_forecast["yhat"].sum()
        st.success(f"🎯 **Total Predicted Revenue (Next 12 Months): R${total_pred:,.0f}**")

    else:
        # Run Prophet live
        st.info("Pre-computed forecast not found. Running Prophet model now — this may take ~30 seconds...")

        if monthly_rev.empty or "ds" not in monthly_rev.columns or "y" not in monthly_rev.columns:
            st.error("Monthly revenue data required to run forecast. Please ensure monthly_revenue.csv is present.")
            st.stop()

        try:
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            prophet_df = monthly_rev[["ds","y"]].copy().dropna()
            prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

            FORECAST_MONTHS = 12
            TEST_MONTHS     = min(12, len(prophet_df) // 4)

            train_df = prophet_df.iloc[:-TEST_MONTHS].copy()
            test_df  = prophet_df.iloc[-TEST_MONTHS:].copy()

            with st.spinner("Training Prophet model..."):
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.1,
                    seasonality_prior_scale=10,
                    interval_width=0.95,
                )
                try:
                    model.add_country_holidays(country_name="BR")
                except Exception:
                    pass
                model.fit(train_df)

            test_future   = model.make_future_dataframe(periods=TEST_MONTHS, freq="MS")
            test_forecast = model.predict(test_future)
            test_pred     = test_forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(TEST_MONTHS).reset_index(drop=True)
            test_actual   = test_df.reset_index(drop=True)

            y_actual = test_actual["y"].values
            y_pred   = test_pred["yhat"].values
            mae      = mean_absolute_error(y_actual, y_pred)
            rmse     = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2       = r2_score(y_actual, y_pred)
            mape     = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

            # Final model on all data
            with st.spinner("Generating 12-month forecast..."):
                final_model = Prophet(
                    yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="multiplicative", changepoint_prior_scale=0.1,
                    seasonality_prior_scale=10, interval_width=0.95,
                )
                try:
                    final_model.add_country_holidays(country_name="BR")
                except Exception:
                    pass
                final_model.fit(prophet_df)
                future   = final_model.make_future_dataframe(periods=FORECAST_MONTHS, freq="MS")
                forecast = final_model.predict(future)

            future_forecast = forecast.tail(FORECAST_MONTHS)[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            future_forecast["yhat"]       = future_forecast["yhat"].clip(lower=0)
            future_forecast["yhat_lower"] = future_forecast["yhat_lower"].clip(lower=0)
            future_forecast["yhat_upper"] = future_forecast["yhat_upper"].clip(lower=0)

            # Metrics
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in [
                (c1, "MAE",       f"R${mae:,.0f}"),
                (c2, "RMSE",      f"R${rmse:,.0f}"),
                (c3, "R² Score",  f"{r2:.4f}"),
                (c4, "Accuracy",  f"{100-mape:.2f}%"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Chart
            fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
            ax.set_facecolor("#0f172a")
            ax.plot(prophet_df["ds"], prophet_df["y"], color=COLORS[0], linewidth=2, label="Historical")
            ax.plot(future_forecast["ds"], future_forecast["yhat"],
                    color=COLORS[2], linewidth=2.5, linestyle="--", label="Forecast")
            ax.fill_between(future_forecast["ds"], future_forecast["yhat_lower"], future_forecast["yhat_upper"],
                            alpha=0.2, color=COLORS[2], label="95% CI")
            ax.axvline(prophet_df["ds"].max(), color="gray", linestyle=":", linewidth=1.5)
            ax.set_title(f"Revenue Forecast | Accuracy: {100-mape:.1f}% | R²: {r2:.4f}",
                         color="white", fontsize=13, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_color("#334155")
            ax.legend(facecolor="#1e293b", labelcolor="white")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig); plt.close(fig)

            # Table
            disp = future_forecast.copy()
            disp["Month"] = disp["ds"].dt.strftime("%b %Y")
            st.dataframe(
                disp[["Month","yhat","yhat_lower","yhat_upper"]]
                .rename(columns={"yhat":"Predicted","yhat_lower":"Lower","yhat_upper":"Upper"})
                .style.format({"Predicted":"R${:,.0f}","Lower":"R${:,.0f}","Upper":"R${:,.0f}"}),
                use_container_width=True
            )
            st.success(f"🎯 **Total Predicted Revenue (Next 12 Months): R${future_forecast['yhat'].sum():,.0f}**")

            # Save for next time
            try:
                future_forecast.to_csv("forecast_next12months.csv", index=False)
                st.info("Forecast saved to forecast_next12months.csv for faster loading next time.")
            except Exception:
                pass

        except ImportError:
            st.error("Prophet is not installed. Run: `pip install prophet`")
        except Exception as e:
            st.error(f"Forecast error: {e}")


# ─────────────────────────────────────────────
# ══ PAGE 6 — ML METRICS ════════════════════
# ─────────────────────────────────────────────
elif page == "🤖 ML Metrics":
    st.title("🤖 ML Model Performance")
    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    ml_cols = [c for c in [
        "price","shipping_charges","item_revenue","delivery_days",
        "approval_lag_hrs","order_month","order_quarter","order_day_of_week",
        "payment_installments","product_weight_g","is_delivered","is_canceled"
    ] if c in df.columns]

    if len(ml_cols) >= 3:
        corr_df = df[ml_cols].dropna()
        corr_matrix = corr_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size":8},
            cbar_kws={"shrink":0.8}
        )
        ax.set_title("Correlation Heatmap — ML Features", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Not enough numeric columns for correlation analysis.")

    # Forecast evaluation
    eval_file = safe_read("forecast_evaluation.csv")
    if not eval_file.empty:
        st.markdown('<div class="section-header">Forecast Evaluation — Actual vs Predicted</div>', unsafe_allow_html=True)
        if "ds" in eval_file.columns:
            eval_file["ds"] = pd.to_datetime(eval_file["ds"])

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        x = range(len(eval_file))
        ax.plot(x, eval_file.get("actual", []), color=COLORS[0], marker="o", linewidth=2, label="Actual")
        ax.plot(x, eval_file.get("predicted", []), color=COLORS[2], marker="s",
                linewidth=2, linestyle="--", label="Predicted")
        if "ds" in eval_file.columns:
            ax.set_xticks(list(x))
            ax.set_xticklabels([d.strftime("%b %Y") for d in eval_file["ds"]], rotation=45, color="white")
        ax.set_title("Actual vs Predicted Revenue (Test Set)", color="white", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#334155")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close(fig)

        st.dataframe(
            eval_file.style.format({
                "actual":"R${:,.0f}", "predicted":"R${:,.0f}",
                "error":"R${:,.0f}", "error_pct":"{:.2f}%"
            }),
            use_container_width=True
        )

    # Metrics file
    metrics_df = safe_read("model_metrics.csv")
    if not metrics_df.empty:
        st.markdown('<div class="section-header">Model Metrics Summary</div>', unsafe_allow_html=True)
        m = metrics_df.iloc[0]
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val in [
            (c1, "MAE",           f"R${m.get('MAE',0):,.0f}"),
            (c2, "RMSE",          f"R${m.get('RMSE',0):,.0f}"),
            (c3, "R² Score",      f"{m.get('R2',0):.4f}"),
            (c4, "MAPE",          f"{m.get('MAPE_pct',0):.2f}%"),
            (c5, "Accuracy",      f"{m.get('Accuracy',0):.2f}%"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("""
    **Model:** Facebook Prophet  
    **Type:** Time Series Forecasting  
    **Training Data:** 2016 – 2025 monthly revenue  
    **Forecast Horizon:** 12 months  
    **Seasonality Mode:** Multiplicative  
    **Holiday Calendar:** Brazil (BR)
    """)
