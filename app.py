"""
Supply Chain Analytics & Revenue Forecasting
Capstone 2 — Streamlit Web App
Fixed: gdown fuzzy arg removed, robust data loading, all pages working
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import subprocess
import sys

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GOOGLE DRIVE DATA DOWNLOAD  (fuzzy arg removed)
# ─────────────────────────────────────────────
DRIVE_FILES = {
    "master_dataset.csv":      "1EpQR-38_CsBxStdUKkOrNBEVrw6IcZmO",
    "cleaned_orders.csv":      "17g4LmJ7oUyzGtJ61zqwMJd-jNEDw3mN9",
    "cleaned_order_items.csv": "1e2UsvsgTz48yQSwThnJycvJNnNp0tsiH",
    "cleaned_customers.csv":   "1aJLBpLsZiGMc38YCKcn7jCVQdcnJo8Cs",
    "cleaned_payments.csv":    "1UzJZc-meF2i2yOYLabp0yoiOfNP3nK5P",
    "cleaned_products.csv":    "1GwwuZr2GgyzBVKM25VljWwQ4ktMcnTid",
}

@st.cache_resource(show_spinner=False)
def ensure_data_files():
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    COMPONENT_FILES = [
        "cleaned_orders.csv", "cleaned_order_items.csv",
        "cleaned_customers.csv", "cleaned_payments.csv", "cleaned_products.csv",
    ]

    def _download(file_id, dest):
        """Download from GDrive — works with both old and new gdown versions."""
        import gdown as _gdown
        import inspect
        url = f"https://drive.google.com/uc?id={file_id}"
        sig = inspect.signature(_gdown.download)
        kwargs = {"quiet": True}
        if "fuzzy" in sig.parameters:
            kwargs["fuzzy"] = True
        _gdown.download(url, dest, **kwargs)

    # Download master_dataset.csv first
    if not os.path.exists("master_dataset.csv"):
        progress = st.progress(0, text="Downloading master_dataset.csv from Google Drive...")
        try:
            _download(DRIVE_FILES["master_dataset.csv"], "master_dataset.csv")
            progress.progress(1.0, text="master_dataset.csv ready!")
        except Exception as e:
            st.warning(f"Could not download master_dataset.csv: {e}")
        progress.empty()

    if os.path.exists("master_dataset.csv"):
        return

    # Fall back: download 5 component files
    missing = [n for n in COMPONENT_FILES if not os.path.exists(n)]
    if not missing:
        return

    progress = st.progress(0, text="Downloading component data files from Google Drive...")
    for i, name in enumerate(missing):
        progress.progress(i / len(missing), text=f"Downloading {name} ({i+1}/{len(missing)})...")
        try:
            _download(DRIVE_FILES[name], name)
        except Exception as e:
            st.warning(f"Could not download {name}: {e}")
    progress.progress(1.0, text="All files ready!")
    progress.empty()

ensure_data_files()

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
    .insight-box {
        background: #0f172a;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#7C3AED", "#0891B2"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def fmt_r(val):
    return f"R${val:,.0f}"

def safe_read(fname):
    if os.path.exists(fname):
        try:
            return pd.read_csv(fname, low_memory=False)
        except Exception:
            pass
    return pd.DataFrame()

def apply_dark_style(ax, title=None):
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#334155")
    if title:
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_all_data():
    def _read(fname):
        if os.path.exists(fname):
            try:
                return pd.read_csv(fname, low_memory=False)
            except Exception:
                pass
        return pd.DataFrame()

    def try_load(names):
        for n in names:
            df = _read(n)
            if not df.empty:
                return df
        return pd.DataFrame()

    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_timestamp", "order_estimated_delivery_date",
    ]

    def parse_dates(df):
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def norm_col(df, target, alts):
        if target not in df.columns:
            for a in alts:
                if a in df.columns:
                    return df.rename(columns={a: target})
        return df

    def add_time_cols(df):
        if "order_purchase_timestamp" in df.columns:
            ts = df["order_purchase_timestamp"]
            df["order_year"]        = ts.dt.year
            df["order_month"]       = ts.dt.month
            df["order_quarter"]     = ts.dt.quarter
            df["order_day_of_week"] = ts.dt.dayofweek
        return df

    def add_derived(df):
        # delivery days
        if "order_delivered_timestamp" in df.columns and "order_purchase_timestamp" in df.columns:
            df["delivery_days"] = (
                (df["order_delivered_timestamp"] - df["order_purchase_timestamp"])
                .dt.total_seconds() / 86400
            ).round(1)
            if "order_estimated_delivery_date" in df.columns:
                df["on_time_delivery"] = (
                    df["order_delivered_timestamp"] <= df["order_estimated_delivery_date"]
                ).astype("Int64")
        # approval lag
        if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
            df["approval_lag_hrs"] = (
                (df["order_approved_at"] - df["order_purchase_timestamp"])
                .dt.total_seconds() / 3600
            ).round(2)
        # status flags
        if "order_status" in df.columns:
            df["is_delivered"] = (df["order_status"] == "delivered").astype(int)
            df["is_canceled"]  = (df["order_status"] == "canceled").astype(int)
        # freight alias
        if "shipping_charges" not in df.columns and "freight_value" in df.columns:
            df["shipping_charges"] = df["freight_value"]
        # item_revenue
        if "price" in df.columns and "shipping_charges" in df.columns and "item_revenue" not in df.columns:
            df["item_revenue"] = df["price"] + df["shipping_charges"]
        return df

    # ── Fast path: master_dataset.csv ───────────────
    master = try_load(["master_dataset.csv"])
    monthly = try_load(["monthly_revenue.csv"])
    if "ds" in monthly.columns:
        monthly["ds"] = pd.to_datetime(monthly["ds"], errors="coerce")

    if not master.empty:
        master = parse_dates(master)
        master = add_time_cols(master)
        master = add_derived(master)
        orders = master[["order_id","order_status","order_purchase_timestamp","customer_id"]].drop_duplicates("order_id") \
            if all(c in master.columns for c in ["order_id","order_status","order_purchase_timestamp","customer_id"]) \
            else pd.DataFrame()
        # rebuild monthly if missing
        if monthly.empty:
            monthly = _build_monthly(master)
        return orders, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), monthly, master

    # ── Slow path: component files ───────────────────
    orders    = parse_dates(try_load(["cleaned_orders.csv",     "final_orders.csv"]))
    items     = try_load(["cleaned_order_items.csv", "final_order_items.csv"])
    customers = try_load(["cleaned_customers.csv",   "final_customers.csv"])
    payments  = try_load(["cleaned_payments.csv",    "final_payments.csv"])
    products  = try_load(["cleaned_products.csv",    "final_products.csv"])

    if orders.empty:
        return orders, items, customers, payments, products, monthly, pd.DataFrame()

    # normalise column names
    orders    = norm_col(orders,    "customer_id", ["customerid","customer"])
    items     = norm_col(items,     "order_id",    ["orderid","order"])
    customers = norm_col(customers, "customer_id", ["customerid","customer"])
    payments  = norm_col(payments,  "order_id",    ["orderid","order"])
    products  = norm_col(products,  "product_id",  ["productid","product"])

    if "shipping_charges" not in items.columns and "freight_value" in items.columns:
        items = items.rename(columns={"freight_value": "shipping_charges"})

    # aggregate payments
    pay_agg = pd.DataFrame()
    if not payments.empty and "order_id" in payments.columns:
        agg_dict = {}
        if "payment_value"        in payments.columns: agg_dict["payment_value"]        = ("payment_value",        "sum")
        if "payment_type"         in payments.columns: agg_dict["payment_type"]         = ("payment_type",         lambda x: x.mode()[0] if len(x) else "unknown")
        if "payment_installments" in payments.columns: agg_dict["payment_installments"] = ("payment_installments", "max")
        if agg_dict:
            pay_agg = payments.groupby("order_id").agg(**agg_dict).reset_index()

    # merge
    m = orders.copy()
    if not customers.empty and "customer_id" in m.columns and "customer_id" in customers.columns:
        cust_cols = ["customer_id"] + [c for c in customers.columns if c not in m.columns]
        m = m.merge(customers[cust_cols], on="customer_id", how="left")
    if not items.empty and "order_id" in m.columns and "order_id" in items.columns:
        item_cols = ["order_id"] + [c for c in items.columns if c not in m.columns]
        m = m.merge(items[item_cols], on="order_id", how="left")
    if not products.empty and "product_id" in m.columns and "product_id" in products.columns:
        prod_cols = ["product_id"] + [c for c in products.columns if c not in m.columns]
        m = m.merge(products[prod_cols], on="product_id", how="left")
    if not pay_agg.empty and "order_id" in m.columns:
        pay_cols = ["order_id"] + [c for c in pay_agg.columns if c not in m.columns]
        m = m.merge(pay_agg[pay_cols], on="order_id", how="left")

    m = parse_dates(m)
    m = add_time_cols(m)
    m = add_derived(m)

    if monthly.empty:
        monthly = _build_monthly(m)

    return orders, items, customers, payments, products, monthly, m


def _build_monthly(master):
    """Build monthly revenue dataframe from master."""
    if master.empty:
        return pd.DataFrame()
    tmp = master.copy()
    if "order_status" in tmp.columns:
        tmp = tmp[tmp["order_status"] == "delivered"]
    if not all(c in tmp.columns for c in ["order_year","order_month","item_revenue"]):
        return pd.DataFrame()
    agg_kw = dict(y=("item_revenue","sum"))
    if "order_id"     in tmp.columns: agg_kw["total_orders"]    = ("order_id",    "nunique")
    if "item_revenue" in tmp.columns: agg_kw["avg_order_value"] = ("item_revenue","mean")
    if "customer_id"  in tmp.columns: agg_kw["total_customers"] = ("customer_id", "nunique")
    monthly = tmp.groupby(["order_year","order_month"]).agg(**agg_kw).reset_index()
    monthly["ds"] = pd.to_datetime(
        monthly["order_year"].astype(str) + "-" +
        monthly["order_month"].astype(str).str.zfill(2) + "-01"
    )
    return monthly.sort_values("ds").reset_index(drop=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
orders, items, customers, payments, products, monthly_rev, master = load_all_data()

if master.empty and orders.empty and monthly_rev.empty:
    st.error(
        "❌ No data files found. Please ensure at least `monthly_revenue.csv` "
        "or the cleaned CSV files are in the same directory as `app.py`."
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

    if "order_year" in df.columns:
        years = sorted(df["order_year"].dropna().unique().astype(int))
        sel_years = st.multiselect("Filter by Year", years, default=years)
        if sel_years:
            df = df[df["order_year"].isin(sel_years)]
            if not monthly_rev.empty and "ds" in monthly_rev.columns:
                monthly_rev = monthly_rev[monthly_rev["ds"].dt.year.isin(sel_years)]

    st.markdown("---")
    st.caption("Capstone 2 — Supply Chain Analytics & Revenue Forecasting")


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📦 Supply Chain Analytics Dashboard")
    st.markdown("End-to-end analytics across orders, customers, products, and revenue.")
    st.markdown("---")

    delivered = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpi_data = [
        (col1, "Total Orders",    f"{df['order_id'].nunique():,}"              if "order_id"      in df.columns else "N/A", "📦"),
        (col2, "Total Revenue",   f"R${df['item_revenue'].sum():,.0f}"         if "item_revenue"  in df.columns else "N/A", "💰"),
        (col3, "Avg Order Value", f"R${df['item_revenue'].mean():,.2f}"        if "item_revenue"  in df.columns else "N/A", "🛒"),
        (col4, "Total Customers", f"{df['customer_id'].nunique():,}"           if "customer_id"   in df.columns else "N/A", "👥"),
        (col5, "Delivery Rate",
         f"{delivered.shape[0]/max(df.shape[0],1)*100:.1f}%"                   if not delivered.empty else "N/A", "✅"),
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

    with col_a:
        st.markdown('<div class="section-header">Order Status Distribution</div>', unsafe_allow_html=True)
        if "order_status" in df.columns:
            status_counts = df["order_status"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
            apply_dark_style(ax, "Order Status")
            wedges, texts, autotexts = ax.pie(
                status_counts.values, labels=status_counts.index,
                autopct="%1.1f%%", colors=COLORS[:len(status_counts)], startangle=90,
            )
            for t in texts + autotexts:
                t.set_color("white"); t.set_fontsize(9)
            st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown('<div class="section-header">Payment Type Distribution</div>', unsafe_allow_html=True)
        pay_col = "payment_type" if "payment_type" in df.columns else None
        if pay_col:
            pay_counts = df[pay_col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
            apply_dark_style(ax, "Payment Types")
            bars = ax.barh(pay_counts.index, pay_counts.values, color=COLORS[:len(pay_counts)])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for bar in bars:
                ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                        f"{int(bar.get_width()):,}", va="center", color="white", fontsize=9)
            st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insights = []
    if "order_status" in df.columns:
        top_status = df["order_status"].value_counts().index[0]
        top_pct    = df["order_status"].value_counts(normalize=True).iloc[0] * 100
        insights.append(f"🟢 <b>{top_status.title()}</b> is the most common order status at <b>{top_pct:.1f}%</b>")
    if "product_category_name" in df.columns:
        top_cat = df["product_category_name"].dropna().value_counts().index[0]
        insights.append(f"🛍️ Top product category: <b>{top_cat.replace('_',' ').title()}</b>")
    if "customer_state" in df.columns:
        top_state = df["customer_state"].dropna().value_counts().index[0]
        insights.append(f"📍 Most orders from state: <b>{top_state}</b>")
    if "delivery_days" in delivered.columns:
        avg_del = delivered["delivery_days"].mean()
        insights.append(f"🚚 Average delivery time: <b>{avg_del:.1f} days</b>")
    if not monthly_rev.empty and "y" in monthly_rev.columns:
        best = monthly_rev.loc[monthly_rev["y"].idxmax()]
        insights.append(f"📈 Best revenue month: <b>{pd.to_datetime(best['ds']).strftime('%b %Y')} — R${best['y']:,.0f}</b>")

    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2 — REVENUE TRENDS
# ═══════════════════════════════════════════════
elif page == "📈 Revenue Trends":
    st.title("📈 Revenue Trends")
    st.markdown("---")

    if monthly_rev.empty or "ds" not in monthly_rev.columns:
        st.warning("Monthly revenue data not available.")
        st.stop()

    st.markdown('<div class="section-header">Monthly Revenue Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
    apply_dark_style(ax, "Monthly Revenue Trend")
    ax.plot(monthly_rev["ds"], monthly_rev["y"], color=COLORS[0], linewidth=2, marker="o", markersize=3)
    ax.fill_between(monthly_rev["ds"], monthly_rev["y"], alpha=0.15, color=COLORS[0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
    ax.grid(True, alpha=0.2)
    st.pyplot(fig); plt.close(fig)

    # Yearly summary
    st.markdown('<div class="section-header">Yearly Revenue Summary</div>', unsafe_allow_html=True)
    if "order_year" in df.columns and "item_revenue" in df.columns:
        delivered_df = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df
        yearly = (
            delivered_df.groupby("order_year")
            .agg(total_revenue=("item_revenue","sum"), total_orders=("order_id","nunique"))
            .reset_index()
        )
        yearly["revenue_yoy_pct"] = yearly["total_revenue"].pct_change() * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            apply_dark_style(ax, "Revenue by Year")
            ax.bar(yearly["order_year"].astype(str), yearly["total_revenue"], color=COLORS[0])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.0f}M"))
            st.pyplot(fig); plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            apply_dark_style(ax, "YoY Growth %")
            yoy = yearly.dropna(subset=["revenue_yoy_pct"])
            bar_colors = [COLORS[1] if v >= 0 else COLORS[2] for v in yoy["revenue_yoy_pct"]]
            ax.bar(yoy["order_year"].astype(str), yoy["revenue_yoy_pct"], color=bar_colors)
            ax.axhline(0, color="white", linewidth=0.8)
            st.pyplot(fig); plt.close(fig)

        st.dataframe(
            yearly.style.format({
                "total_revenue": "R${:,.0f}",
                "total_orders": "{:,}",
                "revenue_yoy_pct": "{:.1f}%"
            }),
            use_container_width=True
        )

    # Monthly seasonality
    st.markdown('<div class="section-header">Monthly Seasonality</div>', unsafe_allow_html=True)
    if "order_month" in df.columns and "item_revenue" in df.columns:
        delivered_df = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df
        monthly_avg = delivered_df.groupby("order_month")["item_revenue"].mean().reset_index()
        monthly_avg["month_name"] = monthly_avg["order_month"].apply(lambda x: MONTH_NAMES[x-1])

        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f172a")
        apply_dark_style(ax, "Average Revenue by Month (Seasonality)")
        ax.bar(monthly_avg["month_name"], monthly_avg["item_revenue"], color=COLORS[0])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig); plt.close(fig)
    elif not monthly_rev.empty:
        # fallback using monthly_rev
        mr = monthly_rev.copy()
        mr["month"] = pd.to_datetime(mr["ds"]).dt.month
        monthly_avg = mr.groupby("month")["y"].mean().reset_index()
        monthly_avg["month_name"] = monthly_avg["month"].apply(lambda x: MONTH_NAMES[x-1])
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f172a")
        apply_dark_style(ax, "Average Revenue by Month (Seasonality)")
        ax.bar(monthly_avg["month_name"], monthly_avg["y"], color=COLORS[0])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# PAGE 3 — CATEGORY ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🗂️ Category Analysis":
    st.title("🗂️ Category Analysis")
    st.markdown("---")

    has_cat = "product_category_name" in df.columns and not df["product_category_name"].dropna().empty
    has_rev = "item_revenue" in df.columns

    if not has_cat or not has_rev:
        # Graceful fallback using monthly_rev
        st.info(
            "ℹ️ Detailed category data requires `master_dataset.csv` or the 5 cleaned component CSV files. "
            "Showing revenue summary from `monthly_revenue.csv` instead."
        )
        if not monthly_rev.empty:
            st.markdown('<div class="section-header">Revenue by Year (from monthly data)</div>', unsafe_allow_html=True)
            mr = monthly_rev.copy()
            mr["year"] = pd.to_datetime(mr["ds"]).dt.year
            yearly_sum = mr.groupby("year")["y"].sum().reset_index().rename(columns={"y":"total_revenue"})

            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
            apply_dark_style(ax, "Annual Revenue Summary")
            ax.bar(yearly_sum["year"].astype(str), yearly_sum["total_revenue"], color=COLORS[0])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig); plt.close(fig)

            st.dataframe(
                yearly_sum.style.format({"total_revenue": "R${:,.0f}"}),
                use_container_width=True
            )
        st.stop()

    delivered_df = df[df["order_status"] == "delivered"] if "order_status" in df.columns else df
    cat_rev = (
        delivered_df.groupby("product_category_name")
        .agg(
            total_revenue=("item_revenue","sum"),
            total_orders=("order_id","nunique"),
            avg_price=("price","mean") if "price" in delivered_df.columns else ("item_revenue","mean"),
        )
        .reset_index().sort_values("total_revenue", ascending=False).head(10)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top 10 by Revenue</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
        apply_dark_style(ax, "Revenue by Category")
        ax.barh(cat_rev["product_category_name"][::-1], cat_rev["total_revenue"][::-1], color=COLORS[0])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.0f}M"))
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Top 10 by Orders</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
        apply_dark_style(ax, "Orders by Category")
        ax.barh(cat_rev["product_category_name"][::-1], cat_rev["total_orders"][::-1], color=COLORS[1])
        st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">Category Revenue Table</div>', unsafe_allow_html=True)
    cat_disp = cat_rev.rename(columns={
        "product_category_name": "Category",
        "total_revenue": "Revenue (R$)",
        "total_orders": "Orders",
        "avg_price": "Avg Price (R$)"
    })
    st.dataframe(
        cat_disp.style.format({
            "Revenue (R$)": "R${:,.0f}",
            "Orders": "{:,}",
            "Avg Price (R$)": "R${:,.2f}"
        }),
        use_container_width=True
    )


# ═══════════════════════════════════════════════
# PAGE 4 — DELIVERY ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🚚 Delivery Analysis":
    st.title("🚚 Delivery Analysis")
    st.markdown("---")

    if "delivery_days" not in df.columns:
        st.info(
            "ℹ️ Delivery analysis requires `master_dataset.csv` or the cleaned component files. "
            "The dataset loaded does not contain delivery timestamp columns."
        )
        # Show whatever order status summary we can
        if "order_status" in df.columns:
            st.markdown('<div class="section-header">Order Status Summary</div>', unsafe_allow_html=True)
            status_counts = df["order_status"].value_counts().reset_index()
            status_counts.columns = ["Status","Count"]
            status_counts["Pct"] = (status_counts["Count"] / status_counts["Count"].sum() * 100).round(2)
            st.dataframe(status_counts.style.format({"Count":"{:,}","Pct":"{:.2f}%"}), use_container_width=True)
        st.stop()

    delivered = df[df["order_status"] == "delivered"].copy() if "order_status" in df.columns else df.copy()
    delivered = delivered.dropna(subset=["delivery_days"])
    delivered = delivered[delivered["delivery_days"] > 0]  # remove negatives / bad data

    col1, col2, col3, col4 = st.columns(4)
    on_time_rate = f"{delivered['on_time_delivery'].mean()*100:.1f}%" if "on_time_delivery" in delivered.columns else "N/A"
    for col, label, value in [
        (col1, "Avg Delivery Days",  f"{delivered['delivery_days'].mean():.1f}d"),
        (col2, "Median Delivery",    f"{delivered['delivery_days'].median():.0f}d"),
        (col3, "Max Delivery Days",  f"{delivered['delivery_days'].max():.0f}d"),
        (col4, "On-Time Rate",       on_time_rate),
    ]:
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
        apply_dark_style(ax, "Delivery Days Distribution")
        delivered["delivery_days"].clip(upper=60).hist(bins=30, ax=ax, color=COLORS[0], edgecolor="#0f172a")
        ax.axvline(delivered["delivery_days"].mean(), color=COLORS[2], linestyle="--",
                   label=f"Mean: {delivered['delivery_days'].mean():.1f}d")
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
            apply_dark_style(ax, "Avg Delivery Days by State")
            ax.barh(state_del["customer_state"][::-1], state_del["delivery_days"][::-1], color=COLORS[3])
            st.pyplot(fig); plt.close(fig)

    if "product_category_name" in delivered.columns and "on_time_delivery" in delivered.columns:
        st.markdown('<div class="section-header">On-Time Delivery by Category</div>', unsafe_allow_html=True)
        cat_otd = (
            delivered.groupby("product_category_name")["on_time_delivery"]
            .mean().sort_values(ascending=False).head(10).reset_index()
        )
        cat_otd["on_time_pct"] = cat_otd["on_time_delivery"] * 100
        fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
        apply_dark_style(ax, "On-Time Delivery % by Category")
        bar_colors = [COLORS[1] if v >= 80 else COLORS[3] if v >= 60 else COLORS[2]
                      for v in cat_otd["on_time_pct"]]
        ax.bar(cat_otd["product_category_name"], cat_otd["on_time_pct"], color=bar_colors)
        ax.set_ylabel("On-Time %", color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.axhline(80, color=COLORS[2], linestyle="--", linewidth=1, label="80% target")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        st.pyplot(fig); plt.close(fig)

    # Delivery days by month
    if "order_month" in delivered.columns:
        st.markdown('<div class="section-header">Avg Delivery Days by Month</div>', unsafe_allow_html=True)
        monthly_del = delivered.groupby("order_month")["delivery_days"].mean().reset_index()
        monthly_del["month_name"] = monthly_del["order_month"].apply(lambda x: MONTH_NAMES[x-1])
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f172a")
        apply_dark_style(ax, "Avg Delivery Days by Month")
        ax.plot(monthly_del["month_name"], monthly_del["delivery_days"],
                color=COLORS[4], marker="o", linewidth=2)
        ax.set_ylabel("Days", color="white")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# PAGE 5 — REVENUE FORECAST
# ═══════════════════════════════════════════════
elif page == "🔮 Revenue Forecast":
    st.title("🔮 Revenue Forecast — Next 12 Months")
    st.markdown("Facebook Prophet time-series forecasting model.")
    st.markdown("---")

    pre_forecast = safe_read("forecast_next12months.csv")
    pre_eval     = safe_read("forecast_evaluation.csv")
    pre_metrics  = safe_read("model_metrics.csv")

    if not pre_forecast.empty:
        if "ds" in pre_forecast.columns:
            pre_forecast["ds"] = pd.to_datetime(pre_forecast["ds"])

        st.markdown('<div class="section-header">Next 12 Months Forecast</div>', unsafe_allow_html=True)

        if not pre_metrics.empty:
            m = pre_metrics.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in [
                (c1, "MAE",      f"R${float(m.get('MAE',0)):,.0f}"),
                (c2, "RMSE",     f"R${float(m.get('RMSE',0)):,.0f}"),
                (c3, "R² Score", f"{float(m.get('R2',0)):.4f}"),
                (c4, "Accuracy", f"{float(m.get('Accuracy',0)):.2f}%"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True
                    )
            st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
        apply_dark_style(ax, "Revenue Forecast — Next 12 Months")
        if not monthly_rev.empty and "ds" in monthly_rev.columns:
            ax.plot(monthly_rev["ds"], monthly_rev["y"],
                    color=COLORS[0], linewidth=2, label="Historical Revenue")
            ax.axvline(monthly_rev["ds"].max(), color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")

        ax.plot(pre_forecast["ds"], pre_forecast["yhat"],
                color=COLORS[2], linewidth=2.5, linestyle="--", label="Forecast", zorder=3)
        if "yhat_lower" in pre_forecast.columns and "yhat_upper" in pre_forecast.columns:
            ax.fill_between(pre_forecast["ds"],
                            pre_forecast["yhat_lower"], pre_forecast["yhat_upper"],
                            alpha=0.2, color=COLORS[2], label="95% CI")

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close(fig)

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
        st.info("Pre-computed forecast not found. Running Prophet model now — this may take ~30 seconds...")

        if monthly_rev.empty or "ds" not in monthly_rev.columns or "y" not in monthly_rev.columns:
            st.error("monthly_revenue.csv is required to run the forecast. Please add it to the app directory.")
            st.stop()

        try:
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            prophet_df = monthly_rev[["ds","y"]].copy().dropna()
            prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

            FORECAST_MONTHS = 12
            TEST_MONTHS     = min(12, max(3, len(prophet_df) // 4))

            train_df = prophet_df.iloc[:-TEST_MONTHS].copy()
            test_df  = prophet_df.iloc[-TEST_MONTHS:].copy()

            with st.spinner("Training Prophet model..."):
                model = Prophet(
                    yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="multiplicative", changepoint_prior_scale=0.1,
                    seasonality_prior_scale=10, interval_width=0.95,
                )
                try:
                    model.add_country_holidays(country_name="BR")
                except Exception:
                    pass
                model.fit(train_df)

            test_future   = model.make_future_dataframe(periods=TEST_MONTHS, freq="MS")
            test_forecast = model.predict(test_future)
            test_pred     = test_forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(TEST_MONTHS).reset_index(drop=True)

            y_actual = test_df.reset_index(drop=True)["y"].values
            y_pred   = test_pred["yhat"].values
            mae      = mean_absolute_error(y_actual, y_pred)
            rmse     = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2       = r2_score(y_actual, y_pred)
            mape     = np.mean(np.abs((y_actual - y_pred) / np.maximum(y_actual, 1))) * 100

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
            for col in ["yhat","yhat_lower","yhat_upper"]:
                future_forecast[col] = future_forecast[col].clip(lower=0)

            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in [
                (c1, "MAE",      f"R${mae:,.0f}"),
                (c2, "RMSE",     f"R${rmse:,.0f}"),
                (c3, "R² Score", f"{r2:.4f}"),
                (c4, "Accuracy", f"{100-mape:.2f}%"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True
                    )
            st.markdown("<br>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
            apply_dark_style(ax, f"Revenue Forecast | Accuracy: {100-mape:.1f}% | R²: {r2:.4f}")
            ax.plot(prophet_df["ds"], prophet_df["y"], color=COLORS[0], linewidth=2, label="Historical")
            ax.plot(future_forecast["ds"], future_forecast["yhat"],
                    color=COLORS[2], linewidth=2.5, linestyle="--", label="Forecast")
            ax.fill_between(future_forecast["ds"],
                            future_forecast["yhat_lower"], future_forecast["yhat_upper"],
                            alpha=0.2, color=COLORS[2], label="95% CI")
            ax.axvline(prophet_df["ds"].max(), color="gray", linestyle=":", linewidth=1.5)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
            ax.legend(facecolor="#1e293b", labelcolor="white")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig); plt.close(fig)

            disp = future_forecast.copy()
            disp["Month"] = disp["ds"].dt.strftime("%b %Y")
            st.dataframe(
                disp[["Month","yhat","yhat_lower","yhat_upper"]]
                .rename(columns={"yhat":"Predicted","yhat_lower":"Lower","yhat_upper":"Upper"})
                .style.format({"Predicted":"R${:,.0f}","Lower":"R${:,.0f}","Upper":"R${:,.0f}"}),
                use_container_width=True
            )
            st.success(f"🎯 **Total Predicted Revenue (Next 12 Months): R${future_forecast['yhat'].sum():,.0f}**")

            try:
                future_forecast.to_csv("forecast_next12months.csv", index=False)
                # save eval + metrics too
                eval_df = pd.DataFrame({
                    "ds": test_df["ds"].values,
                    "actual": y_actual,
                    "predicted": y_pred,
                    "error": y_actual - y_pred,
                    "error_pct": np.abs((y_actual - y_pred) / np.maximum(y_actual,1)) * 100,
                })
                eval_df.to_csv("forecast_evaluation.csv", index=False)
                metrics_out = pd.DataFrame([{
                    "MAE": round(mae,2), "RMSE": round(rmse,2),
                    "R2": round(r2,4), "MAPE_pct": round(mape,2),
                    "Accuracy": round(100-mape,2)
                }])
                metrics_out.to_csv("model_metrics.csv", index=False)
                st.info("Forecast saved — will load instantly on next run.")
            except Exception:
                pass

        except ImportError:
            st.error("Prophet is not installed. Run: `pip install prophet`")
        except Exception as e:
            st.error(f"Forecast error: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════
# PAGE 6 — ML METRICS
# ═══════════════════════════════════════════════
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
        if len(corr_df) > 0:
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
        st.info("Feature correlation requires master dataset. Not enough numeric columns found.")

    # Forecast evaluation
    eval_file_df = safe_read("forecast_evaluation.csv")
    if not eval_file_df.empty:
        st.markdown('<div class="section-header">Forecast Evaluation — Actual vs Predicted</div>', unsafe_allow_html=True)
        if "ds" in eval_file_df.columns:
            eval_file_df["ds"] = pd.to_datetime(eval_file_df["ds"])

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
        apply_dark_style(ax, "Actual vs Predicted Revenue (Test Set)")
        x = range(len(eval_file_df))
        ax.plot(x, eval_file_df["actual"],    color=COLORS[0], marker="o", linewidth=2, label="Actual")
        ax.plot(x, eval_file_df["predicted"], color=COLORS[2], marker="s",
                linewidth=2, linestyle="--", label="Predicted")
        if "ds" in eval_file_df.columns:
            ax.set_xticks(list(x))
            ax.set_xticklabels([d.strftime("%b %Y") for d in eval_file_df["ds"]], rotation=45, color="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close(fig)

        fmt_map = {}
        if "actual"    in eval_file_df.columns: fmt_map["actual"]    = "R${:,.0f}"
        if "predicted" in eval_file_df.columns: fmt_map["predicted"] = "R${:,.0f}"
        if "error"     in eval_file_df.columns: fmt_map["error"]     = "R${:,.0f}"
        if "error_pct" in eval_file_df.columns: fmt_map["error_pct"] = "{:.2f}%"
        st.dataframe(eval_file_df.style.format(fmt_map), use_container_width=True)

    # Model metrics
    metrics_df = safe_read("model_metrics.csv")
    if not metrics_df.empty:
        st.markdown('<div class="section-header">Model Metrics Summary</div>', unsafe_allow_html=True)
        m = metrics_df.iloc[0]
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val in [
            (c1, "MAE",      f"R${float(m.get('MAE',0)):,.0f}"),
            (c2, "RMSE",     f"R${float(m.get('RMSE',0)):,.0f}"),
            (c3, "R² Score", f"{float(m.get('R2',0)):.4f}"),
            (c4, "MAPE",     f"{float(m.get('MAPE_pct',0)):.2f}%"),
            (c5, "Accuracy", f"{float(m.get('Accuracy',0)):.2f}%"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Run the Revenue Forecast page once to generate and save model metrics.")

    st.markdown("---")
    st.markdown("""
**Model:** Facebook Prophet  
**Type:** Time Series Forecasting  
**Training Data:** 2016 – 2026 monthly revenue  
**Forecast Horizon:** 12 months  
**Seasonality Mode:** Multiplicative  
**Holiday Calendar:** Brazil (BR)
    """)
