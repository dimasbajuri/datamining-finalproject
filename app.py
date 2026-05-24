from __future__ import annotations

import os
import math
import textwrap
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except Exception:
    StandardScaler = None
    KMeans = None


# =========================================================
# CONFIG
# =========================================================
APP_NAME = "Alradi Mart"
DATA_FILE = "final_dataset_fixed.csv"
PAGE_ICON = "🛒"

st.set_page_config(
    page_title=f"{APP_NAME} | Marketplace & Data Mining",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLING
# =========================================================
def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800;900&display=swap');

:root{
  --bg:#070d18;
  --panel:#0c1728;
  --panel2:#101e33;
  --line:#1e3a5f;
  --muted:#9aa8bb;
  --text:#eef5ff;
  --blue:#2f8cff;
  --blue2:#58b5ff;
  --gold:#ffcf4a;
  --danger:#ff4d61;
}
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at top left, rgba(47,140,255,.16), transparent 24%),
              radial-gradient(circle at top right, rgba(13,45,90,.22), transparent 26%),
              var(--bg);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stHeader"]{background:rgba(7,13,24,.88);}
[data-testid="stToolbar"]{display:none;}
.block-container{padding-top:1.1rem; max-width:1360px;}
hr{border-color:rgba(88,181,255,.15);}

/* text */
h1,h2,h3{letter-spacing:-.035em;}
.small-muted{color:var(--muted);font-size:.84rem;}
.section-title{font-size:1.35rem;font-weight:900;margin:1.2rem 0 .35rem;}
.section-sub{color:var(--muted);margin-top:-.25rem;margin-bottom:.75rem;font-size:.9rem;}

/* header */
.top-strip{
  display:flex; justify-content:space-between; gap:20px; color:#bdc8d8; font-size:.82rem;
  margin:2px 0 12px;
}
.header-shell{
  border:1px solid rgba(88,181,255,.25);
  background:linear-gradient(135deg, rgba(13,25,44,.96), rgba(8,14,25,.96));
  border-radius:24px;
  padding:20px 22px;
  box-shadow:0 16px 50px rgba(0,0,0,.25);
  margin-bottom:12px;
}
.logo-wrap{display:flex; align-items:center; gap:14px;}
.logo-mark{
  width:54px;height:54px;border-radius:18px;
  background:linear-gradient(135deg,var(--blue),#1754b7);
  display:flex;align-items:center;justify-content:center;
  font-weight:900;color:white;letter-spacing:.06em;
  box-shadow:0 12px 26px rgba(47,140,255,.3);
}
.logo-text{font-size:1.9rem;font-weight:900;line-height:1;}
.logo-text span{color:var(--blue2);}
.logo-caption{font-size:.82rem;color:var(--muted);margin-top:5px;}
.search-help{color:var(--muted);font-size:.78rem;margin-top:7px;}

/* Streamlit inputs/buttons */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input{
  background:#0a1322 !important;
  color:#eef5ff !important;
  border:1px solid rgba(88,181,255,.30) !important;
  border-radius:14px !important;
}
.stTextInput input:focus{
  border-color:rgba(88,181,255,.85) !important;
  box-shadow:0 0 0 3px rgba(47,140,255,.15) !important;
}
.stButton > button, .stDownloadButton > button{
  border:1px solid rgba(88,181,255,.45);
  background:linear-gradient(135deg,#2f8cff,#145ec5);
  color:white;
  border-radius:13px;
  font-weight:800;
  min-height:42px;
  transition:.18s ease;
  box-shadow:0 10px 28px rgba(47,140,255,.18);
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-2px);
  border-color:rgba(136,203,255,.85);
  box-shadow:0 16px 36px rgba(47,140,255,.28);
}
button[kind="secondary"]{
  background:#0b1526 !important;
}

/* cards */
.panel{
  border:1px solid rgba(88,181,255,.22);
  background:rgba(13,25,44,.78);
  border-radius:22px;
  padding:18px;
  box-shadow:0 14px 38px rgba(0,0,0,.18);
}
.hero{
  min-height:250px;
  border:1px solid rgba(88,181,255,.30);
  border-radius:24px;
  padding:34px;
  background:
    radial-gradient(circle at 78% 38%, rgba(88,181,255,.34), transparent 0 18%, transparent 19%),
    linear-gradient(135deg,#102847,#07182d 58%,#0c1424);
  position:relative;
  overflow:hidden;
}
.hero:before{
  content:"";
  position:absolute;right:52px;top:42px;width:280px;height:122px;border-radius:34px;
  background:linear-gradient(135deg,#9ad2ff,#105ac5);
  opacity:.58;filter:drop-shadow(0 16px 30px rgba(36,116,224,.35));
  animation:floaty 4s ease-in-out infinite;
}
.hero:after{
  content:"";
  position:absolute;right:115px;top:88px;width:105px;height:80px;border-radius:24px;
  background:linear-gradient(135deg,#4f91e5,#0e3d8a);
  opacity:.86;animation:floaty2 5s ease-in-out infinite;
}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes floaty2{0%,100%{transform:translateX(0)}50%{transform:translateX(12px)}}
.kicker{font-size:.78rem;letter-spacing:.20em;color:#8fc9ff;font-weight:900;text-transform:uppercase;}
.hero h1{font-size:2.65rem;font-weight:900;margin:.45rem 0 .45rem;max-width:640px;}
.hero p{font-size:1.02rem;color:#c8d3e3;max-width:610px;}
.side-banner{
  min-height:118px;border-radius:22px;padding:20px;border:1px solid rgba(88,181,255,.25);
  background:linear-gradient(135deg,#111b31,#0b1f3a);
  margin-bottom:14px;
}
.side-banner b{font-size:1.2rem;}
.wallet-row{
  display:flex;gap:10px;align-items:center;margin:8px 0;padding:12px;
  border:1px solid rgba(88,181,255,.18);border-radius:17px;background:#101b2e;
}
.wallet-icon{
  width:34px;height:34px;border-radius:12px;background:linear-gradient(135deg,#2f8cff,#155fc8);
  display:flex;align-items:center;justify-content:center;font-weight:900;
}
.wallet-val{font-weight:900;color:#9bd0ff;}
.keyword-wrap{display:flex;flex-wrap:wrap;gap:9px;}
.chip{
  display:inline-flex;align-items:center;justify-content:center;
  padding:8px 11px;border-radius:10px;background:#142033;border:1px solid rgba(255,255,255,.06);
  color:#dfe8f5;font-size:.82rem;font-weight:800;
}

/* service bar */
.service-row{
  display:grid;grid-template-columns:repeat(8,1fr);gap:12px;
  border:1px solid rgba(88,181,255,.22);border-radius:22px;padding:14px;background:#0c1728;margin-top:14px;
}
.service-card{
  min-height:96px;border:1px solid rgba(88,181,255,.13);border-radius:18px;background:#101b2e;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;
  transition:.18s ease;
}
.service-card:hover{transform:translateY(-3px);border-color:rgba(88,181,255,.55);background:#132846;}
.icon-circle{
  width:40px;height:40px;border-radius:14px;background:linear-gradient(135deg,#9cd5ff,#1f73d7);
  color:#06111f;font-weight:900;display:flex;align-items:center;justify-content:center;
}
.service-name{text-align:center;font-weight:850;font-size:.80rem;color:#e9f3ff;}

/* category grid */
.category-box{
  border:1px solid rgba(88,181,255,.22);
  background:#0c1728;
  border-radius:22px;
  overflow:hidden;
}
.category-head{padding:17px 18px;font-size:1.2rem;font-weight:900;border-bottom:1px solid rgba(88,181,255,.13);}
.category-grid{
  display:grid;grid-template-columns:repeat(8,1fr);
}
.cat-tile{
  min-height:128px;padding:16px 10px;border-right:1px solid rgba(88,181,255,.10);border-bottom:1px solid rgba(88,181,255,.10);
  text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
  transition:.16s ease;background:#0d192b;
}
.cat-tile:hover{background:#122844;transform:translateY(-2px);}
.cat-icon{
  width:58px;height:58px;border-radius:50%;
  background:linear-gradient(135deg,#c7e8ff,#76bdff);
  display:flex;align-items:center;justify-content:center;color:#07111f;font-weight:900;font-size:1.25rem;
}
.cat-name{font-size:.83rem;font-weight:900;line-height:1.25;color:#dfe8f5;min-height:34px;display:flex;align-items:center;}
.cat-count{font-size:.72rem;color:#97a7bc;}

/* products */
.product-card{
  height:360px;border:1px solid rgba(88,181,255,.18);border-radius:18px;overflow:hidden;
  background:#0c1728;margin-bottom:10px;box-shadow:0 14px 32px rgba(0,0,0,.18);
  transition:.16s ease;
}
.product-card:hover{transform:translateY(-4px);border-color:rgba(88,181,255,.5);}
.product-art{
  height:126px;background:linear-gradient(135deg,#c7e8ff,#2a5c92);
  display:flex;align-items:center;justify-content:center;color:#06111f;font-weight:900;font-size:2rem;
  position:relative;
}
.discount{
  position:absolute;left:12px;top:12px;background:#ff4d61;color:#fff;border-radius:12px;padding:6px 9px;font-size:.75rem;font-weight:900;
}
.product-body{padding:14px 14px 12px;}
.pname{font-weight:900;line-height:1.25;min-height:44px;font-size:.95rem;color:#f1f7ff;}
.price{font-size:1.08rem;color:#65b7ff;font-weight:900;margin:9px 0 2px;}
.old-price{color:#7f8c9f;text-decoration:line-through;font-size:.82rem;}
.saving{color:#ffcf4a;font-weight:850;font-size:.78rem;margin:9px 0 8px;}
.pmeta{color:#d1dbea;font-size:.82rem;margin-bottom:8px;}
.store{color:#98a7ba;font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.product-button-gap{margin-top:-4px;margin-bottom:16px;}

/* admin */
.metric-card{
  padding:18px;border-radius:20px;border:1px solid rgba(88,181,255,.22);
  background:linear-gradient(135deg,#0e1b2f,#0b1424);
}
.metric-label{color:#9aa8bb;font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em;}
.metric-value{font-size:1.45rem;font-weight:900;color:#f4f8ff;margin-top:6px;}
.metric-help{color:#7f8da2;font-size:.75rem;margin-top:3px;}
.insight{
  border-left:4px solid #2f8cff;padding:14px 16px;background:#0d1b2f;border-radius:14px;margin:10px 0;color:#dbe8f8;
}
.login-card{
  border:1px solid rgba(88,181,255,.25);background:linear-gradient(135deg,#101d32,#07111f);
  border-radius:28px;padding:32px;box-shadow:0 24px 70px rgba(0,0,0,.35);
}
.footer{
  margin-top:34px;border-top:1px solid rgba(88,181,255,.14);padding:22px 0;color:#a9b7ca;
}
.footer-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;}
.footer-card{display:flex;gap:12px;align-items:center;}
.footer-icon{width:38px;height:38px;border-radius:13px;background:#102e55;display:flex;align-items:center;justify-content:center;font-weight:900;color:#75c2ff;}

@media (max-width: 1000px){
  .service-row{grid-template-columns:repeat(4,1fr);}
  .category-grid{grid-template-columns:repeat(4,1fr);}
}
@media (max-width: 720px){
  .service-row{grid-template-columns:repeat(2,1fr);}
  .category-grid{grid-template-columns:repeat(2,1fr);}
  .hero h1{font-size:2rem;}
}
</style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# =========================================================
# UTILITIES
# =========================================================
def safe_text(x, default="-"):
    if pd.isna(x):
        return default
    return str(x)

def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"

def compact_num(x):
    try:
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.1f}K"
        return f"{x:,.0f}"
    except Exception:
        return "0"

def initials(name):
    words = [w for w in str(name).replace("_", " ").split() if w]
    if not words:
        return "AM"
    return "".join(w[0].upper() for w in words[:2])

def prettify_cat(cat):
    return str(cat).replace("_", " ").title()

def html_block(s):
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)

def goto(view, **kwargs):
    st.session_state.view = view
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


# =========================================================
# DATA LOADING + FEATURE ENGINEERING
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    if not os.path.exists(path):
        path = DATA_FILE

    df_raw = pd.read_csv(path)
    df = df_raw.copy()

    # basic cleaning
    df["order_purchase_timestamp"] = pd.to_datetime(df.get("order_purchase_timestamp"), errors="coerce")
    df["product_category_name_english"] = df.get("product_category_name_english").fillna("unknown").astype(str)
    df["product_category_clean"] = df["product_category_name_english"].replace({"nan": "unknown"}).astype(str)
    df["product_name"] = df.get("product_name").fillna("Unknown Product").astype(str)
    df["product_id"] = df.get("product_id").fillna("UNKNOWN_PRODUCT").astype(str)
    df["order_id"] = df.get("order_id").fillna("UNKNOWN_ORDER").astype(str)
    df["customer_id"] = df.get("customer_id").fillna("UNKNOWN_CUSTOMER").astype(str)

    for col, default in [("price", 0), ("quantity", 1), ("rating", np.nan), ("stock", 0), ("discount_percent", 0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["quantity"] = df["quantity"].clip(lower=1)
    df["price"] = df["price"].clip(lower=0)
    df["discount_percent"] = df["discount_percent"].clip(lower=0, upper=95)
    df["rating"] = df["rating"].clip(lower=0, upper=5)

    df["gross_sales"] = df["price"] * df["quantity"]
    df["discount_value"] = df["gross_sales"] * df["discount_percent"] / 100
    df["net_sales"] = df["gross_sales"] - df["discount_value"]
    df["old_price"] = np.where(df["discount_percent"] > 0, df["price"] / (1 - df["discount_percent"] / 100), df["price"])
    df["month_date"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["order_purchase_timestamp"].dt.year
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["day_name"] = df["order_purchase_timestamp"].dt.day_name()
    df["hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_valid_sales"] = (df["price"] > 0) & (df["quantity"] > 0)

    bins = [-0.1, 0, 5, 10, 20, 35, 50, 100]
    labels = ["0%", "1-5%", "6-10%", "11-20%", "21-35%", "36-50%", ">50%"]
    df["discount_bucket"] = pd.cut(df["discount_percent"], bins=bins, labels=labels)

    valid = df[df["is_valid_sales"]].copy()

    catalog = (
        valid.groupby(["product_id", "product_name", "product_category_clean"], as_index=False)
        .agg(
            price=("price", "mean"),
            old_price=("old_price", "mean"),
            rating=("rating", "mean"),
            stock=("stock", "median"),
            sold=("quantity", "sum"),
            orders=("order_id", "nunique"),
            revenue=("net_sales", "sum"),
            gross_sales=("gross_sales", "sum"),
            discount_percent=("discount_percent", "mean"),
            customers=("customer_id", "nunique"),
        )
        .sort_values(["sold", "revenue", "rating"], ascending=False)
    )
    catalog["product_category_display"] = catalog["product_category_clean"].apply(prettify_cat)
    catalog["popularity_score"] = (
        catalog["sold"].rank(pct=True) * 0.35
        + catalog["orders"].rank(pct=True) * 0.25
        + catalog["revenue"].rank(pct=True) * 0.25
        + catalog["rating"].rank(pct=True) * 0.15
    )

    return df_raw, df, valid, catalog

df_raw, df, df_valid, catalog_df = load_data()


# =========================================================
# ANALYTICS FUNCTIONS
# =========================================================
@st.cache_data(show_spinner=False)
def category_summary(valid):
    return (
        valid.groupby("product_category_clean", as_index=False)
        .agg(
            total_order_items=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            total_products=("product_id", "nunique"),
            total_customers=("customer_id", "nunique"),
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
            avg_discount_percent=("discount_percent", "mean"),
            total_units=("quantity", "sum"),
            median_stock=("stock", "median"),
        )
        .assign(category=lambda x: x["product_category_clean"].apply(prettify_cat))
        .sort_values("net_sales", ascending=False)
    )

@st.cache_data(show_spinner=False)
def product_summary(valid):
    return (
        valid.groupby(["product_id", "product_name", "product_category_clean"], as_index=False)
        .agg(
            total_order_items=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            total_customers=("customer_id", "nunique"),
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
            median_stock=("stock", "median"),
            avg_discount_percent=("discount_percent", "mean"),
            total_units=("quantity", "sum"),
        )
        .assign(category=lambda x: x["product_category_clean"].apply(prettify_cat))
        .sort_values("net_sales", ascending=False)
    )

@st.cache_data(show_spinner=False)
def monthly_summary(valid):
    out = (
        valid.dropna(subset=["month_date"])
        .groupby("month_date", as_index=False)
        .agg(
            total_order_items=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            avg_rating=("rating", "mean"),
            avg_discount_percent=("discount_percent", "mean"),
            units=("quantity", "sum"),
        )
        .sort_values("month_date")
    )
    return out

@st.cache_data(show_spinner=False)
def discount_summary(valid):
    return (
        valid.groupby("discount_bucket", observed=False, as_index=False)
        .agg(
            total_order_items=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            avg_rating=("rating", "mean"),
            avg_price=("price", "mean"),
            units=("quantity", "sum"),
        )
    )

@st.cache_data(show_spinner=False)
def rfm_table(valid):
    max_date = valid["order_purchase_timestamp"].max()
    rfm = (
        valid.groupby("customer_id")
        .agg(
            last_order=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("net_sales", "sum"),
            units=("quantity", "sum"),
            categories=("product_category_clean", "nunique"),
        )
        .reset_index()
    )
    rfm["recency"] = (max_date - rfm["last_order"]).dt.days
    # Robust quantile scoring
    try:
        rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except Exception:
        rfm["R"] = rfm["F"] = rfm["M"] = 3
    rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["segment"] = np.select(
        [
            rfm["rfm_score"] >= 13,
            (rfm["R"] <= 2) & (rfm["F"] >= 4),
            (rfm["R"] >= 4) & (rfm["F"] <= 2),
            rfm["M"] >= 4,
        ],
        ["VIP / Loyal", "At Risk Loyal", "New / Promising", "High Value"],
        default="Regular",
    )
    return rfm.sort_values(["rfm_score", "monetary"], ascending=False)

@st.cache_data(show_spinner=False)
def forecast_monthly(valid, horizon=6):
    ts = monthly_summary(valid)[["month_date", "net_sales", "total_orders", "units"]].copy()
    ts = ts.dropna().sort_values("month_date")
    if len(ts) < 4:
        return ts, pd.DataFrame()
    # remove incomplete-looking last month if extreme drop
    if len(ts) >= 5 and ts["net_sales"].iloc[-1] < ts["net_sales"].iloc[:-1].median() * 0.25:
        ts_model = ts.iloc[:-1].copy()
    else:
        ts_model = ts.copy()

    y = ts_model["net_sales"].values.astype(float)
    x = np.arange(len(y))
    if len(y) >= 2:
        coef = np.polyfit(x, y, 1)
        trend = np.poly1d(coef)
    else:
        trend = lambda z: np.repeat(y.mean(), len(np.atleast_1d(z)))

    roll = pd.Series(y).rolling(3, min_periods=1).mean().values
    last_roll = roll[-1]
    std = np.std(y - np.poly1d(np.polyfit(x, y, 1))(x)) if len(y) >= 3 else np.std(y)

    future_dates = pd.date_range(ts_model["month_date"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future_x = np.arange(len(y), len(y) + horizon)
    base = 0.55 * trend(future_x) + 0.45 * last_roll
    base = np.maximum(base, 0)

    fc = pd.DataFrame({
        "month_date": future_dates,
        "forecast_net_sales": base,
        "lower_bound": np.maximum(base - 1.28 * std, 0),
        "upper_bound": base + 1.28 * std,
    })
    return ts, fc

@st.cache_data(show_spinner=False)
def association_rules(valid, top_n=50):
    # market basket association pairs
    baskets = valid.groupby("order_id")["product_id"].apply(lambda x: sorted(set(x))).reset_index(name="items")
    baskets["basket_size"] = baskets["items"].apply(len)
    multibasket = baskets[baskets["basket_size"] >= 2].copy()

    item_counter = Counter()
    pair_counter = Counter()
    for items in multibasket["items"]:
        item_counter.update(items)
        for a, b in combinations(items, 2):
            pair_counter[(a, b)] += 1

    total_baskets = max(len(baskets), 1)
    rows = []
    for (a, b), support_count in pair_counter.most_common(top_n * 4):
        support = support_count / total_baskets
        conf_a_to_b = support_count / max(item_counter[a], 1)
        conf_b_to_a = support_count / max(item_counter[b], 1)
        lift_a_to_b = conf_a_to_b / max(item_counter[b] / total_baskets, 1e-9)
        rows.append({
            "antecedent_id": a,
            "consequent_id": b,
            "support_count": support_count,
            "support": support,
            "confidence_a_to_b": conf_a_to_b,
            "confidence_b_to_a": conf_b_to_a,
            "lift": lift_a_to_b,
        })
    rules = pd.DataFrame(rows)
    if not rules.empty:
        names = catalog_df[["product_id", "product_name", "product_category_display"]]
        rules = rules.merge(names.rename(columns={"product_id":"antecedent_id", "product_name":"antecedent_name", "product_category_display":"antecedent_category"}), on="antecedent_id", how="left")
        rules = rules.merge(names.rename(columns={"product_id":"consequent_id", "product_name":"consequent_name", "product_category_display":"consequent_category"}), on="consequent_id", how="left")
        rules = rules.sort_values(["lift", "support_count"], ascending=False).head(top_n)
    return baskets, multibasket, rules

@st.cache_data(show_spinner=False)
def product_segments(valid):
    ps = product_summary(valid).copy()
    revenue_q70 = ps["net_sales"].quantile(0.70)
    orders_q70 = ps["total_orders"].quantile(0.70)
    orders_q30 = ps["total_orders"].quantile(0.30)
    stock_q25 = ps["median_stock"].quantile(0.25)
    stock_q75 = ps["median_stock"].quantile(0.75)

    ps["business_segment"] = np.select(
        [
            (ps["net_sales"] >= revenue_q70) & (ps["total_orders"] >= orders_q70),
            (ps["total_orders"] >= orders_q70) & (ps["median_stock"] <= stock_q25),
            (ps["total_orders"] <= orders_q30) & (ps["median_stock"] >= stock_q75),
            (ps["avg_rating"] < 3.7) & (ps["total_orders"] >= orders_q30),
        ],
        ["Star Product", "Restock Priority", "Slow Moving / Overstock", "Quality Risk"],
        default="Normal",
    )
    ps["recommended_action"] = np.select(
        [
            ps["business_segment"].eq("Star Product"),
            ps["business_segment"].eq("Restock Priority"),
            ps["business_segment"].eq("Slow Moving / Overstock"),
            ps["business_segment"].eq("Quality Risk"),
        ],
        [
            "Pertahankan exposure, jadikan produk hero campaign.",
            "Tambah stok dan jaga availability.",
            "Buat bundling, clearance sale, atau turunkan pembelian stok.",
            "Evaluasi supplier, deskripsi produk, dan kualitas.",
        ],
        default="Monitor berkala."
    )
    return ps.sort_values(["business_segment", "net_sales"], ascending=[True, False])

def get_kpi(valid):
    return {
        "total_rows_raw": len(df_raw),
        "total_rows_valid": len(valid),
        "total_orders": valid["order_id"].nunique(),
        "total_customers": valid["customer_id"].nunique(),
        "total_products": valid["product_id"].nunique(),
        "total_categories": valid["product_category_clean"].nunique(),
        "gross_sales": valid["gross_sales"].sum(),
        "net_sales": valid["net_sales"].sum(),
        "aov": valid.groupby("order_id")["net_sales"].sum().mean(),
        "avg_rating": valid["rating"].mean(),
        "avg_discount": valid["discount_percent"].mean(),
        "total_units": valid["quantity"].sum(),
    }


# =========================================================
# RECOMMENDER
# =========================================================
def filter_products(query="", category="Semua Produk", limit=32):
    d = catalog_df.copy()
    if category and category != "Semua Produk":
        d = d[d["product_category_display"] == category]
    if query:
        q = str(query).lower().strip()
        d = d[
            d["product_name"].str.lower().str.contains(q, na=False)
            | d["product_category_display"].str.lower().str.contains(q, na=False)
            | d["product_category_clean"].str.lower().str.contains(q, na=False)
        ]
    return d.sort_values(["popularity_score", "sold", "rating"], ascending=False).head(limit)

def flash_sale_products(limit=20):
    return catalog_df.sort_values(["discount_percent", "sold", "rating"], ascending=False).head(limit)

def recommendations_for_product(product_id, top_n=12):
    # 1) Market basket co-purchase
    orders = df_valid.loc[df_valid["product_id"] == product_id, "order_id"].unique()
    co = df_valid[df_valid["order_id"].isin(orders) & (df_valid["product_id"] != product_id)]
    if not co.empty:
        counts = co.groupby("product_id").agg(co_purchase_count=("order_id", "nunique")).reset_index()
        out = counts.merge(catalog_df, on="product_id", how="inner")
        if not out.empty:
            out["recommendation_reason"] = "Sering dibeli bersama produk ini"
            return out.sort_values(["co_purchase_count", "popularity_score"], ascending=False).head(top_n)

    # 2) Fallback: content/category + price proximity
    p = catalog_df[catalog_df["product_id"] == product_id]
    if p.empty:
        return catalog_df.head(top_n)
    p = p.iloc[0]
    d = catalog_df[catalog_df["product_id"] != product_id].copy()
    d["category_match"] = (d["product_category_clean"] == p["product_category_clean"]).astype(int)
    d["price_gap"] = (d["price"] - p["price"]).abs()
    d["score"] = d["category_match"] * 2 + d["popularity_score"] - d["price_gap"].rank(pct=True) * 0.25
    d["recommendation_reason"] = "Produk mirip berdasarkan kategori, harga, rating, dan popularitas"
    return d.sort_values("score", ascending=False).head(top_n)


# =========================================================
# STATE
# =========================================================
def init_state():
    defaults = {
        "view": "home",
        "logged_in": False,
        "role": "guest",
        "username": "Guest",
        "cart": [],
        "selected_product": None,
        "search_query": "",
        "selected_category": "Semua Produk",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()


# =========================================================
# UI COMPONENTS
# =========================================================
def top_header():
    left, right = st.columns([1, 1])
    with left:
        html_block("<div class='top-strip'><div>Seller Centre &nbsp;|&nbsp; Mulai Berjualan &nbsp;|&nbsp; Download</div></div>")
    with right:
        html_block("<div class='top-strip' style='justify-content:flex-end;'><div>Notifikasi &nbsp;&nbsp; Bantuan &nbsp;&nbsp; Bahasa Indonesia</div></div>")

    html_block("<div class='header-shell'>")
    c_logo, c_search, c_cart, c_login = st.columns([1.25, 3.7, .95, 1.05], vertical_alignment="center")
    with c_logo:
        html_block(
            f"""
            <div class='logo-wrap'>
              <div class='logo-mark'>AM</div>
              <div>
                <div class='logo-text'>Alradi <span>Mart</span></div>
                <div class='logo-caption'>Marketplace & Data Mining Dashboard</div>
              </div>
            </div>
            """
        )
    with c_search:
        q = st.text_input(
            "Cari produk",
            value=st.session_state.search_query,
            label_visibility="collapsed",
            placeholder="Cari produk, kategori, atau merek di Alradi Mart",
            key="header_search",
        )
        if q != st.session_state.search_query:
            st.session_state.search_query = q
            st.session_state.view = "search" if q else "home"
    with c_cart:
        if st.button(f"Keranjang ({len(st.session_state.cart)})", use_container_width=True):
            goto("cart")
    with c_login:
        if st.session_state.logged_in:
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.role = "guest"
                st.session_state.username = "Guest"
                st.session_state.view = "home"
                st.rerun()
        else:
            if st.button("Login", use_container_width=True):
                goto("login")
    html_block("</div>")

    if st.session_state.search_query:
        st.info(f"Menampilkan hasil pencarian untuk: **{st.session_state.search_query}**")

def product_card_html(row):
    name = safe_text(row.get("product_name"))[:58]
    price = money(row.get("price", 0))
    old_price = money(row.get("old_price", row.get("price", 0)))
    disc = int(round(float(row.get("discount_percent", 0))))
    rating = float(row.get("rating", 0))
    sold = int(float(row.get("sold", row.get("total_units", 0))))
    cat = safe_text(row.get("product_category_display", prettify_cat(row.get("product_category_clean", "-"))))
    ini = initials(name)
    return (
        f"<div class='product-card'>"
        f"<div class='product-art'><span class='discount'>-{disc}%</span>{ini}</div>"
        f"<div class='product-body'>"
        f"<div class='pname'>{name}</div>"
        f"<div class='price'>{price}</div>"
        f"<div class='old-price'>{old_price}</div>"
        f"<div class='saving'>Diskon {disc}% dari dataset</div>"
        f"<div class='pmeta'>Rating {rating:.1f} · {sold:,} terjual</div>"
        f"<div class='store'>Official {cat}</div>"
        f"</div></div>"
    )

def show_product_grid(data, title=None, subtitle=None, limit=20, cols=5, key_prefix="prod"):
    if title:
        html_block(f"<div class='section-title'>{title}</div>")
    if subtitle:
        html_block(f"<div class='section-sub'>{subtitle}</div>")
    data = data.head(limit).reset_index(drop=True)
    if data.empty:
        st.warning("Tidak ada produk yang cocok.")
        return
    for start in range(0, len(data), cols):
        row_cols = st.columns(cols)
        for j, col in enumerate(row_cols):
            idx = start + j
            if idx < len(data):
                item = data.iloc[idx]
                with col:
                    st.markdown(product_card_html(item), unsafe_allow_html=True)
                    if st.button("Lihat Detail", key=f"{key_prefix}_{idx}_{item.product_id}", use_container_width=True):
                        goto("detail", selected_product=item.product_id)

def services_bar():
    services = [
        ("TP", "Top-Up & Tagihan"), ("PU", "Pulsa"), ("DT", "Paket Data"), ("PL", "Listrik PLN"),
        ("BP", "BPJS"), ("VG", "Voucher Game"), ("TK", "Tiket Kereta"), ("•••", "Lainnya")
    ]
    cards = "".join(
        f"<div class='service-card'><div class='icon-circle'>{abbr}</div><div class='service-name'>{label}</div></div>"
        for abbr, label in services
    )
    html_block(f"<div class='service-row'>{cards}</div>")

def wallet_panel():
    html_block(
        """
        <div class='panel'>
          <h3 style='margin-top:0'>Akun & Promo</h3>
          <div class='wallet-row'><div class='wallet-icon'>$</div><div><div class='small-muted'>Saldo Alradi Mart</div><div class='wallet-val'>$125.00</div></div></div>
          <div class='wallet-row'><div class='wallet-icon'>C</div><div><div class='small-muted'>Alradi Coins</div><div class='wallet-val'>2,450</div></div></div>
          <div class='wallet-row'><div class='wallet-icon'>K</div><div><div class='small-muted'>Kupon Saya</div><div class='wallet-val'>12 tersedia</div></div></div>
        </div>
        """
    )
    if st.button("Lihat Semua Promo", use_container_width=True):
        st.toast("Demo promo: voucher tersimpan untuk simulasi marketplace.")

def keyword_panel():
    keywords = ["Laptop", "Skincare", "Air Fryer", "Tas Wanita", "Sepatu", "Baby", "Keyboard", "Parfum"]
    html_block(
        "<div class='panel'><h3 style='margin-top:0'>Trending Keywords</h3>"
        + "<div class='keyword-wrap'>"
        + "".join(f"<span class='chip'>{k}</span>" for k in keywords)
        + "</div></div>"
    )

def category_grid():
    cat_summary = category_summary(df_valid)
    top = cat_summary.sort_values("total_order_items", ascending=False).head(16).reset_index(drop=True)
    cards = ""
    for _, r in top.iterrows():
        cat = r["category"]
        cards += (
            f"<div class='cat-tile'>"
            f"<div class='cat-icon'>{initials(cat)}</div>"
            f"<div class='cat-name'>{cat}</div>"
            f"<div class='cat-count'>{int(r['total_products']):,} produk</div>"
            f"</div>"
        )
    html_block(f"<div class='category-box'><div class='category-head'>Kategori</div><div class='category-grid'>{cards}</div></div>")
    # Functional category buttons below grid, compact and hidden-looking
    st.caption("Pilih kategori untuk menampilkan produk terkait:")
    labels = ["Semua Produk"] + top["category"].tolist()
    chosen = st.selectbox("Kategori", labels, index=labels.index(st.session_state.selected_category) if st.session_state.selected_category in labels else 0, label_visibility="collapsed")
    if chosen != st.session_state.selected_category:
        st.session_state.selected_category = chosen
        st.session_state.view = "category" if chosen != "Semua Produk" else "home"
        st.rerun()

def footer():
    html_block(
        """
        <div class='footer'>
          <div class='footer-grid'>
            <div class='footer-card'><div class='footer-icon'>GO</div><div><b>Gratis Ongkir</b><br><span class='small-muted'>Simulasi benefit marketplace</span></div></div>
            <div class='footer-card'><div class='footer-icon'>AM</div><div><b>Pembayaran Aman</b><br><span class='small-muted'>Checkout demo tersimpan di sesi</span></div></div>
            <div class='footer-card'><div class='footer-icon'>RT</div><div><b>Rekomendasi Tepat</b><br><span class='small-muted'>Market basket + content fallback</span></div></div>
            <div class='footer-card'><div class='footer-icon'>BI</div><div><b>Business Insight</b><br><span class='small-muted'>Admin EDA & data mining</span></div></div>
          </div>
        </div>
        """
    )

def hero_and_sidebar():
    c_main, c_side = st.columns([3.25, .95])
    with c_main:
        html_block(
            """
            <div class='hero'>
              <div class='kicker'>Alradi Mart</div>
              <h1>Diskon terbaik dari dataset toko online</h1>
              <p>Harga, kategori, rating, stok, diskon, dan histori transaksi diambil dari dataset yang kamu lampirkan.</p>
            </div>
            """
        )
        services_bar()
    with c_side:
        wallet_panel()
        st.write("")
        keyword_panel()


# =========================================================
# PAGES: MARKETPLACE
# =========================================================
def login_page():
    top_header()
    c1, c2, c3 = st.columns([1, 1.15, 1])
    with c2:
        html_block(
            """
            <div class='login-card'>
              <div class='kicker'>Secure Login</div>
              <h1 style='font-size:2rem;margin:.4rem 0'>Masuk ke Alradi Mart</h1>
              <p class='small-muted'>Pilih role untuk masuk sebagai user marketplace atau admin analytics.</p>
            </div>
            """
        )
        role = st.radio("Masuk sebagai", ["User", "Admin"], horizontal=True)
        username = st.text_input("Username", value="admin" if role == "Admin" else "user")
        password = st.text_input("Password", value="admin123" if role == "Admin" else "user123", type="password")
        if st.button("Masuk", use_container_width=True, type="primary"):
            valid_login = (role == "User" and username == "user" and password == "user123") or (
                role == "Admin" and username == "admin" and password == "admin123"
            )
            if valid_login:
                st.session_state.logged_in = True
                st.session_state.role = role.lower()
                st.session_state.username = username
                st.session_state.view = "admin" if role == "Admin" else "home"
                st.rerun()
            else:
                st.error("Login gagal. Demo: user/user123 atau admin/admin123")

def home_page():
    top_header()
    st.caption("Dikirim ke Jakarta Pusat · Demo marketplace berbasis dataset transaksi Alradi Mart")
    hero_and_sidebar()
    st.write("")
    category_grid()
    st.write("")
    show_product_grid(flash_sale_products(20), "Flash Sale", "Produk dengan diskon, rating, dan transaksi terbaik.", 20, 5, "flash")
    show_product_grid(catalog_df.sort_values("popularity_score", ascending=False), "Produk Pilihan Untukmu", "Disusun dari penjualan, revenue, rating, dan popularitas.", 20, 5, "home")
    footer()

def search_or_category_page():
    top_header()
    if st.button("Kembali ke Beranda"):
        st.session_state.search_query = ""
        st.session_state.selected_category = "Semua Produk"
        goto("home")
    query = st.session_state.search_query
    cat = st.session_state.selected_category
    title = "Hasil Pencarian" if query else f"Kategori: {cat}"
    data = filter_products(query=query, category=cat, limit=60)
    show_product_grid(data, title, f"{len(data):,} produk ditampilkan dari dataset.", 40, 5, "result")
    footer()

def detail_page():
    top_header()
    pid = st.session_state.selected_product
    row = catalog_df[catalog_df["product_id"] == pid]
    if row.empty:
        st.warning("Produk tidak ditemukan.")
        if st.button("Kembali"):
            goto("home")
        return
    p = row.iloc[0]
    if st.button("Kembali ke Beranda"):
        goto("home")

    c1, c2 = st.columns([1, 1.35])
    with c1:
        html_block(
            f"""
            <div class='product-card' style='height:420px'>
              <div class='product-art' style='height:230px;font-size:4rem'><span class='discount'>-{int(round(p.discount_percent))}%</span>{initials(p.product_name)}</div>
              <div class='product-body'>
                <div class='store'>Official {p.product_category_display}</div>
                <div class='price' style='font-size:1.5rem'>{money(p.price)}</div>
                <div class='old-price'>{money(p.old_price)}</div>
              </div>
            </div>
            """
        )
    with c2:
        st.title(p.product_name)
        st.caption(f"Product ID: {p.product_id}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Harga", money(p.price))
        m2.metric("Rating", f"{p.rating:.1f}/5")
        m3.metric("Terjual", f"{int(p.sold):,}")
        m4.metric("Stok", f"{int(p.stock):,}")
        st.write(
            f"Produk ini termasuk kategori **{p.product_category_display}**. "
            "Data harga, rating, stok, diskon, dan penjualan diambil dari dataset transaksi toko."
        )
        qty = st.number_input("Jumlah", min_value=1, max_value=max(int(p.stock), 1), value=1, step=1)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Masukkan Keranjang", use_container_width=True):
                item = p.to_dict()
                item["quantity"] = int(qty)
                st.session_state.cart.append(item)
                st.success("Produk masuk keranjang.")
        with b2:
            if st.button("Beli Sekarang", use_container_width=True):
                item = p.to_dict()
                item["quantity"] = int(qty)
                st.session_state.cart.append(item)
                goto("cart")

    rec = recommendations_for_product(pid, top_n=15)
    st.write("")
    show_product_grid(rec, "Rekomendasi Produk", "Jika pembeli melihat/membeli produk ini, sistem merekomendasikan produk yang sering dibeli bersama atau produk yang mirip.", 15, 5, "rec")
    footer()

def cart_page():
    top_header()
    st.title("Keranjang Belanja")
    if st.button("Kembali ke Beranda"):
        goto("home")
    if not st.session_state.cart:
        st.info("Keranjang masih kosong.")
        return
    cart = pd.DataFrame(st.session_state.cart)
    cart["subtotal"] = cart["price"] * cart["quantity"]
    for i, item in enumerate(st.session_state.cart):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"**{item['product_name']}**")
            st.caption(f"{item.get('product_category_display','')} · {money(item['price'])} × {item['quantity']}")
        with c2:
            st.write(money(item["price"] * item["quantity"]))
        with c3:
            if st.button("Hapus", key=f"remove_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()
        st.divider()
    st.metric("Total", money(cart["subtotal"].sum()))
    if st.button("Checkout Demo", use_container_width=True):
        st.success("Checkout berhasil disimulasikan. Data belanja tersimpan selama sesi aplikasi.")
        st.session_state.cart = []

# =========================================================
# PAGES: ADMIN ANALYTICS
# =========================================================
def metric_card(label, value, help_text=""):
    html_block(
        f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{value}</div>
          <div class='metric-help'>{help_text}</div>
        </div>
        """
    )

def admin_header():
    top_header()
    if st.session_state.role != "admin":
        st.warning("Halaman admin hanya untuk role Admin.")
        if st.button("Login Admin"):
            goto("login")
        return False
    html_block(
        """
        <div class='hero' style='min-height:190px'>
          <div class='kicker'>Admin Analytics</div>
          <h1>Dashboard Analisis Lengkap Alradi Mart</h1>
          <p>Berisi EDA, KPI, kategori, produk, waktu transaksi, diskon, rating, stok, segmentasi, forecasting, market basket, content-based, hybrid recommender, dan insight bisnis.</p>
        </div>
        """
    )
    return True

def admin_dashboard():
    if not admin_header():
        return

    K = get_kpi(df_valid)
    tabs = st.tabs([
        "Ringkasan KPI",
        "Data Quality",
        "Kategori",
        "Produk",
        "Waktu",
        "Diskon & Rating",
        "Stok & Segmentasi",
        "Forecasting",
        "Recommender",
        "Customer/RFM",
        "Insight & Export",
    ])

    with tabs[0]:
        st.subheader("KPI Utama Toko")
        cols = st.columns(6)
        values = [
            ("Net Sales", money(K["net_sales"]), "revenue setelah diskon"),
            ("Orders", f"{K['total_orders']:,}", "order unik"),
            ("Customers", f"{K['total_customers']:,}", "customer unik"),
            ("Products", f"{K['total_products']:,}", "produk unik"),
            ("AOV", money(K["aov"]), "rata-rata order value"),
            ("Avg Rating", f"{K['avg_rating']:.2f}", "rata-rata rating"),
        ]
        for col, (lab, val, help_text) in zip(cols, values):
            with col:
                metric_card(lab, val, help_text)
        st.write("")
        c1, c2 = st.columns([1.4, 1])
        ms = monthly_summary(df_valid)
        cs = category_summary(df_valid)
        with c1:
            fig = px.line(ms, x="month_date", y="net_sales", markers=True, title="Monthly Net Sales")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(cs.head(8), names="category", values="net_sales", title="Revenue Share Top 8 Kategori")
            st.plotly_chart(fig, use_container_width=True)
        html_block("<div class='insight'><b>Interpretasi:</b> KPI ini menjawab kondisi umum toko: omzet bersih, jumlah order, jumlah customer, rata-rata nilai order, dan kualitas rating.</div>")

    with tabs[1]:
        st.subheader("Data Quality & Feature Engineering")
        info_df = pd.DataFrame({
            "column": df_raw.columns,
            "dtype": [str(df_raw[c].dtype) for c in df_raw.columns],
            "missing_count": [df_raw[c].isna().sum() for c in df_raw.columns],
            "missing_percent": [df_raw[c].isna().mean() * 100 for c in df_raw.columns],
            "unique_count": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
        }).sort_values("missing_percent", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Raw", f"{len(df_raw):,}")
        c2.metric("Rows Valid Revenue", f"{len(df_valid):,}")
        c3.metric("Columns", f"{df_raw.shape[1]:,}")
        c4.metric("Missing Cells", f"{int(df_raw.isna().sum().sum()):,}")
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        html_block(
            """
            <div class='insight'>
            <b>Feature engineering yang dipakai:</b> gross_sales = price × quantity, discount_value = gross_sales × discount_percent,
            net_sales = gross_sales - discount_value, old_price estimasi harga sebelum diskon, month_date, day_name, hour, dan discount_bucket.
            </div>
            """
        )

    with tabs[2]:
        st.subheader("Analisis Kategori")
        cs = category_summary(df_valid)
        c1, c2 = st.columns([1.35, 1])
        with c1:
            fig = px.bar(cs.head(15), x="category", y="net_sales", color="avg_rating", title="Top Kategori Berdasarkan Net Sales")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(cs, x="total_orders", y="net_sales", size="total_products", color="avg_rating", hover_name="category", title="Peta Kategori: Orders vs Net Sales")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cs, use_container_width=True, hide_index=True)
        html_block("<div class='insight'><b>Tujuan:</b> menentukan kategori penyumbang omzet terbesar, kategori paling ramai, dan kategori yang layak diprioritaskan untuk campaign.</div>")

    with tabs[3]:
        st.subheader("Analisis Produk")
        ps = product_summary(df_valid)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.bar(ps.head(20), x="product_name", y="net_sales", color="avg_rating", title="Top Produk Berdasarkan Net Sales")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(ps.head(1000), x="total_orders", y="net_sales", size="total_units", color="avg_rating", hover_name="product_name", title="Produk: Orders vs Revenue")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ps.head(300), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Analisis Waktu Transaksi")
        ms = monthly_summary(df_valid)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(ms, x="month_date", y=["net_sales", "gross_sales"], markers=True, title="Gross vs Net Sales Bulanan"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(ms, x="month_date", y="total_orders", title="Jumlah Order Bulanan"), use_container_width=True)
        hour = df_valid.groupby("hour", as_index=False).agg(orders=("order_id", "nunique"), net_sales=("net_sales", "sum"))
        day = df_valid.groupby("day_name", as_index=False).agg(orders=("order_id", "nunique"), net_sales=("net_sales", "sum"))
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.bar(hour, x="hour", y="orders", title="Order Berdasarkan Jam"), use_container_width=True)
        c4.plotly_chart(px.bar(day, x="day_name", y="orders", title="Order Berdasarkan Hari"), use_container_width=True)
        html_block("<div class='insight'><b>Tujuan:</b> menemukan bulan, hari, dan jam transaksi yang paling ramai untuk perencanaan promosi dan kapasitas operasional.</div>")

    with tabs[5]:
        st.subheader("Analisis Diskon dan Risiko Rating")
        ds = discount_summary(df_valid)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(ds, x="discount_bucket", y="net_sales", title="Net Sales per Bucket Diskon"), use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(ds, x="avg_price", y="net_sales", size="total_orders", color="avg_rating", title="Efektivitas Diskon"), use_container_width=True)
        cs = category_summary(df_valid)
        risk = cs[(cs["total_orders"] >= cs["total_orders"].quantile(.50))].sort_values("avg_rating").head(15)
        st.write("Kategori besar dengan rating relatif rendah")
        st.dataframe(risk[["category", "total_orders", "net_sales", "avg_rating", "avg_discount_percent"]], use_container_width=True, hide_index=True)
        html_block("<div class='insight'><b>Tujuan:</b> mengecek apakah diskon besar benar-benar berdampak pada sales dan menemukan kategori laku yang kualitas/ratingnya perlu dijaga.</div>")

    with tabs[6]:
        st.subheader("Analisis Stok dan Segmentasi Produk")
        seg = product_segments(df_valid)
        restock = seg[seg["business_segment"] == "Restock Priority"].head(50)
        overstock = seg[seg["business_segment"] == "Slow Moving / Overstock"].head(50)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(seg, x="business_segment", title="Distribusi Segmentasi Produk"), use_container_width=True)
        c2.plotly_chart(px.scatter(seg.head(2000), x="median_stock", y="total_orders", color="business_segment", hover_name="product_name", title="Stok vs Order"), use_container_width=True)
        st.write("Prioritas Restock")
        st.dataframe(restock[["product_name", "category", "total_orders", "net_sales", "median_stock", "recommended_action"]], use_container_width=True, hide_index=True)
        st.write("Slow Moving / Overstock")
        st.dataframe(overstock[["product_name", "category", "total_orders", "net_sales", "median_stock", "recommended_action"]], use_container_width=True, hide_index=True)

    with tabs[7]:
        st.subheader("Forecasting Penjualan 6 Bulan ke Depan")
        ts, fc = forecast_monthly(df_valid, 6)
        fig = go.Figure()
        if not ts.empty:
            fig.add_trace(go.Scatter(x=ts["month_date"], y=ts["net_sales"], mode="lines+markers", name="Historical Net Sales"))
        if not fc.empty:
            fig.add_trace(go.Scatter(x=fc["month_date"], y=fc["forecast_net_sales"], mode="lines+markers", name="Forecast"))
            fig.add_trace(go.Scatter(x=fc["month_date"], y=fc["upper_bound"], mode="lines", name="Upper Bound", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=fc["month_date"], y=fc["lower_bound"], mode="lines", name="Lower Bound", line=dict(dash="dot")))
        fig.update_layout(title="Forecast Net Sales Bulanan", yaxis_title="Net Sales")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fc.assign(
            forecast_net_sales=fc["forecast_net_sales"].map(money) if not fc.empty else [],
            lower_bound=fc["lower_bound"].map(money) if not fc.empty else [],
            upper_bound=fc["upper_bound"].map(money) if not fc.empty else [],
        ), use_container_width=True, hide_index=True)
        html_block("<div class='insight'><b>Metode:</b> forecasting sederhana berbasis tren linear dan rolling mean 3 bulan. Ini cocok sebagai baseline untuk tugas data mining dan dapat dijelaskan dengan mudah.</div>")

    with tabs[8]:
        st.subheader("Sistem Rekomendasi Produk")
        baskets, multibasket, rules = association_rules(df_valid, top_n=50)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Order Basket", f"{len(baskets):,}")
        c2.metric("Basket >= 2 Produk", f"{len(multibasket):,}")
        c3.metric("Association Rules", f"{len(rules):,}")
        if not rules.empty:
            st.dataframe(
                rules[["antecedent_name", "consequent_name", "support_count", "support", "confidence_a_to_b", "lift", "antecedent_category", "consequent_category"]],
                use_container_width=True,
                hide_index=True,
            )
            st.plotly_chart(px.scatter(rules, x="support", y="confidence_a_to_b", size="support_count", color="lift", hover_name="antecedent_name", title="Association Rules: Support vs Confidence"), use_container_width=True)
        st.write("Coba rekomendasi hybrid berdasarkan produk:")
        product_choice = st.selectbox("Pilih produk", catalog_df["product_id"].head(3000).tolist(), format_func=lambda x: catalog_df.loc[catalog_df["product_id"] == x, "product_name"].iloc[0])
        rec = recommendations_for_product(product_choice, top_n=10)
        show_product_grid(rec, "Output Rekomendasi Hybrid", "Market basket digunakan jika ada pasangan transaksi; jika tidak ada, fallback ke content-based similarity.", 10, 5, "admin_rec")
        html_block("<div class='insight'><b>Metode:</b> rekomendasi memakai Market Basket/Association Rules untuk produk yang sering dibeli bersama, lalu Content-Based sebagai fallback berdasarkan kategori, harga, rating, dan popularitas.</div>")

    with tabs[9]:
        st.subheader("Customer Segmentation / RFM")
        rfm = rfm_table(df_valid)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.plotly_chart(px.histogram(rfm, x="segment", title="Distribusi Segment Customer"), use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(rfm.head(3000), x="recency", y="monetary", color="segment", size="frequency", hover_name="customer_id", title="RFM: Recency vs Monetary"), use_container_width=True)
        if StandardScaler is not None and KMeans is not None and len(rfm) >= 10:
            X = rfm[["recency", "frequency", "monetary", "categories"]].fillna(0)
            scaled = StandardScaler().fit_transform(X)
            rfm["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)
            st.write("KMeans Customer Cluster")
            st.plotly_chart(px.scatter(rfm.head(3000), x="recency", y="monetary", color="cluster", size="frequency", hover_name="customer_id", title="KMeans Segmentation"), use_container_width=True)
        st.dataframe(rfm.head(300), use_container_width=True, hide_index=True)

    with tabs[10]:
        st.subheader("Ringkasan Insight dan Export")
        cs = category_summary(df_valid)
        ps = product_summary(df_valid)
        seg = product_segments(df_valid)
        top_cat = cs.iloc[0]
        top_product = ps.iloc[0]
        restock_count = (seg["business_segment"] == "Restock Priority").sum()
        quality_count = (seg["business_segment"] == "Quality Risk").sum()
        insights = [
            f"Kategori terbesar adalah {top_cat['category']} dengan net sales {money(top_cat['net_sales'])} dan {int(top_cat['total_orders']):,} order.",
            f"Produk terbesar berdasarkan net sales adalah {top_product['product_name']} dengan net sales {money(top_product['net_sales'])}.",
            f"AOV toko adalah {money(K['aov'])}, sehingga campaign bundling dapat diarahkan untuk menaikkan rata-rata nilai order.",
            f"Terdapat {restock_count:,} produk yang masuk Restock Priority dan perlu dijaga stoknya.",
            f"Terdapat {quality_count:,} produk/kasus Quality Risk yang perlu evaluasi rating dan kualitas.",
            "Sistem rekomendasi menggunakan market basket untuk menangkap pola produk yang sering dibeli bersama, lalu content-based fallback untuk produk yang tidak memiliki pasangan transaksi kuat.",
        ]
        for x in insights:
            html_block(f"<div class='insight'>{x}</div>")

        st.download_button("Download Category Summary CSV", cs.to_csv(index=False).encode("utf-8"), "category_summary.csv", "text/csv")
        st.download_button("Download Product Summary CSV", ps.to_csv(index=False).encode("utf-8"), "product_summary.csv", "text/csv")
        st.download_button("Download Product Segmentation CSV", seg.to_csv(index=False).encode("utf-8"), "product_segmentation.csv", "text/csv")
        ts, fc = forecast_monthly(df_valid, 6)
        st.download_button("Download Forecast CSV", fc.to_csv(index=False).encode("utf-8"), "forecast_6_months.csv", "text/csv")


# =========================================================
# ROUTER
# =========================================================
def main():
    if st.session_state.view == "login":
        login_page()
    elif st.session_state.view == "detail":
        detail_page()
    elif st.session_state.view == "cart":
        cart_page()
    elif st.session_state.view == "admin":
        admin_dashboard()
    elif st.session_state.view in ["search", "category"]:
        search_or_category_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
