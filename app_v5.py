import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from api.regional_ai import regionalInsights, regionalRecommend

# -------------------------
# INITIATE SESSION STATE FOR AI GENERATION 
# -------------------------

# Session state for latest AI outputs
if "insights_latest" not in st.session_state:
    st.session_state["insights_latest"] = None
if "recs_latest" not in st.session_state:
    st.session_state["recs_latest"] = None


# === ASSUMING YOU HAVE THESE TWO FUNCTIONS DEFINED SOMEWHERE ===
# If not, here are minimal stubs you can replace with your real GenAI calls

# -------------------------
# PAGE CONFIG & HELPER
# -------------------------
st.set_page_config(page_title="Supplier Analytics Dashboard", layout="wide")

def format_millions(value):
    if pd.isna(value) or value == 0:
        return "$0"
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"

# -------------------------
# DATA LOADING & PARSING
# -------------------------
@st.cache_data
def load_and_parse_data(data_path: str):
    # Load all raw files
    df_sup_raw = pd.read_csv(data_path + "supplier_master.csv", sep=";")
    df_prod_raw = pd.read_csv(data_path + "supplier_product.csv", sep=";")
    df_trade_raw = pd.read_csv(data_path + "trade_flow.csv", sep=";")
    df_tariff_raw = pd.read_csv(data_path + "trade_tariff.csv", sep=";")
    df_price_raw = pd.read_csv(data_path + "supplier_price.csv", sep=";")
    df_news_raw = pd.read_csv(data_path + "news_master.csv", sep=";")
    df_reg_ins_raw = pd.read_csv(data_path + "regional_insight.csv", sep=";")
    df_reg_rec_raw = pd.read_csv(data_path + "regional_recommend.csv", sep=";")
    df_sup_ins_raw = pd.read_csv(data_path + "supplier_insight.csv", sep=";")
    df_sup_rec_raw = pd.read_csv(data_path + "supplier_recommend.csv", sep=";")
    df_com_ins_raw = pd.read_csv(data_path + "compliance_insight.csv", sep=";")
    df_com_rec_raw = pd.read_csv(data_path + "compliance_recommend.csv", sep=";")

    # === Parse supplier_product ===
    if len(df_prod_raw.columns) == 1:
        df_prod = df_prod_raw.iloc[:, 0].str.split(';', expand=True)
        df_prod.columns = ['supplier_id', 'product_category']
    else:
        df_prod = df_prod_raw.copy()

    # === Parse trade_flow ===
    if len(df_trade_raw.columns) == 1:
        df_trade = df_trade_raw.iloc[:, 0].str.split(';', expand=True)
        df_trade.columns = ['report_id', 'date', 'importer', 'exporter', 'product', 'uom', 'volume', 'currency', 'value', 'unit_price']
    else:
        df_trade = df_trade_raw.copy()

    # === Parse tariff ===
    if len(df_tariff_raw.columns) == 1:
        df_tariff = df_tariff_raw.iloc[:, 0].str.split(';', expand=True)
        df_tariff.columns = ['tariff_id', 'start_date', 'importing_country', 'product_category', 'exporting_country', 'tariff_pct']
    else:
        df_tariff = df_tariff_raw.copy()

    # === Parse news ===
    if len(df_news_raw.columns) == 1:
        df_news = df_news_raw.iloc[:, 0].str.split(';', expand=True)
        df_news.columns = ['news_id', 'news_date', 'impacted_country', 'news_category', 'news_title', 'news_summary']
    else:
        df_news = df_news_raw.copy()

    # === SAFE & ROBUST INSIGHT PARSING FUNCTION ===
    def parse_insight(df_raw, id_col, desc_col):
        if df_raw.empty:
            return pd.DataFrame(columns=[id_col, 'timestamp', 'region', 'country', desc_col])

        if len(df_raw.columns) == 1:
            col_name = df_raw.columns[0]
            if df_raw[col_name].dtype == 'object' and df_raw[col_name].str.contains(';').any():
                expanded = df_raw[col_name].str.split(';', expand=True)
                if expanded.shape[1] >= 5:
                    expanded = expanded.iloc[:, :5]
                    expanded.columns = [id_col, 'timestamp', 'region', 'country', desc_col]
                    df = expanded
                else:
                    df = pd.DataFrame(columns=[id_col, 'timestamp', 'region', 'country', desc_col])
            else:
                df = pd.DataFrame(columns=[id_col, 'timestamp', 'region', 'country', desc_col])
        else:
            df = df_raw.copy()
            expected = [id_col, 'timestamp', 'region', 'country', desc_col]
            if list(df.columns[:5]) != expected:
                df.columns = expected + list(df.columns[5:])

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

    # === Apply parsing to all insight files ===
    df_reg_ins = parse_insight(df_reg_ins_raw, 'insight_id', 'insight_desc')
    df_reg_rec = parse_insight(df_reg_rec_raw, 'recommend_id', 'recommend_desc')
    df_sup_ins = parse_insight(df_sup_ins_raw, 'insight_id', 'insight_desc')
    df_sup_rec = parse_insight(df_sup_rec_raw, 'recommend_id', 'recommend_desc')
    df_com_ins = parse_insight(df_com_ins_raw, 'insight_id', 'insight_desc')
    df_com_rec = parse_insight(df_com_rec_raw, 'recommend_id', 'recommend_desc')

    # === Type conversions ===
    if "date" in df_trade.columns:
        df_trade["date"] = pd.to_datetime(df_trade["date"], format="%Y.%m.%d", errors="coerce")
    for col in ['volume', 'value', 'unit_price']:
        if col in df_trade.columns:
            df_trade[col] = pd.to_numeric(df_trade[col], errors="coerce").fillna(0)
    if "start_date" in df_tariff.columns:
        df_tariff["start_date"] = pd.to_datetime(df_tariff["start_date"], format="%Y.%m.%d", errors="coerce")
    if "tariff_pct" in df_tariff.columns:
        df_tariff["tariff_pct"] = pd.to_numeric(df_tariff["tariff_pct"], errors="coerce")
    if "news_date" in df_news.columns:
        df_news["news_date"] = pd.to_datetime(df_news["news_date"], format="%Y.%m.%d", errors="coerce")

    # Map supplier_id if missing
    if 'supplier_id' not in df_trade.columns and 'country' in df_sup_raw.columns:
        country_to_sup = df_sup_raw.groupby('country')['supplier_id'].first().to_dict()
        df_trade['supplier_id'] = df_trade['exporter'].map(country_to_sup)

    return (
        df_sup_raw, df_prod, df_trade, df_tariff, df_price_raw, df_news,
        df_reg_ins, df_reg_rec, df_sup_ins, df_sup_rec, df_com_ins, df_com_rec
    )

# -------------------------
# SUPPLIER PROCESSING
# -------------------------
@st.cache_data
def sanitize_suppliers(df_sup: pd.DataFrame) -> pd.DataFrame:
    df = df_sup.copy()
    cols = ["certified_audited", "code_conduct", "timely_delivery", "product_quality",
            "service_reliability", "esg_environment", "esg_social", "esg_governance"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    df["supplier_id"] = df["supplier_id"].astype(str)
    return df

@st.cache_data
def calculate_scores(df_sup: pd.DataFrame) -> pd.DataFrame:
    df = df_sup.copy()
    df["compliance_raw"] = df["certified_audited"] * 0.75 + df["code_conduct"] * 0.25
    df["compliance_score"] = df["compliance_raw"] * 0.4
    df["performance_raw"] = df[["timely_delivery", "product_quality", "service_reliability"]].mean(axis=1)
    df["performance_score"] = df["performance_raw"] * 0.3
    df["esg_raw"] = df["esg_environment"] * 0.4 + df["esg_social"] * 0.4 + df["esg_governance"] * 0.2
    df["sustainability_score"] = df["esg_raw"] * 0.3
    df["supplier_score"] = df["compliance_score"] + df["performance_score"] + df["sustainability_score"]
    df["risk_rating"] = df["supplier_score"].apply(lambda x: "Low" if x >= 75 else "Medium" if x >= 50 else "High")
    df["supplier_score"] = df["supplier_score"].round(2)
    return df

@st.cache_data
def add_coordinates(df_sup: pd.DataFrame) -> pd.DataFrame:
    df = df_sup.copy()
    coords = {
        "Bangladesh": (23.6850, 90.3563), "China": (35.8617, 104.1954), "France": (46.2276, 2.2137),
        "Germany": (51.1657, 10.4515), "India": (20.5937, 78.9629), "Indonesia": (-0.7893, 113.9213),
        "Italy": (41.8719, 12.5674), "Japan": (36.2048, 138.2529), "Pakistan": (30.3753, 69.3451),
        "South Korea": (35.9078, 127.7669), "United Kingdom": (55.3781, 3.4360),
        "Thailand": (15.8700, 100.9925), "United States": (37.0902, -95.7129), "Vietnam": (14.0583, 108.2772)
    }
    df["latitude"] = df["country"].map(lambda x: coords.get(x, (None, None))[0])
    df["longitude"] = df["country"].map(lambda x: coords.get(x, (None, None))[1])
    return df

# -------------------------
# FILTERING
# -------------------------
def apply_filters(df_sup, df_trade, df_prod, filters):
    df_sup_f = df_sup.copy()
    df_trade_f = df_trade.copy()
    df_prod_f = df_prod.copy()

    if filters['date_range'] and 'date' in df_trade_f.columns:
        start, end = filters['date_range']
        df_trade_f = df_trade_f[
            (df_trade_f['date'] >= pd.Timestamp(start)) &
            (df_trade_f['date'] <= pd.Timestamp(end))
        ]

    if filters['categories'] and 'All' not in filters['categories']:
        df_prod_f = df_prod_f[df_prod_f['product_category'].isin(filters['categories'])]
        ids = df_prod_f['supplier_id'].astype(str).unique()
        df_sup_f = df_sup_f[df_sup_f['supplier_id'].isin(ids)]
        df_trade_f = df_trade_f[df_trade_f['supplier_id'].isin(ids)]

    if filters['regions'] and 'All' not in filters['regions']:
        df_sup_f = df_sup_f[df_sup_f['region'].isin(filters['regions'])]
    if filters['import_countries'] and 'All' not in filters['import_countries']:
        df_trade_f = df_trade_f[df_trade_f['importer'].isin(filters['import_countries'])]
    if filters['export_countries'] and 'All' not in filters['export_countries']:
        df_trade_f = df_trade_f[df_trade_f['exporter'].isin(filters['export_countries'])]

    # 🔧 NEW: make sure supplier_score is numeric before comparing
    df_sup_f["supplier_score"] = pd.to_numeric(df_sup_f["supplier_score"], errors="coerce")

    df_sup_f = df_sup_f[
        (df_sup_f['supplier_score'] >= float(filters['score_range'][0])) &
        (df_sup_f['supplier_score'] <= float(filters['score_range'][1]))
    ]

    # Keep ALL filtered suppliers; restrict trade rows to those suppliers only
    if 'supplier_id' in df_trade_f.columns:
        valid_ids = df_sup_f['supplier_id'].unique()
        df_trade_f = df_trade_f[df_trade_f['supplier_id'].isin(valid_ids)]

    return df_sup_f, df_trade_f, df_prod_f



# -------------------------
# CHART FUNCTIONS (ALL IMPLEMENTED & SAFE)
# -------------------------
def create_time_series_chart(df_trade: pd.DataFrame):
    if df_trade.empty or 'date' not in df_trade.columns or df_trade['date'].isna().all():
        return None
    df_plot = df_trade.groupby('date').agg({'volume': 'sum', 'value': 'sum'}).reset_index()
    if df_plot.empty:
        return None

    # Convert value into millions
    df_plot['value_m'] = df_plot['value'] / 1_000_000

    base = alt.Chart(df_plot).encode(x=alt.X('date:T', title='Date'))
    volume = base.mark_line(color='steelblue').encode(
        y=alt.Y('volume:Q', title='Volume (MT)', axis=alt.Axis(titleColor='steelblue'))
    )
    value = base.mark_line(color='orange').encode(
        y=alt.Y('value_m:Q', title='Value ($M)', axis=alt.Axis(titleColor='orange'))
    )
    return alt.layer(volume, value).resolve_scale(y='independent').properties(height=350)


def create_benchmark_chart(df_sup: pd.DataFrame, df_prod: pd.DataFrame):
    if len(df_sup) < 10 or df_prod.empty:
        return None
    top_suppliers = df_sup.nlargest(10, 'supplier_score')[['supplier_id', 'supplier_name', 'supplier_score', 'country', 'region']]
    sup_with_cat = df_sup.merge(df_prod[['supplier_id', 'product_category']], on='supplier_id', how='left')
    cat_avg = sup_with_cat.groupby('product_category')['supplier_score'].mean().to_dict()
    reg_avg = df_sup.groupby('region')['supplier_score'].mean().to_dict()
    top_suppliers = top_suppliers.merge(df_prod[['supplier_id', 'product_category']], on='supplier_id', how='left')
    top_suppliers['category_avg'] = top_suppliers['product_category'].map(cat_avg)
    top_suppliers['region_avg'] = top_suppliers['region'].map(reg_avg)
    df_melted = top_suppliers.melt(id_vars=['supplier_id', 'supplier_name'],
                                   value_vars=['supplier_score', 'category_avg', 'region_avg'],
                                   var_name='Metric', value_name='Score')
    return alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Score:Q'), y=alt.Y('supplier_name:N', sort='-x'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['supplier_score', 'category_avg', 'region_avg'],
                                                   range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
        tooltip=['supplier_name', 'Metric', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(height=400)

def create_radar_chart(supplier_id: str, df_sup: pd.DataFrame):
    supplier = df_sup[df_sup['supplier_id'] == supplier_id]
    if supplier.empty:
        return None
    sup = supplier.iloc[0]
    categories = ['Compliance', 'Performance', 'ESG']
    values = [sup['compliance_score']/0.4, sup['performance_score']/0.3, sup['sustainability_score']/0.3]
    values += [values[0]]
    categories += [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=sup.get('supplier_name', supplier_id)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
    return fig

def create_tariff_chart(df_tariff: pd.DataFrame):
    """
    For each exporting country, find the row with the latest start_date
    and plot its tariff_pct as the current applicable tariff.
    """
    if df_tariff.empty:
        return None

    required_cols = {"start_date", "exporting_country", "tariff_pct"}
    if not required_cols.issubset(df_tariff.columns):
        st.warning("Tariff data is missing required columns.")
        return None

    df = df_tariff.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"])

    if df.empty:
        return None

    # Sort by date and keep the most recent row per exporting_country
    df_latest = (
        df.sort_values("start_date")
          .groupby("exporting_country", as_index=False)
          .last()
    )

    chart = (
        alt.Chart(df_latest)
        .mark_bar()
        .encode(
            y=alt.Y("exporting_country:N", sort="-x", title="Exporting Country"),
            x=alt.X("tariff_pct:Q", title="Latest Tariff (%)"),
            color=alt.Color(
                "tariff_pct:Q",
                title="Tariff (%)",
                scale=alt.Scale(scheme="reds")  # gradient from light to dark
            ),
            tooltip=[
                alt.Tooltip("exporting_country:N", title="Country"),
                alt.Tooltip("tariff_pct:Q", title="Tariff (%)", format=".2f"),
                alt.Tooltip("start_date:T", title="Effective From"),
            ],
        )
        .properties(
            height=350,
            title="Latest Applicable Tariff Rate by Exporting Country",
        )
    )

    return chart





def check_data_quality(df_sup, df_trade):
    return {
        'supplier_missing': df_sup.isnull().sum().to_dict(),
        'trade_missing': df_trade.isnull().sum().to_dict(),
        'supplier_duplicates': df_sup.duplicated(subset=['supplier_id']).sum(),
        'trade_duplicates': df_trade.duplicated(subset=['report_id']).sum() if 'report_id' in df_trade.columns else 0,
        'price_outliers': 0,
        'negative_volumes': (df_trade['volume'] < 0).sum() if 'volume' in df_trade.columns else 0
    }

def fix_duplicates(df, subset_cols):
    return df.drop_duplicates(subset=subset_cols, keep='first')

def fill_missing_with_median(df, cols):
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

def explain_supplier(supplier_id: str, df_sup: pd.DataFrame, df_trade: pd.DataFrame) -> str:
    supplier = df_sup[df_sup['supplier_id'] == supplier_id]
    if supplier.empty:
        return "Supplier not found."
    sup = supplier.iloc[0]
    explanation = f"### Supplier: {sup.get('supplier_name', 'Unknown')} ({supplier_id})\n\n"
    explanation += f"**Overall Score: {sup['supplier_score']:.2f}/100** - {sup['risk_rating']} Risk\n\n"
    explanation += "**Score Breakdown:**\n"
    explanation += f"- Compliance: {sup['compliance_score']:.2f}/40\n"
    explanation += f"- Performance: {sup['performance_score']:.2f}/30\n"
    explanation += f"- ESG: {sup['sustainability_score']:.2f}/30\n\n"
    if sup['supplier_score'] >= 75:
        explanation += "**Assessment:** Strong performer with low risk profile.\n"
    elif sup['supplier_score'] >= 50:
        explanation += "**Assessment:** Moderate performance. Monitor closely.\n"
    else:
        explanation += "**Assessment:** High risk supplier. Immediate action required.\n"
    return explanation

# -------------------------
# MAIN APP
# -------------------------
st.title("Trade & Supplier Intelligence Dashboard")
st.write("Real-time supplier insights with interactive filters and AI-powered analytics.")

data_path = "./data/"


# Load data (cached)
(df_sup_raw, df_prod, df_trade, df_tariff, df_price_raw, df_news,
 df_reg_ins, df_reg_rec, df_sup_ins, df_sup_rec, df_com_ins, df_com_rec) = load_and_parse_data(data_path)

# Process suppliers
df_sup = sanitize_suppliers(df_sup_raw)
df_sup = calculate_scores(df_sup)
df_sup = add_coordinates(df_sup)

df_sup["supplier_id"] = df_sup["supplier_id"].astype(str)
df_prod["supplier_id"] = df_prod["supplier_id"].astype(str)
if "supplier_id" in df_trade.columns:
    df_trade["supplier_id"] = df_trade["supplier_id"].astype(str)

# ========================
# SIDEBAR: FILTERS + REFRESH BUTTON AT THE BOTTOM
# ========================
st.sidebar.header("Filters")

product_categories = ["All"] + sorted(df_prod["product_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Product Category", product_categories, default=["All"])

regions = ["All"] + sorted(df_sup["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=["All"])

import_countries = ["All"] + sorted(df_trade["importer"].dropna().unique().astype(str).tolist())
selected_import = st.sidebar.multiselect("Import Country", import_countries, default=["All"])

export_countries = ["All"] + sorted(df_trade["exporter"].dropna().unique().astype(str).tolist())
selected_export = st.sidebar.multiselect("Export Country", export_countries, default=["All"])

score_range = st.sidebar.slider("Supplier Score Range", 0, 100, (0, 100), step=1)

# Date Range Filter
if "date" in df_trade.columns and not df_trade["date"].isna().all():
    min_date, max_date = df_trade["date"].min().date(), df_trade["date"].max().date()
    default_start = pd.Timestamp('2024-06-01').date()
    default_end = pd.Timestamp('2025-06-30').date()
    if default_start < min_date: default_start = min_date
    if default_end > max_date: default_end = max_date
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

# ========================
# REFRESH BUTTON — BELOW DATE RANGE (BOTTOM OF SIDEBAR)
# ========================
st.sidebar.markdown("---")
refresh = st.sidebar.button("Refresh Data", type="primary", use_container_width=True)

if refresh:
    st.cache_data.clear()  # Clears all cached CSVs and processed data
    st.success("All data reloaded from source files!")
    st.rerun()  # Instantly refresh the app with fresh data

# ========================
# BUILD FILTERS DICTIONARY & APPLY
# ========================
filters = {
    'categories': selected_categories,
    'regions': selected_regions,
    'import_countries': selected_import,
    'export_countries': selected_export,
    'score_range': score_range,
    'date_range': date_range
}

df_sup_f, df_trade_f, df_prod_f = apply_filters(df_sup, df_trade, df_prod, filters)

df_tab = df_trade_f.merge(df_sup_f, on="supplier_id", how="left").merge(
    df_prod_f[["supplier_id", "product_category"]].drop_duplicates(), on="supplier_id", how="left")


# tab1, tab2, tab3, tab4 = st.tabs(["Regional Sourcing", "Supplier Intelligence", "Risk & Compliance", "Data Quality"])

tab1, tab2, tab3= st.tabs(["Regional Sourcing", "Supplier Intelligence", "Risk & Compliance"])

# -------------------------
# TAB 1: REGIONAL SOURCING
# -------------------------
with tab1:
    st.subheader("Regional Sourcing — Product & Country Overview")

    if df_tab.empty:
        st.warning("No data after applying filters. Try relaxing filters.")
    else:
        c1, c2 = st.columns([1,1])

        with c1:
            st.markdown("#### Product Category Distribution")
            cat_agg = df_tab.groupby("product_category").agg(
                count=("product_category", "size"),
                num_suppliers=("supplier_id", lambda x: x.nunique()),
                total_value=("value", "sum"),
                total_volume=("volume", "sum")
            ).reset_index()
            
            top_suppliers = df_tab.loc[df_tab.groupby("product_category")["supplier_score"].idxmax()][["product_category", "supplier_name"]]
            cat_agg = cat_agg.merge(top_suppliers, on="product_category", how="left")
            cat_agg["total_value_fmt"] = cat_agg["total_value"].apply(format_millions)
            cat_agg["total_volume_fmt"] = cat_agg["total_volume"].apply(lambda x: f"{int(x):,} MT")
            
            pie = alt.Chart(cat_agg).mark_arc().encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("product_category:N", legend=alt.Legend(title="Category")),
                tooltip=[
                    alt.Tooltip("product_category:N", title="Category"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("num_suppliers:Q", title="Suppliers"),
                    alt.Tooltip("total_value_fmt:N", title="Total Value"),
                    alt.Tooltip("total_volume_fmt:N", title="Total Volume"),
                    alt.Tooltip("supplier_name:N", title="Top Supplier")
                ]
            ).properties(height=350)
            st.altair_chart(pie, use_container_width=True)

        with c2:
            st.markdown("#### Top 10 Countries by Average Supplier Score")
            country_scores = (
                df_tab.groupby("exporter")
                .agg(avg_supplier_score=("supplier_score", "mean"),
                     suppliers=("supplier_id", lambda s: s.nunique()))
                .reset_index()
                .rename(columns={"exporter": "Country"})
                .sort_values("avg_supplier_score", ascending=False)
                .head(10)
            )
            
            if not country_scores.empty:
                comps = (
                    df_tab.groupby("exporter")
                    .agg(compliance_score=("compliance_score", "mean"),
                         performance_score=("performance_score", "mean"),
                         sustainability_score=("sustainability_score", "mean"))
                    .reset_index()
                    .rename(columns={"exporter": "Country"})
                )
                merged = comps.merge(country_scores[["Country","avg_supplier_score","suppliers"]], on="Country")
                merged = merged.sort_values("avg_supplier_score", ascending=False)
                merged["Country"] = pd.Categorical(merged["Country"], categories=merged["Country"].tolist(), ordered=True)
                
                melted = merged.melt(id_vars=["Country","avg_supplier_score","suppliers"],
                                     value_vars=["compliance_score","performance_score","sustainability_score"],
                                     var_name="Component", value_name="Score")
                melted["Component"] = melted["Component"].map({
                    "compliance_score":"Compliance",
                    "performance_score":"Performance",
                    "sustainability_score":"ESG"
                })
                
                bar = alt.Chart(melted).mark_bar().encode(
                    x=alt.X("Score:Q", title="Score"),
                    y=alt.Y("Country:N", sort=alt.EncodingSortField(field="avg_supplier_score", order="descending"), title="Country"),
                    color=alt.Color("Component:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(title="Score Type")),
                    tooltip=[
                        alt.Tooltip("Country:N"),
                        alt.Tooltip("Component:N", title="Score Type"),
                        alt.Tooltip("Score:Q", format=".2f"),
                        alt.Tooltip("avg_supplier_score:Q", title="Total Score", format=".2f"),
                        alt.Tooltip("suppliers:Q", title="# Suppliers")
                    ]
                ).properties(height=350)
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("No country data available")

        st.markdown("---")
        st.markdown("#### Regional Trade & Tariff Overview")

        col_tariff, col_ts = st.columns(2)

        with col_tariff:
            st.markdown("**Tariff Rates by Exporting Country (Latest)**")
            if not df_tariff.empty:
                tariff_chart = create_tariff_chart(df_tariff)
                if tariff_chart:
                    st.altair_chart(tariff_chart, use_container_width=True)
                else:
                    st.info("No tariff data available (missing dates or tariff percentages).")
            else:
                st.info("No tariff data available")

        with col_ts:
            st.markdown("**Trade Trends Over Time**")
            ts_chart = create_time_series_chart(df_trade_f)
            if ts_chart:
                st.altair_chart(ts_chart, use_container_width=True)
            else:
                st.info("No trade data available for time series.")

        st.markdown("---")


        # Map visualization
        col_map, col_ai = st.columns([2, 1])
        
        df_country_map = (
            df_sup_f.groupby("country")
            .agg(
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
                num_suppliers=("supplier_id", "nunique"),
                avg_supplier_score=("supplier_score", "mean")
            )
            .reset_index()
        )
        
        trade_by_country = (
            df_trade_f.groupby("exporter")
            .agg(total_volume=("volume", "sum"), total_value=("value", "sum"))
            .reset_index()
            .rename(columns={"exporter": "country"})
        )
        
        df_country_map = df_country_map.merge(trade_by_country, on="country", how="left")
        df_country_map["total_volume"] = df_country_map["total_volume"].fillna(0)
        df_country_map["total_value"] = df_country_map["total_value"].fillna(0)
        df_country_map = df_country_map.dropna(subset=["latitude", "longitude"])
        df_country_map = df_country_map.sort_values("avg_supplier_score", ascending=False)

        with col_map:
            st.markdown("#### Supplier Map (bubbles by country)")
            
            if df_country_map.empty:
                st.info("No country locations available to plot.")
            else:
                norm = mcolors.Normalize(vmin=df_country_map["avg_supplier_score"].min(), 
                                        vmax=df_country_map["avg_supplier_score"].max())
                cmap = cm.get_cmap("RdYlGn")
                df_country_map["color"] = df_country_map["avg_supplier_score"].apply(
                    lambda x: [int(c*255) for c in cmap(norm(x))[:3]]
                )

                min_radius, max_radius = 50000, 500000
                min_sup = df_country_map["num_suppliers"].min()
                max_sup = df_country_map["num_suppliers"].max()
                
                if max_sup > min_sup:
                    df_country_map["radius"] = ((df_country_map["num_suppliers"] - min_sup) / 
                                               (max_sup - min_sup) * (max_radius - min_radius) + min_radius)
                else:
                    df_country_map["radius"] = max_radius

                df_country_map["total_volume_fmt"] = df_country_map["total_volume"].apply(lambda x: f"{int(x):,} MT")
                df_country_map["total_value_fmt"] = df_country_map["total_value"].apply(format_millions)

                scatter = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_country_map,
                    get_position='[longitude, latitude]',
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True,
                    radius_min_pixels=5,
                    radius_max_pixels=100
                )
                
                vs = pdk.ViewState(
                    latitude=float(df_country_map["latitude"].mean()), 
                    longitude=float(df_country_map["longitude"].mean()), 
                    zoom=2 if len(df_country_map) > 3 else 4
                )

                deck = pdk.Deck(
                    layers=[scatter],
                    initial_view_state=vs,
                    tooltip={"text": "Country: {country}\nSuppliers: {num_suppliers}\nAvg Score: {avg_supplier_score}\nTotal Volume: {total_volume_fmt}\nTotal Value: {total_value_fmt}"}
                )
                st.pydeck_chart(deck, use_container_width=True)

        with col_ai:
            st.markdown("### AI Insights")

            # Parse news dates once
            if "news_date_dt" not in df_news.columns:
                df_news["news_date_dt"] = pd.to_datetime(
                    df_news["news_date"], format="%Y.%m.%d", errors="coerce"
                )

            insight_path = os.path.join(data_path, "regional_insight.csv")
            recommend_path = os.path.join(data_path, "regional_recommend.csv")

            # Generate button
            if st.button("Generate Insights & Recommendations", type="primary", use_container_width=True):
                with st.spinner("Generating AI insights from latest news..."):
                    # Start with full news
                    news_f = df_news.copy()

                    # Apply current dashboard filters via df_tab (countries in scope)
                    countries_in_scope = set()
                    if "exporter" in df_tab.columns:
                        countries_in_scope.update(df_tab["exporter"].dropna().unique())
                    if "importer" in df_tab.columns:
                        countries_in_scope.update(df_tab["importer"].dropna().unique())

                    if countries_in_scope:
                        news_f = news_f[news_f["impacted_country"].isin(countries_in_scope)]

                    # Apply date filter if active
                    if date_range and len(date_range) == 2:
                        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        news_f = news_f[
                            (news_f["news_date_dt"] >= start_dt) &
                            (news_f["news_date_dt"] <= end_dt)
                        ]

                   
                    # Prepare filter labels in a way that matches regional_ai filters
                    # Prepare filter labels in a way that matches regional_ai filters
                    cat_values = [c for c in selected_categories if c != "All"]
                    cat_label = ", ".join(cat_values) if cat_values else "All"

                    reg_values = [r for r in selected_regions if r != "All"]
                    if not reg_values:
                        reg_label = "All"
                    elif len(reg_values) == 1:
                        reg_label = reg_values[0]
                    else:
                        reg_label = "Multiple"

                    imp_values = [i for i in selected_import if i != "All"]
                    if not imp_values:
                        imp_label = "All"
                    elif len(imp_values) == 1:
                        imp_label = imp_values[0]
                    else:
                        imp_label = "Multiple"

                    exp_values = [e for e in selected_export if e != "All"]
                    if not exp_values:
                        exp_label = "All"
                    elif len(exp_values) == 1:
                        exp_label = exp_values[0]
                    else:
                        exp_label = "Multiple"



                    # If filters result in no news, fall back to all news
                    if news_f.empty:
                        news_f = df_news.copy()

                    try:
                        insights_df = regionalInsights(
                            news_df=news_f,
                            category=cat_label,
                            region=reg_label,
                            import_country=imp_label,
                            export_country=exp_label,
                            save_path=insight_path,
                        )

                        recs_df = regionalRecommend(
                            news_df=news_f,
                            category=cat_label,
                            region=reg_label,
                            import_country=imp_label,
                            export_country=exp_label,
                            insights_path=insight_path,
                            save_path=recommend_path,
                        )

                        # Save to session state for display
                        st.session_state["insights_latest"] = insights_df
                        st.session_state["recs_latest"] = recs_df

                        st.success("AI insights & recommendations generated successfully!")
                    except Exception as e:
                        st.error(f"Error during AI generation: {e}")

            # === DISPLAY INSIGHTS ===
            st.markdown("#### Latest Insights")
            insights_latest = st.session_state.get("insights_latest")

            if insights_latest is None or insights_latest.empty:
                st.info("Click the button above to generate AI-powered insights.")
            else:
                st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

                for idx, row in insights_latest.iterrows():
                    desc = str(row.get("insight_desc") or "").strip()
                    if desc and desc not in ["nan", ""]:
                        st.markdown(f"**{idx + 1}.** {desc}")

            st.markdown("---")

            # === DISPLAY RECOMMENDATIONS ===
            st.markdown("#### Recommended Actions")

            recs_latest = st.session_state.get("recs_latest")

            if recs_latest is None or recs_latest.empty:
                st.info("No recommendations yet. Generate insights first.")
            else:
                # Find the text column
                text_col = next(
                    (col for col in ["recommend_desc", "recommendation_text", "recommend_text", "text"] if col in recs_latest.columns),
                    None
                )

                if not text_col:
                    st.warning("No recommendation text found.")
                else:
                    for idx, row in recs_latest.iterrows():
                        desc = str(row[text_col]).strip()
                        if desc and desc not in ["", "nan", "N/A"]:
                            st.markdown(f"**{idx + 1}.** {desc}")
        
        st.markdown("---")
        
        # NEW: Top 5 Most Recent News
        st.markdown("#### Recent Supply Chain News")
        
        if not df_news.empty and 'news_date' in df_news.columns:
            # Sort by date and get top 5
            df_news_sorted = df_news.sort_values('news_date', ascending=False)
            top_news = df_news_sorted.head(5)
            
            for idx, row in top_news.iterrows():
                with st.expander(f"📰 {row['news_title']} ({row['news_date'].strftime('%Y-%m-%d') if pd.notna(row['news_date']) else 'N/A'})"):
                    st.write(f"**Country:** {row.get('impacted_country', 'N/A')}")
                    st.write(f"**Category:** {row.get('news_category', 'N/A')}")
                    st.write(f"**Summary:** {row.get('news_summary', 'N/A')}")
        else:
            st.info("No news data available")



# -------------------------
# TAB 2: SUPPLIER INTELLIGENCE
# -------------------------
with tab2:
    st.subheader("Supplier Intelligence")
    k1, k2, k3 = st.columns(3)
    total_suppliers = df_sup_f["supplier_id"].nunique()
    avg_score_all = df_sup_f["supplier_score"].mean() if total_suppliers>0 else np.nan
    k1.metric("Suppliers in view", f"{total_suppliers}")
    k2.metric("Avg supplier score", f"{avg_score_all:.2f}" if not np.isnan(avg_score_all) else "N/A")
    k3.metric("Countries in view", df_tab["exporter"].nunique())

    # st.markdown("### Trade Trends Over Time")
    # ts_chart = create_time_series_chart(df_trade_f)
    # if ts_chart:
    #     st.altair_chart(ts_chart, use_container_width=True)
    # else:
    #     st.info("No trade data available for time series.")

    st.markdown("### Supplier Performance Benchmarks")
    if len(df_sup_f) >= 10:
        benchmark_chart = create_benchmark_chart(df_sup_f, df_prod_f)
        if benchmark_chart:
            st.altair_chart(benchmark_chart, use_container_width=True)
        else:
            st.info("Not enough data for benchmark.")
    else:
        st.info("Need at least 10 suppliers.")

    st.markdown("---")
    
    # Supplier Comparison Scatterplot
    st.markdown("### Supplier Comparison: Value vs Volume")
    
    if not df_trade_f.empty:
        # Aggregate by supplier
        supplier_agg = df_trade_f.groupby('supplier_id').agg(
            total_value=('value', 'sum'),
            total_volume=('volume', 'sum'),
            avg_unit_price=('unit_price', 'mean'),
            exporter=('exporter', 'first')
        ).reset_index()

        supplier_agg['total_value_m'] = supplier_agg['total_value'] / 1_000_000

        
        # Merge with supplier scores
        supplier_agg = supplier_agg.merge(
            df_sup_f[['supplier_id', 'supplier_name', 'supplier_score', 'country']], 
            on='supplier_id', 
            how='left'
        )
        
        # Country filter for scatterplot
        scatter_countries = sorted(supplier_agg['country'].dropna().unique().tolist())
        selected_scatter_country = st.selectbox("Filter by Country", ["All"] + scatter_countries)
        
        if selected_scatter_country != "All":
            supplier_agg_filtered = supplier_agg[supplier_agg['country'] == selected_scatter_country]
        else:
            supplier_agg_filtered = supplier_agg
        
        if len(supplier_agg_filtered) > 0:
            # Create scatterplot
            scatter = alt.Chart(supplier_agg_filtered).mark_circle().encode(
                x=alt.X('total_volume:Q', title='Total Volume (MT)', scale=alt.Scale(zero=False)),
                y=alt.Y('total_value_m:Q', title='Total Spend ($M)', scale=alt.Scale(zero=False)),
                size=alt.Size('avg_unit_price:Q', title='Avg Unit Price', scale=alt.Scale(range=[50, 1000])),
                color=alt.Color('supplier_score:Q', title='Supplier Score', scale=alt.Scale(scheme='redyellowgreen')),
                tooltip=[
                    alt.Tooltip('supplier_name:N', title='Supplier'),
                    alt.Tooltip('country:N', title='Country'),
                    alt.Tooltip('total_volume:Q', title='Volume (MT)', format=',.0f'),
                    alt.Tooltip('total_value_m:Q', title='Value ($M)', format=',.2f'),
                    alt.Tooltip('avg_unit_price:Q', title='Avg Unit Price', format='$,.2f'),
                    alt.Tooltip('supplier_score:Q', title='Score', format='.2f')
                ]
            ).properties(height=450).interactive()

            
            st.altair_chart(scatter, use_container_width=True)
            st.caption("💡 Bubble size = Unit Price | Color = Supplier Score (red=low, green=high)")
        else:
            st.info("No data available for selected country.")
    else:
        st.info("No trade data available for scatterplot.")
    
    st.markdown("---")
    
    # Supplier drill-down section
    st.markdown("### Supplier Drill-Down")
    
    if len(df_sup_f) > 0:
        col_select, col_nav = st.columns([3, 1])
        
        with col_select:
            supplier_list = df_sup_f.sort_values('supplier_score', ascending=False)['supplier_id'].tolist()
            selected_supplier_id = st.selectbox("Select Supplier", supplier_list, 
                                               format_func=lambda x: f"{x} - {df_sup_f[df_sup_f['supplier_id']==x]['supplier_name'].iloc[0] if 'supplier_name' in df_sup_f.columns else x}")
        
        with col_nav:
            st.write("")  # Spacing
            st.write("")
            current_idx = supplier_list.index(selected_supplier_id)
            
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Prev"):
                    if current_idx > 0:
                        selected_supplier_id = supplier_list[current_idx - 1]
                        st.rerun()
            with col_next:
                if st.button("Next ➡️"):
                    if current_idx < len(supplier_list) - 1:
                        selected_supplier_id = supplier_list[current_idx + 1]
                        st.rerun()
        
        if selected_supplier_id:
            supplier_detail = df_sup_f[df_sup_f['supplier_id'] == selected_supplier_id].iloc[0]
            
            # Display supplier info
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            col_info1.metric("Supplier Score", f"{supplier_detail['supplier_score']:.2f}")
            col_info2.metric("Risk Rating", supplier_detail['risk_rating'])
            col_info3.metric("Country", supplier_detail['country'])
            col_info4.metric("Region", supplier_detail['region'])
            
            # Radar chart and category mix
            col_radar, col_cat = st.columns([1, 1])
            
            with col_radar:
                radar_fig = create_radar_chart(selected_supplier_id, df_sup_f)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
            
            with col_cat:
                st.markdown("#### Product Categories")
                supplier_cats = df_prod_f[df_prod_f['supplier_id'] == selected_supplier_id]
                
                if not supplier_cats.empty:
                    # UPDATED: Add unit price to tooltip
                    # Get unit prices for each category for this supplier
                    supplier_trades = df_trade_f[df_trade_f['supplier_id'] == selected_supplier_id]
                    
                    if not supplier_trades.empty:
                        # Calculate average unit price per category
                        cat_prices = supplier_trades.groupby('product')['unit_price'].mean().to_dict()
                        
                        # Create category counts
                        cat_counts = supplier_cats['product_category'].value_counts().reset_index()
                        cat_counts.columns = ['product_category', 'count']
                        
                        # Map unit prices
                        cat_counts['avg_unit_price'] = cat_counts['product_category'].map(cat_prices).fillna(0)
                        cat_counts['price_fmt'] = cat_counts['avg_unit_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                        
                        cat_pie = alt.Chart(cat_counts).mark_arc().encode(
                            theta=alt.Theta("count:Q"),
                            color=alt.Color("product_category:N"),
                            tooltip=[
                                alt.Tooltip("product_category:N", title="Category"),
                                alt.Tooltip("count:Q", title="Count"),
                                alt.Tooltip("price_fmt:N", title="Avg Unit Price")
                            ]
                        ).properties(height=400)
                        st.altair_chart(cat_pie, use_container_width=True)
                    else:
                        # Fallback if no trade data
                        cat_pie = alt.Chart(supplier_cats).mark_arc().encode(
                            theta=alt.Theta(field="product_category", aggregate="count"),
                            color=alt.Color("product_category:N"),
                            tooltip=["product_category:N", alt.Tooltip(field="product_category", aggregate="count")]
                        ).properties(height=400)
                        st.altair_chart(cat_pie, use_container_width=True)
                else:
                    st.info("No category data for this supplier.")
            
            # UPDATED: Trade volume AND value over time (dual line)
            st.markdown("#### Trade Volume & Value Over Time")
            supplier_trades = df_trade_f[df_trade_f['supplier_id'] == selected_supplier_id]
            if not supplier_trades.empty and 'date' in supplier_trades.columns:
                trade_ts = supplier_trades.groupby('date').agg({
                    'volume': 'sum',
                    'value': 'sum'
                }).reset_index()

                trade_ts['value_m'] = trade_ts['value'] / 1_000_000

                
                # Create dual axis chart
                base = alt.Chart(trade_ts).encode(x=alt.X('date:T', title='Date'))
                
                volume_line = base.mark_line(color='steelblue', point=True).encode(
                    y=alt.Y('volume:Q', title='Volume (MT)', axis=alt.Axis(titleColor='steelblue')),
                    tooltip=[
                        alt.Tooltip('date:T', title='Date'),
                        alt.Tooltip('volume:Q', title='Volume', format=',.0f')
                    ]
                )
                
                value_line = base.mark_line(color='orange', point=True).encode(
                    y=alt.Y('value_m:Q', title='Value ($M)', axis=alt.Axis(titleColor='orange')),
                    tooltip=[
                        alt.Tooltip('date:T', title='Date'),
                        alt.Tooltip('value_m:Q', title='Value ($M)', format=',.2f')
                    ]
                )

                
                trade_chart = alt.layer(volume_line, value_line).resolve_scale(
                    y='independent'
                ).properties(height=250)
                
                st.altair_chart(trade_chart, use_container_width=True)
                st.caption("💡 Blue = Volume | Orange = Value")
            else:
                st.info("No trade history available for this supplier.")
            
            st.markdown("---")
            
            # AI widgets for this specific supplier
            # col_insight_sup, col_rec_sup = st.columns(2)
            
            # with col_insight_sup:
            #     st.markdown("### AI Insights")
            #     if st.button("🤖 Explain This Supplier"):
            #         explanation = explain_supplier(selected_supplier_id, df_sup_f, df_trade_f)
            #         st.markdown(explanation)
            #     else:
            #         st.info("Click button to generate AI analysis for this supplier.")
            
            # with col_rec_sup:
            #     st.markdown("### AI Recommendations")
                
            #     # Generate specific recommendations for this supplier
            #     recommendations = []
                
            #     if supplier_detail['supplier_score'] < 50:
            #         recommendations.append("🚨 High risk supplier - immediate audit recommended")
            #     elif supplier_detail['supplier_score'] < 75:
            #         recommendations.append("⚠️ Monitor closely - improvement plan suggested")
            #     else:
            #         recommendations.append("✅ Strong performer - consider capacity expansion")
                
            #     if supplier_detail['compliance_score'] < 20:
            #         recommendations.append("📋 Compliance certification urgently needed")
                
            #     if supplier_detail['performance_score'] < 15:
            #         recommendations.append("📦 Performance issues detected - quality review required")
                
            #     if supplier_detail['sustainability_score'] < 15:
            #         recommendations.append("🌱 ESG improvement program needed")
                
            #     # Check trade volatility
            #     if not supplier_trades.empty:
            #         vol_std = supplier_trades['volume'].std()
            #         vol_mean = supplier_trades['volume'].mean()
            #         if vol_mean > 0 and (vol_std / vol_mean) > 0.5:
            #             recommendations.append("📊 High volume volatility - stabilize order patterns")
                
            #     if recommendations:
            #         rec_text = "\n\n".join(recommendations)
            #         st.warning(rec_text)
            #     else:
            #         st.success("No immediate actions required!")
    else:
        st.info("No suppliers available with current filters.")
    
    st.markdown("---")
    
    # # Overall AI widgets for all suppliers in view
    # col_insight_all, col_rec_all = st.columns(2)
    
    # with col_insight_all:
    #     st.markdown("### AI Insights (All Suppliers)")
        
    #     if not df_sup_f.empty:
    #         avg_score = df_sup_f['supplier_score'].mean()
    #         top_country = df_sup_f.groupby('country')['supplier_score'].mean().idxmax()
    #         top_score = df_sup_f.groupby('country')['supplier_score'].mean().max()
            
    #         insights_text = f"""
    #         **Performance Overview:**
            
    #         📊 Average score: **{avg_score:.2f}**
            
    #         🏆 Top country: **{top_country}** ({top_score:.2f})
            
    #         📈 Total suppliers: **{len(df_sup_f)}**
    #         """
    #         st.info(insights_text)
    #     else:
    #         st.info("Adjust filters to see insights.")
    
    # with col_rec_all:
    #     st.markdown("### AI Recommendations (All Suppliers)")
        
    #     if not df_sup_f.empty:
    #         recommendations = []
            
    #         # Risk distribution
    #         high_risk_pct = (df_sup_f['risk_rating'] == 'High').sum() / len(df_sup_f) * 100
    #         if high_risk_pct > 20:
    #             recommendations.append(f"⚠️ {high_risk_pct:.0f}% high-risk suppliers - review portfolio")
            
    #         # Geographic concentration
    #         top_country_count = df_sup_f['country'].value_counts().iloc[0]
    #         concentration = (top_country_count / len(df_sup_f) * 100)
    #         if concentration > 40:
    #             top_conc_country = df_sup_f['country'].value_counts().index[0]
    #             recommendations.append(f"🌍 {concentration:.0f}% concentrated in {top_conc_country} - diversify")
            
    #         # Low performers
    #         low_performers = (df_sup_f['supplier_score'] < 50).sum()
    #         if low_performers > 0:
    #             recommendations.append(f"🔍 {low_performers} suppliers below 50 score - capability assessment needed")
            
    #         # Performance trends
    #         if len(df_sup_f) >= 10:
    #             recommendations.append("✅ Sufficient supplier base for benchmarking")
    #         else:
    #             recommendations.append("📈 Expand supplier base for better risk distribution")
            
    #         if recommendations:
    #             rec_text = "\n\n".join(recommendations)
    #             st.warning(rec_text)
    #         else:
    #             st.success("Portfolio is well balanced!")
    #     else:
    #         st.info("Adjust filters to see recommendations.")

# -------------------------
# TAB 3: RISK & COMPLIANCE
# -------------------------
with tab3:
    st.subheader("Risk & Compliance")
    
    # Certification metrics
    cert_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
    cols = st.columns(5)
    
    for idx, cert in enumerate(cert_cols):
        with cols[idx]:
            if cert in df_sup_f.columns:
                cert_data = df_sup_f[cert].fillna('N')
                if cert_data.dtype == 'object':
                    cert_count = (cert_data.str.strip().str.upper() == 'Y').sum()
                else:
                    cert_count = (pd.to_numeric(cert_data, errors='coerce') > 0).sum()
                
                total = len(df_sup_f)
                pct = (cert_count / total * 100) if total > 0 else 0
                label = cert.replace('cert_', '').replace('iso_', 'ISO ').upper()
                st.metric(label, f"{pct:.1f}%", f"{cert_count}/{total}")
            else:
                label = cert.replace('cert_', '').replace('iso_', 'ISO ').upper()
                st.metric(label, "N/A", "No data")
    
    st.markdown("---")
    
    # Missing documents with SEARCH functionality
    st.markdown("### Suppliers Missing Key Documents")
    
    # NEW: Add search box
    search_term = st.text_input("🔍 Search by Supplier Name or ID", "")
    
    doc_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
    missing_docs = df_sup_f[['supplier_id', 'supplier_name', 'country'] + 
                            [col for col in doc_cols if col in df_sup_f.columns]].copy()
    
    if 'supplier_score' in df_sup_f.columns:
        missing_docs['supplier_score'] = df_sup_f['supplier_score']
    
    has_missing = False
    for col in doc_cols:
        if col in missing_docs.columns:
            cert_data = missing_docs[col].fillna('N')
            if cert_data.dtype == 'object':
                missing_docs[f'{col}_missing'] = ~(cert_data.str.upper() == 'Y')
            else:
                missing_docs[f'{col}_missing'] = pd.to_numeric(cert_data, errors='coerce').fillna(0) == 0
            
            if missing_docs[f'{col}_missing'].any():
                has_missing = True
    
    missing_cols = [col for col in missing_docs.columns if col.endswith('_missing')]
    if has_missing:
        suppliers_with_missing = missing_docs[missing_docs[missing_cols].any(axis=1)]
        
        # NEW: Apply search filter
        if search_term:
            suppliers_with_missing = suppliers_with_missing[
                suppliers_with_missing['supplier_name'].str.contains(search_term, case=False, na=False) |
                suppliers_with_missing['supplier_id'].str.contains(search_term, case=False, na=False)
            ]
        
        display_cols = ['supplier_id', 'supplier_name', 'country'] + \
                      [col for col in doc_cols if col in df_sup_f.columns]
        if 'supplier_score' in suppliers_with_missing.columns:
            display_cols.append('supplier_score')
        
        display_df = suppliers_with_missing[display_cols].copy()
        
        for col in doc_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].fillna('N').astype(str).apply(
                    lambda x: '✓' if x.strip().upper() == 'Y' else '✗'
                )
        
        rename_map = {
            'supplier_id': 'Supplier ID', 'supplier_name': 'Supplier Name', 'country': 'Country',
            'cert_gots': 'GOTS', 'cert_oeko': 'OEKO-TEX', 'cert_grs': 'GRS',
            'iso_14001': 'ISO 14001', 'iso_45001': 'ISO 45001', 'supplier_score': 'Supplier Score'
        }
        display_df = display_df.rename(columns=rename_map)
        
        st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
        st.info(f"Showing {len(display_df)} suppliers with missing documents" + (f" (filtered by '{search_term}')" if search_term else ""))
    else:
        st.success("All suppliers have submitted required documents!")
    
    st.markdown("---")
    
    # Risk distribution and AI widgets side by side
    col_risk, col_ai_insight, col_ai_rec = st.columns([1, 1, 1])
    
    with col_risk:
        st.markdown("### Risk Distribution")
        if not df_sup_f.empty:
            risk_counts = df_sup_f["risk_rating"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            pie = alt.Chart(risk_counts).mark_arc().encode(
                theta="Count:Q",
                color=alt.Color("Risk:N", sort=["Low","Medium","High"]),
                tooltip=["Risk:N", "Count:Q"]
            ).properties(height=350)
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No data available.")
    
    with col_ai_insight:
        st.markdown("### AI Insights")
        
        if not df_sup_f.empty:
            total = len(df_sup_f)
            risk_counts = df_sup_f["risk_rating"].value_counts()
            high_risk = risk_counts.get("High", 0)
            low_risk = risk_counts.get("Low", 0)
            medium_risk = risk_counts.get("Medium", 0)
            
            high_risk_pct = (high_risk / total * 100) if total > 0 else 0
            low_risk_pct = (low_risk / total * 100) if total > 0 else 0
            
            # Certification insights
            cert_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
            total_certs = 0
            for cert in cert_cols:
                if cert in df_sup_f.columns:
                    cert_data = df_sup_f[cert].fillna('N')
                    if cert_data.dtype == 'object':
                        total_certs += (cert_data.str.strip().str.upper() == 'Y').sum()
                    else:
                        total_certs += (pd.to_numeric(cert_data, errors='coerce') > 0).sum()
            
            avg_certs = total_certs / total if total > 0 else 0
            
            insights_text = f"""
            **Risk Overview:**
            
            ✅ Low risk: **{low_risk_pct:.1f}%** ({low_risk} suppliers)
            
            ⚠️ Medium risk: **{(medium_risk/total*100):.1f}%** ({medium_risk} suppliers)
            
            🚨 High risk: **{high_risk_pct:.1f}%** ({high_risk} suppliers)
            
            📋 Avg certifications: **{avg_certs:.1f}** per supplier
            """
            st.info(insights_text)
        else:
            st.info("No data available.")
    
    with col_ai_rec:
        st.markdown("### AI Recommendations")
        
        if not df_sup_f.empty:
            recommendations = []
            
            # Risk-based recommendations
            risk_counts = df_sup_f["risk_rating"].value_counts()
            high_risk = risk_counts.get("High", 0)
            high_risk_pct = (high_risk / len(df_sup_f) * 100) if len(df_sup_f) > 0 else 0
            
            if high_risk_pct > 20:
                recommendations.append(f"🚨 {high_risk_pct:.0f}% high risk - prioritize audits")
            
            # Certification recommendations
            cert_cols_check = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
            low_cert = []
            
            for cert in cert_cols_check:
                if cert in df_sup_f.columns:
                    cert_data = df_sup_f[cert].fillna('N')
                    if cert_data.dtype == 'object':
                        cert_count = (cert_data.str.strip().str.upper() == 'Y').sum()
                    else:
                        cert_count = (pd.to_numeric(cert_data, errors='coerce') > 0).sum()
                    
                    pct = (cert_count / len(df_sup_f) * 100) if len(df_sup_f) > 0 else 0
                    
                    if pct < 30:
                        label = cert.replace('cert_', '').replace('iso_', 'ISO ').upper()
                        low_cert.append(label)
            
            if low_cert:
                certs_str = ", ".join(low_cert[:2])
                recommendations.append(f"📜 Low {certs_str} rates - certification drive needed")
            
            # Score-based
            low_scorers = df_sup_f[df_sup_f["supplier_score"] < 50]
            if len(low_scorers) > 0:
                recommendations.append(f"🔍 {len(low_scorers)} suppliers <50 score - capability review")
            
            # Compliance gaps
            low_compliance = df_sup_f[df_sup_f["compliance_score"] < 20]
            if len(low_compliance) > 0:
                recommendations.append(f"⚖️ {len(low_compliance)} suppliers with compliance gaps")
            
            if high_risk_pct < 10 and len(low_cert) == 0:
                recommendations.append("✅ Strong compliance - maintain protocols")
            
            if recommendations:
                rec_text = "\n\n".join(recommendations)
                st.warning(rec_text)
            else:
                st.success("Risk profile is healthy!")
        else:
            st.info("Adjust filters to see recommendations.")

# -------------------------
# TAB 4: DATA QUALITY
# -------------------------
# with tab4:
#     st.subheader("Data Quality & Integrity Checks")
    
#     # Run quality checks
#     quality_report = check_data_quality(df_sup, df_trade)
    
#     col_q1, col_q2 = st.columns(2)
    
#     with col_q1:
#         st.markdown("### Supplier Data Quality")
        
#         missing_sup = quality_report['supplier_missing']
#         if any(v > 0 for v in missing_sup.values()):
#             st.warning("**Missing Values Found:**")
#             for col, count in missing_sup.items():
#                 if count > 0:
#                     st.write(f"- {col}: {count} missing")
#         else:
#             st.success("No missing values in supplier data")
        
#         if quality_report['supplier_duplicates'] > 0:
#             st.error(f"**{quality_report['supplier_duplicates']} duplicate suppliers found**")
#             if st.button("Remove Supplier Duplicates"):
#                 df_sup = fix_duplicates(df_sup, ['supplier_id'])
#                 st.success("Duplicates removed!")
#                 st.rerun()
#         else:
#             st.success("No duplicate suppliers")
    
#     with col_q2:
#         st.markdown("### Trade Data Quality")
        
#         missing_trade = quality_report['trade_missing']
#         if any(v > 0 for v in missing_trade.values()):
#             st.warning("**Missing Values Found:**")
#             for col, count in missing_trade.items():
#                 if count > 0:
#                     st.write(f"- {col}: {count} missing")
#         else:
#             st.success("No missing values in trade data")
        
#         if quality_report['trade_duplicates'] > 0:
#             st.error(f"**{quality_report['trade_duplicates']} duplicate trades found**")
#             if st.button("Remove Trade Duplicates"):
#                 df_trade = fix_duplicates(df_trade, ['report_id'])
#                 st.success("Duplicates removed!")
#                 st.rerun()
#         else:
#             st.success("No duplicate trades")
    
#     st.markdown("---")
    
#     # Outliers and anomalies
#     col_out1, col_out2 = st.columns(2)
    
#     with col_out1:
#         st.markdown("### Price Outliers")
#         if quality_report['price_outliers'] > 0:
#             st.warning(f"**{quality_report['price_outliers']} price outliers detected**")
#             st.info("Outliers are prices >3 standard deviations from mean")
#         else:
#             st.success("No price outliers detected")
    
#     with col_out2:
#         st.markdown("### Data Anomalies")
#         if quality_report['negative_volumes'] > 0:
#             st.error(f"**{quality_report['negative_volumes']} negative volume entries found**")
#         else:
#             st.success("No negative volumes")
    
#     st.markdown("---")
    
#     # Quick fix buttons
#     st.markdown("### Quick Fixes")
#     col_fix1, col_fix2, col_fix3 = st.columns(3)
    
#     with col_fix1:
#         if st.button("Fill Missing with Median"):
#             numeric_cols = ['volume', 'value', 'unit_price']
#             df_trade = fill_missing_with_median(df_trade, numeric_cols)
#             st.success("Missing values filled!")
#             st.rerun()
    
#     with col_fix2:
#         if st.button("Drop Null Coordinates"):
#             df_sup = df_sup.dropna(subset=['latitude', 'longitude'])
#             st.success("Rows with null coordinates removed!")
#             st.rerun()
    
#     with col_fix3:
#         if st.button("Reset All Filters"):
#             st.success("Filters reset! Refresh page to apply.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("Enhanced Supplier Intelligence Dashboard v2.0 | Powered by AI Analytics")