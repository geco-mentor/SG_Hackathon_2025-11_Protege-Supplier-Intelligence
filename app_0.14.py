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
from api.compliance_ai import complianceInsights, complianceRecommend
from api.supplier_ai import supplierInsights, supplierRecommend


# -------------------------
# INITIATE SESSION STATE FOR AI GENERATION 
# -------------------------

# Session state for latest AI outputs
if "insights_latest" not in st.session_state:
    st.session_state["insights_latest"] = None
if "recs_latest" not in st.session_state:
    st.session_state["recs_latest"] = None

# Separate session state for Supplier AI (Tab 2)
if "supplier_insights_latest" not in st.session_state:
    st.session_state["supplier_insights_latest"] = None
if "supplier_recs_latest" not in st.session_state:
    st.session_state["supplier_recs_latest"] = None

# Session state for compliance-level AI outputs (Tab 3)
if "comp_insights_latest" not in st.session_state:
    st.session_state["comp_insights_latest"] = None
if "comp_recs_latest" not in st.session_state:
    st.session_state["comp_recs_latest"] = None

# === ASSUMING YOU HAVE THESE TWO FUNCTIONS DEFINED SOMEWHERE ===
# If not, here are minimal stubs you can replace with your real GenAI calls

# -------------------------
# PAGE CONFIG & HELPER
# -------------------------
st.set_page_config(page_title="PROTΞGΞ SourceVantage", layout="wide")

# Custom CSS to increase font size by 30%
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-size: 130% !important;
    }
    .stMarkdown, .stText {
        font-size: 130% !important;
    }
    .stMetric {
        font-size: 130% !important;
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 130% !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 130% !important;
    }
    .stSelectbox, .stMultiSelect, .stSlider, .stButton {
        font-size: 130% !important;
    }
    /* Increase sidebar filter label font size by 20% */
    .stSidebar label {
        font-size: 120% !important;
    }
    /* Increase tab names font size by 20% */
    button[data-baseweb="tab"] {
        font-size: 120% !important;
    }
    button {
        font-size: 130% !important;
    }
    .stDataFrame {
        font-size: 130% !important;
    }
    .stExpander {
        font-size: 130% !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-size: 130% !important;
    }
    /* Tooltip font size - set to 14px for all tooltips */
    #vg-tooltip-element {
        font-size: 14px !important;
    }
    #vg-tooltip-element table tr td {
        font-size: 14px !important;
    }
    .vega-tooltip {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

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

    # --- helper so AI output CSVs are optional ---
    def safe_read_csv(path: str, sep: str = ";") -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_csv(path, sep=sep)
        # if file doesn't exist yet, return empty DataFrame
        return pd.DataFrame()


    df_sup_raw  = pd.read_csv(data_path + "supplier_master.csv", sep=";")
    df_prod_raw = pd.read_csv(data_path + "supplier_product.csv", sep=";")
    df_trade_raw = pd.read_csv(data_path + "trade_flow.csv", sep=";")
    df_tariff_raw = pd.read_csv(data_path + "trade_tariff.csv", sep=";")
    df_price_raw = pd.read_csv(data_path + "supplier_price.csv", sep=";")
    df_news_raw  = pd.read_csv(data_path + "news_master.csv", sep=";")



    # AI output files – optional on first run
    df_reg_ins_raw = safe_read_csv(os.path.join(data_path, "regional_insight.csv"))
    df_reg_rec_raw = safe_read_csv(os.path.join(data_path, "regional_recommend.csv"))
    df_sup_ins_raw = safe_read_csv(os.path.join(data_path, "supplier_insight.csv"))
    df_sup_rec_raw = safe_read_csv(os.path.join(data_path, "supplier_recommend.csv"))
    df_com_ins_raw = safe_read_csv(os.path.join(data_path, "compliance_insight.csv"))
    df_com_rec_raw = safe_read_csv(os.path.join(data_path, "compliance_recommend.csv"))

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

    # Map supplier_id if missing - IMPROVED VERSION
    if 'supplier_id' not in df_trade.columns and 'country' in df_sup_raw.columns:
        # Create a mapping of country to list of supplier IDs
        country_to_sups = df_sup_raw.groupby('country')['supplier_id'].apply(list).to_dict()
        
        # For each trade record, assign to a supplier from that country
        # We'll use a round-robin approach based on report_id
        def assign_supplier(row):
            country = row['exporter']
            if country in country_to_sups and len(country_to_sups[country]) > 0:
                sups = country_to_sups[country]
                # Use report_id modulo to distribute trades evenly
                if 'report_id' in row:
                    try:
                        idx = int(row['report_id']) % len(sups)
                        return sups[idx]
                    except:
                        return sups[0]
                return sups[0]
            return None
        
        df_trade['supplier_id'] = df_trade.apply(assign_supplier, axis=1)

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

    # Apply supplier filter first if specific suppliers are selected
    if 'selected_suppliers' in filters and filters['selected_suppliers'] and 'All' not in filters['selected_suppliers']:
        df_sup_f = df_sup_f[df_sup_f['supplier_id'].isin(filters['selected_suppliers'])]
        df_trade_f = df_trade_f[df_trade_f['supplier_id'].isin(filters['selected_suppliers'])]
        df_prod_f = df_prod_f[df_prod_f['supplier_id'].isin(filters['selected_suppliers'])]

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

    #  NEW: make sure supplier_score is numeric before comparing
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
    
    # Calculate floor for value axis: lowest value - 20%
    min_value = df_plot['value_m'].min()
    value_floor = min_value * 0.8  # lowest - 20%

    base = alt.Chart(df_plot).encode(x=alt.X('date:T', title='Date', axis=alt.Axis(labelFontSize=16, titleFontSize=18)))
    volume = base.mark_line(color='steelblue', strokeWidth=3).encode(
        y=alt.Y('volume:Q', title='Volume (MT)', axis=alt.Axis(titleColor='steelblue', labelFontSize=16, titleFontSize=18))
    )
    value = base.mark_line(color='orange', strokeWidth=3).encode(
        y=alt.Y('value_m:Q', title='Value (USD Million)', 
                scale=alt.Scale(domain=[value_floor, df_plot['value_m'].max()]),
                axis=alt.Axis(titleColor='orange', labelFontSize=16, titleFontSize=18))
    )
    return alt.layer(volume, value).resolve_scale(y='independent').properties(height=400).configure_axis(
        labelFontSize=16,
        titleFontSize=18
    )


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
    
    # Create readable labels for the legend
    metric_labels = {
        'supplier_score': 'Supplier Score',
        'category_avg': 'Category Average',
        'region_avg': 'Region Average'
    }
    df_melted['Metric_Label'] = df_melted['Metric'].map(metric_labels)
    
    return alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Score:Q', title='Score', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
        y=alt.Y('supplier_name:N', sort='-x', title='Supplier Name', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
        color=alt.Color('Metric_Label:N', 
                       title='Metric',
                       scale=alt.Scale(domain=['Supplier Score', 'Category Average', 'Region Average'],
                                      range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                       legend=alt.Legend(labelFontSize=16, titleFontSize=18)),
        tooltip=['supplier_name', 'Metric_Label', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(height=450).configure_axis(
        labelFontSize=16,
        titleFontSize=18
    )

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
    Create a heatmap matrix showing tariff rates for different importing-exporting country combinations.
    Shows the most recent tariff rate within the filtered date range for each country pair.
    """
    if df_tariff.empty:
        return None

    required_cols = {"start_date", "importing_country", "exporting_country", "tariff_pct"}
    if not required_cols.issubset(df_tariff.columns):
        st.warning("Tariff data is missing required columns.")
        return None

    df = df_tariff.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "tariff_pct", "importing_country", "exporting_country"])

    if df.empty:
        return None

    # For each importing-exporting pair, get the most recent tariff rate within the date range
    df_latest = (
        df.sort_values("start_date", ascending=True)
          .groupby(["importing_country", "exporting_country"], as_index=False)
          .last()  # Get the most recent entry for each pair
    )
    
    if df_latest.empty:
        return None
    
    # Get date range for title
    min_date = df['start_date'].min().strftime('%Y-%m-%d')
    max_date = df['start_date'].max().strftime('%Y-%m-%d')
    
    # Count unique country pairs
    num_pairs = len(df_latest)
    
    if min_date == max_date:
        title_text = f"Tariff Matrix (as of {max_date}) - {num_pairs} trade routes"
    else:
        title_text = f"Tariff Matrix ({min_date} to {max_date}) - {num_pairs} trade routes"
    
    # Create heatmap matrix with larger fonts and no label truncation
    chart = (
        alt.Chart(df_latest)
        .mark_rect()
        .encode(
            x=alt.X("exporting_country:N", 
                   title="Exporting From", 
                   axis=alt.Axis(labelAngle=-90, labelFontSize=14, titleFontSize=16, labelLimit=0)),
            y=alt.Y("importing_country:N", 
                   title="Importing To",
                   axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelLimit=0)),
            color=alt.Color(
                "tariff_pct:Q",
                title="Tariff (%)",
                scale=alt.Scale(scheme="reds"),
                legend=alt.Legend(titleFontSize=14, labelFontSize=12)
            ),
            tooltip=[
                alt.Tooltip("importing_country:N", title="Importing To"),
                alt.Tooltip("exporting_country:N", title="Exporting From"),
                alt.Tooltip("tariff_pct:Q", title="Tariff (%)", format=".2f"),
                alt.Tooltip("start_date:T", title="Effective From", format="%Y-%m-%d"),
            ],
        )
        .properties(
            width=600,
            height=500,
            title=alt.TitleParams(title_text, fontSize=18),
        )
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16
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
st.sidebar.header("SourceVantage")

product_categories = ["All"] + sorted(df_prod["product_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect("Product Category", product_categories, default=["All"])

regions = ["All"] + sorted(df_sup["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=["All"])

import_countries = ["All"] + sorted(df_trade["importer"].dropna().unique().astype(str).tolist())
selected_import = st.sidebar.multiselect("Import Country", import_countries, default=["All"])

export_countries = ["All"] + sorted(df_trade["exporter"].dropna().unique().astype(str).tolist())
selected_export = st.sidebar.multiselect("Export Country", export_countries, default=["All"])

# Supplier Search - Multiselect (sorted numerically)
all_supplier_ids = df_sup["supplier_id"].dropna().unique().tolist()
# Convert to int for sorting, then back to string
all_supplier_ids = sorted(all_supplier_ids, key=lambda x: int(x) if str(x).isdigit() else 999999)
supplier_names = {sid: df_sup[df_sup['supplier_id']==sid]['supplier_name'].iloc[0] 
                  if 'supplier_name' in df_sup.columns and not df_sup[df_sup['supplier_id']==sid].empty 
                  else sid 
                  for sid in all_supplier_ids}
selected_suppliers = st.sidebar.multiselect(
    "Search Supplier",
    ["All"] + all_supplier_ids,
    default=["All"],
    format_func=lambda x: "All Suppliers" if x == "All" else f"{x} - {supplier_names.get(x, x)}"
)

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
    'date_range': date_range,
    'selected_suppliers': selected_suppliers
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
        # MOVED TO TOP: Map and AI Widgets
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
        
        # Calculate trade volumes from df_tab which has the merged data
        trade_by_country = (
            df_tab.groupby("country")
            .agg(total_volume=("volume", "sum"), total_value=("value", "sum"))
            .reset_index()
        )
        
        df_country_map = df_country_map.merge(trade_by_country, on="country", how="left")
        df_country_map["total_volume"] = df_country_map["total_volume"].fillna(0)
        df_country_map["total_value"] = df_country_map["total_value"].fillna(0)
        df_country_map = df_country_map.dropna(subset=["latitude", "longitude"])
        df_country_map = df_country_map.sort_values("avg_supplier_score", ascending=False)

        with col_map:
            with st.expander(" **Supplier Score by Country**", expanded=True):
                if df_country_map.empty:
                    st.info("No country locations available to plot.")
                else:
                    # Color coding based on supplier score
                    min_score = df_country_map["avg_supplier_score"].min()
                    max_score = df_country_map["avg_supplier_score"].max()
                    
                    # Normalize scores to 0-1 for color mapping
                    norm = mcolors.Normalize(vmin=min_score, vmax=max_score)
                    cmap = cm.get_cmap("RdYlGn")
                    df_country_map["color"] = df_country_map["avg_supplier_score"].apply(
                        lambda x: [int(c*255) for c in cmap(norm(x))[:3]]
                    )
                    
                    # Bubble size based on number of suppliers
                    min_radius, max_radius = 50000, 500000
                    min_sup = df_country_map["num_suppliers"].min()
                    max_sup = df_country_map["num_suppliers"].max()
                    
                    if max_sup > min_sup:
                        df_country_map["radius"] = ((df_country_map["num_suppliers"] - min_sup) / 
                                                   (max_sup - min_sup) * (max_radius - min_radius) + min_radius)
                    else:
                        df_country_map["radius"] = max_radius

                    df_country_map["total_volume_fmt"] = df_country_map["total_volume"].apply(lambda x: f"{int(x):,} MT")
                    df_country_map["total_value_fmt"] = df_country_map["total_value"].apply(lambda x: f"${int(x):,}")

                    # Create bubble scatter layer
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
                        tooltip={
                            "html": "<b>Country:</b> {country}<br/><b>Suppliers:</b> {num_suppliers}<br/><b>Total Volume:</b> {total_volume_fmt}<br/><b>Total Value:</b> {total_value_fmt}",
                            "style": {
                                "backgroundColor": "steelblue",
                                "color": "white",
                                "fontSize": "14px"
                            }
                        }
                    )
                    st.pydeck_chart(deck, use_container_width=True)
                    
                    # Add color legend
                    st.caption("Red (Low Score) → Green (High Score) | Bubble size = Number of suppliers")


        with col_ai:
            with st.expander(" **Insights & Recommendations**", expanded=True):
                # Parse news dates once
                if "news_date_dt" not in df_news.columns:
                    df_news["news_date_dt"] = pd.to_datetime(
                        df_news["news_date"], format="%Y.%m.%d", errors="coerce"
                    )

                insight_path = os.path.join(data_path, "regional_insight.csv")
                recommend_path = os.path.join(data_path, "regional_recommend.csv")

                # Generate button
                if st.button("Generate Insights & Recommendations", type="primary", use_container_width=True, key="tab1_gen_ai"):
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
                st.markdown("**Latest Insights**")
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
                st.markdown("**Recommended Actions**")

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
        
        # Product and Country Charts (Collapsible)
        with st.expander(" **Product Category & Country Analysis**", expanded=False):
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
                    color=alt.Color("product_category:N", legend=alt.Legend(
                        title="Category",
                        orient="right",
                        offset=-10,
                        titleFontSize=16,
                        labelFontSize=14
                    )),
                    tooltip=[
                        alt.Tooltip("product_category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Count"),
                        alt.Tooltip("num_suppliers:Q", title="Suppliers"),
                        alt.Tooltip("total_value_fmt:N", title="Total Value"),
                        alt.Tooltip("total_volume_fmt:N", title="Total Volume"),
                        alt.Tooltip("supplier_name:N", title="Top Supplier")
                    ]
                ).properties(height=400)
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
                        x=alt.X("Score:Q", title="Score", axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                        y=alt.Y("Country:N", sort=alt.EncodingSortField(field="avg_supplier_score", order="descending"), 
                               title="Country", axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                        color=alt.Color("Component:N", scale=alt.Scale(scheme="tableau10"), 
                                       legend=alt.Legend(title="Score Type", titleFontSize=16, labelFontSize=14)),
                        tooltip=[
                            alt.Tooltip("Country:N"),
                            alt.Tooltip("Component:N", title="Score Type"),
                            alt.Tooltip("Score:Q", format=".2f"),
                            alt.Tooltip("avg_supplier_score:Q", title="Total Score", format=".2f"),
                            alt.Tooltip("suppliers:Q", title="# Suppliers")
                        ]
                    ).properties(height=400).configure_axis(
                        labelFontSize=16,
                        titleFontSize=18
                    )
                    st.altair_chart(bar, use_container_width=True)
                else:
                    st.info("No country data available")

         # Trade & Tariff Charts (Collapsible)
        with st.expander(" **Trade Overview**", expanded=False):
            col_tariff, col_ts = st.columns(2)

            with col_tariff:
                st.markdown("**Tariff Rates by Trade Route**")
                if not df_tariff.empty:
                    # Filter tariff data based on sidebar filters
                    df_tariff_filtered = df_tariff.copy()
                    
                    # Ensure start_date is datetime
                    df_tariff_filtered['start_date'] = pd.to_datetime(df_tariff_filtered['start_date'], errors='coerce')
                    
                    # Filter by date range - keep tariffs that were active during the selected period
                    if date_range and 'start_date' in df_tariff_filtered.columns:
                        start, end = date_range
                        df_tariff_filtered = df_tariff_filtered[
                            df_tariff_filtered['start_date'] <= pd.Timestamp(end)
                        ]
                    
                    # Filter by regions (convert regions to countries)
                    if selected_regions and 'All' not in selected_regions:
                        region_countries = df_sup[df_sup['region'].isin(selected_regions)]['country'].unique().tolist()
                        if region_countries:
                            df_tariff_filtered = df_tariff_filtered[
                                df_tariff_filtered['exporting_country'].isin(region_countries) |
                                df_tariff_filtered['importing_country'].isin(region_countries)
                            ]
                    
                    # Filter by import countries
                    if selected_import and 'All' not in selected_import:
                        df_tariff_filtered = df_tariff_filtered[
                            df_tariff_filtered['importing_country'].isin(selected_import)
                        ]
                    
                    # Filter by export countries
                    if selected_export and 'All' not in selected_export:
                        df_tariff_filtered = df_tariff_filtered[
                            df_tariff_filtered['exporting_country'].isin(selected_export)
                        ]
                    
                    # Filter by product categories (if that column exists in tariff data)
                    if 'product_category' in df_tariff_filtered.columns:
                        if selected_categories and 'All' not in selected_categories:
                            df_tariff_filtered = df_tariff_filtered[
                                df_tariff_filtered['product_category'].isin(selected_categories)
                            ]
                    
                    # Filter by selected suppliers (show tariffs for their countries)
                    if selected_suppliers and 'All' not in selected_suppliers:
                        supplier_countries = df_sup_f['country'].unique().tolist()
                        if supplier_countries:
                            df_tariff_filtered = df_tariff_filtered[
                                df_tariff_filtered['exporting_country'].isin(supplier_countries) |
                                df_tariff_filtered['importing_country'].isin(supplier_countries)
                            ]
                    
                    if not df_tariff_filtered.empty:
                        tariff_chart = create_tariff_chart(df_tariff_filtered)
                        if tariff_chart:
                            st.altair_chart(tariff_chart, use_container_width=True)
                            
                            # Show summary statistics
                            num_routes = df_tariff_filtered.groupby(['importing_country', 'exporting_country']).size().shape[0]
                            avg_tariff = df_tariff_filtered.groupby(['importing_country', 'exporting_country']).last()['tariff_pct'].mean()
                            max_tariff = df_tariff_filtered.groupby(['importing_country', 'exporting_country']).last()['tariff_pct'].max()
                            
                            st.caption(f"📊 Showing {num_routes} trade routes | Avg tariff: {avg_tariff:.1f}% | Max tariff: {max_tariff:.1f}% | Responds to all sidebar filters")
                        else:
                            st.info("No tariff data available for selected filters.")
                    else:
                        st.info("No tariff data matches the selected filters.")
                else:
                    st.info("No tariff data available")

            with col_ts:
                st.markdown("**Trade Trends Over Time**")
                ts_chart = create_time_series_chart(df_trade_f)
                if ts_chart:
                    st.altair_chart(ts_chart, use_container_width=True)
                else:
                    st.info("No trade data available for time series.")
        
        # Recent News (NOT nested - remove outer expander)
        st.markdown("####  Regional News")
        
        if not df_news.empty and 'news_date' in df_news.columns:
            # Filter news based on selected export countries
            df_news_filtered = df_news.copy()
            if selected_export and 'All' not in selected_export:
                df_news_filtered = df_news_filtered[df_news_filtered['impacted_country'].isin(selected_export)]
            
            # Sort by date and get top 5
            df_news_sorted = df_news_filtered.sort_values('news_date', ascending=False)
            top_news = df_news_sorted.head(5)
            
            if top_news.empty:
                st.info("No news available for selected countries. Showing general news instead.")
                top_news = df_news.sort_values('news_date', ascending=False).head(5)
            
            for idx, row in top_news.iterrows():
                with st.expander(f" {row['news_title']}", expanded=False):
                    # Format date as "d mmm yyyy" (e.g., "21 Nov 2025")
                    if pd.notna(row['news_date']):
                        formatted_date = row['news_date'].strftime('%-d %b %Y') if os.name != 'nt' else row['news_date'].strftime('%#d %b %Y')
                    else:
                        formatted_date = 'N/A'
                    st.write(f"**Date:** {formatted_date}")
                    st.write(f"**Country:** {row.get('impacted_country', 'N/A')}")
                    st.write(f"**Category:** {row.get('news_category', 'N/A')}")
                    st.write(f"**Summary:** {row.get('news_summary', 'N/A')}")
        else:
            st.info("No news data available")
        
        # Supplier Table at bottom of Tab 1

        st.markdown("---")
        st.markdown("#### Supplier Overview")

        if not df_sup_f.empty:
            # --- NEW: restrict to suppliers that actually appear in the filtered trade data ---
            if not df_tab.empty and "supplier_id" in df_tab.columns:
                valid_suppliers = (
                    df_tab["supplier_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                df_sup_overview = df_sup_f[df_sup_f["supplier_id"].isin(valid_suppliers)].copy()
            else:
                # if no trade after filters, fall back to the filtered supplier list
                df_sup_overview = df_sup_f.copy()

            if df_sup_overview.empty:
                st.info("No suppliers to display with current filters.")
            else:
                doc_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
                supplier_table = df_sup_overview[
                    ['supplier_id', 'supplier_name', 'country', 'supplier_score']
                    + [col for col in doc_cols if col in df_sup_overview.columns]
                ].copy()

                # Format supplier score
                if 'supplier_score' in supplier_table.columns:
                    supplier_table['supplier_score'] = supplier_table['supplier_score'].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )

                # Convert certification columns to tick/cross
                for col in doc_cols:
                    if col in supplier_table.columns:
                        cert_data = supplier_table[col].fillna('N').astype(str)
                        supplier_table[col] = cert_data.apply(
                            lambda x: '✓' if x.strip().upper() == 'Y' else '✗'
                        )

                # Rename columns
                rename_map = {
                    'supplier_id': 'Supplier ID',
                    'supplier_name': 'Supplier Name',
                    'country': 'Country',
                    'supplier_score': 'Supplier Score',
                    'cert_gots': 'GOTS',
                    'cert_oeko': 'OEKO-TEX',
                    'cert_grs': 'GRS',
                    'iso_14001': 'ISO 14001',
                    'iso_45001': 'ISO 45001',
                }
                supplier_table = supplier_table.rename(columns=rename_map)

                # Apply color styling to checkmarks
                def style_certs(val):
                    if val == '✓':
                        return 'color: green; font-weight: bold; font-size: 16px'
                    elif val == '✗':
                        return 'color: red; font-weight: bold; font-size: 16px'
                    return ''

                cert_columns = ['GOTS', 'OEKO-TEX', 'GRS', 'ISO 14001', 'ISO 45001']
                existing_cert_cols = [col for col in cert_columns if col in supplier_table.columns]

                if existing_cert_cols:
                    styled_df = supplier_table.style.applymap(style_certs, subset=existing_cert_cols)
                    st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
                else:
                    st.dataframe(supplier_table, use_container_width=True, height=400, hide_index=True)

                st.info(f"Showing {len(supplier_table)} suppliers")
        else:
            st.info("No suppliers to display with current filters.")

# -------------------------
# TAB 2: SUPPLIER INTELLIGENCE
# -------------------------
with tab2:
    st.subheader("Supplier Intelligence")
    k1, k2, k3 = st.columns(3)
    total_suppliers = df_sup_f["supplier_id"].nunique()
    avg_score_all = df_sup_f["supplier_score"].mean() if total_suppliers>0 else np.nan
    k1.metric("Suppliers in view", df_tab["supplier_id"].nunique())
    k2.metric("Avg supplier score", f"{avg_score_all:.2f}" if not np.isnan(avg_score_all) else "N/A")
    k3.metric("Countries in view", df_tab["exporter"].nunique())

    st.markdown("---")

    # TOP ROW: SCATTER PLOT + INSIGHTS & RECOMMENDATIONS (matching Tab 1 layout)
    col_scatter, col_ai = st.columns([2, 1])
    
    with col_scatter:
        # SUPPLIER COMPARISON SCATTERPLOT
        with st.expander(" **Supplier Comparison: Value vs Volume**", expanded=True):
            if not df_sup_f.empty:
                # Start with all filtered suppliers
                supplier_base = df_sup_f[['supplier_id', 'supplier_name', 'supplier_score', 'country']].copy()
                
                # Aggregate trade data by supplier
                if not df_trade_f.empty:
                    supplier_trade = df_trade_f.groupby('supplier_id').agg(
                        total_value=('value', 'sum'),
                        total_volume=('volume', 'sum'),
                        avg_unit_price=('unit_price', 'mean')
                    ).reset_index()
                    
                    # Merge with supplier base (left join to keep all suppliers)
                    supplier_agg = supplier_base.merge(supplier_trade, on='supplier_id', how='left')
                else:
                    supplier_agg = supplier_base.copy()
                    supplier_agg['total_value'] = 0
                    supplier_agg['total_volume'] = 0
                    supplier_agg['avg_unit_price'] = 0
                
                # Fill NaN values for suppliers without trade data
                supplier_agg['total_value'] = supplier_agg['total_value'].fillna(0)
                supplier_agg['total_volume'] = supplier_agg['total_volume'].fillna(0)
                supplier_agg['avg_unit_price'] = supplier_agg['avg_unit_price'].fillna(0)
                supplier_agg['total_value_m'] = supplier_agg['total_value'] / 1_000_000
                
                # Only show suppliers with some trade activity
                supplier_agg_filtered = supplier_agg[supplier_agg['total_value'] > 0]
                
                if len(supplier_agg_filtered) > 0:
                    # Create scatterplot
                    scatter = alt.Chart(supplier_agg_filtered).mark_circle().encode(
                        x=alt.X('total_volume:Q', title='Total Volume (MT)', scale=alt.Scale(zero=False),
                               axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                        y=alt.Y('total_value_m:Q', title='Total Spend ($M)', scale=alt.Scale(zero=False),
                               axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                        size=alt.Size('avg_unit_price:Q', title='Avg Unit Price', scale=alt.Scale(range=[50, 1000])),
                        color=alt.Color('supplier_score:Q', title='Supplier Score', scale=alt.Scale(scheme='redyellowgreen'),
                                       legend=alt.Legend(titleFontSize=16, labelFontSize=14)),
                        tooltip=[
                            alt.Tooltip('supplier_name:N', title='Supplier'),
                            alt.Tooltip('country:N', title='Country'),
                            alt.Tooltip('total_volume:Q', title='Volume (MT)', format=',.0f'),
                            alt.Tooltip('total_value_m:Q', title='Value ($M)', format=',.2f'),
                            alt.Tooltip('avg_unit_price:Q', title='Avg Unit Price', format='$,.2f'),
                            alt.Tooltip('supplier_score:Q', title='Score', format='.2f')
                        ]
                    ).properties(height=450).interactive().configure_axis(
                        labelFontSize=16,
                        titleFontSize=18
                    )

                    
                    st.altair_chart(scatter, use_container_width=True)
                    st.caption(" Bubble size = Unit Price | Color = Supplier Score (red=low, green=high)")
                else:
                    st.info("No data available.")
            else:
                st.info("No trade data available for scatterplot.")
    
    with col_ai:
        # AI INSIGHTS & RECOMMENDATIONS (Supplier AI)
        with st.expander(" **Insights & Recommendations**", expanded=True):

            # ----- Choose ONE supplier in view for detailed AI -----
            sup_ai_id = sup_ai_name = sup_ai_region = sup_ai_country = None
            if not df_sup_f.empty:
                # You can change this selection rule later (e.g. lowest score, filtered, etc.)
                row_ai = df_sup_f.sort_values("supplier_score", ascending=False).iloc[0]
                sup_ai_id = str(row_ai.get("supplier_id", ""))
                sup_ai_name = str(row_ai.get("supplier_name", ""))
                sup_ai_region = str(row_ai.get("region", ""))
                sup_ai_country = str(row_ai.get("country", ""))

            # If no suppliers in view, show message and stop
            if not sup_ai_id:
                st.info("No suppliers in current view for AI. Adjust filters and try again.")
            else:
                # ----- Build filter-label strings (same style as Regional AI) -----
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

                # ----- Filter NEWS for this view (similar to Tab 1) -----
                news_f = df_news.copy()

                # Ensure parsed date column exists
                if "news_date_dt" not in news_f.columns:
                    news_f["news_date_dt"] = pd.to_datetime(
                        news_f["news_date"], errors="coerce"
                    )

                # Countries present in current trade table
                countries_in_scope = set()
                if "exporter" in df_tab.columns:
                    countries_in_scope.update(df_tab["exporter"].dropna().unique())
                if "importer" in df_tab.columns:
                    countries_in_scope.update(df_tab["importer"].dropna().unique())

                if countries_in_scope:
                    news_f = news_f[news_f["impacted_country"].isin(countries_in_scope)]

                # Apply date filter
                if date_range and len(date_range) == 2:
                    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    news_f = news_f[
                        (news_f["news_date_dt"] >= start_dt) &
                        (news_f["news_date_dt"] <= end_dt)
                    ]

                # If filters kill everything, fall back to all news
                if news_f.empty:
                    news_f = df_news.copy()

                # ----- Filter TARIFF rows broadly aligned with current filters -----
                tariff_f = df_tariff.copy()
                if not tariff_f.empty:
                    # date range
                    if "start_date" in tariff_f.columns:
                        tariff_f["start_date"] = pd.to_datetime(tariff_f["start_date"], errors="coerce")
                        if date_range and len(date_range) == 2:
                            _, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                            tariff_f = tariff_f[tariff_f["start_date"] <= end_dt]

                    # import & export country filters
                    if selected_import and "All" not in selected_import:
                        tariff_f = tariff_f[tariff_f["importing_country"].isin(selected_import)]
                    if selected_export and "All" not in selected_export:
                        tariff_f = tariff_f[tariff_f["exporting_country"].isin(selected_export)]

                    # category / product filters (if column exists)
                    if "product_category" in tariff_f.columns and selected_categories and "All" not in selected_categories:
                        tariff_f = tariff_f[tariff_f["product_category"].isin(selected_categories)]
                else:
                    tariff_f = pd.DataFrame()

                # ----- Define paths for Supplier AI CSVs -----
                sup_insight_path = os.path.join(data_path, "supplier_insight.csv")
                sup_recommend_path = os.path.join(data_path, "supplier_recommend.csv")

                # ----- BUTTON: run supplierInsights then supplierRecommend -----
                if st.button(
                    "Generate Insights & Recommendations",
                    type="primary",
                    use_container_width=True,
                    key="tab2_gen_ai",
                ):
                    with st.spinner(f"Generating AI insights for {sup_ai_name}..."):
                        try:
                            # 1. Insights for ONE supplier (matches supplier_ai signature)
                            sup_ins_df = supplierInsights(
                                supplier_id=sup_ai_id,
                                supplier_master=df_sup_f,
                                supplier_products=df_prod_f,
                                news_df=news_f,
                                tariff_df=tariff_f,
                                category=cat_label,
                                region=reg_label,
                                import_country=imp_label,
                                export_country=exp_label,
                                save_path=sup_insight_path,
                            )

                            # 2. Recommendations based on latest insights for that supplier
                            sup_recs_df = supplierRecommend(
                                supplier_id=sup_ai_id,
                                region=sup_ai_region or reg_label,
                                import_country=imp_label,
                                export_country=exp_label,
                                insights_path=sup_insight_path,
                                save_path=sup_recommend_path,
                            )

                            st.session_state["supplier_insights_latest"] = sup_ins_df
                            st.session_state["supplier_recs_latest"] = sup_recs_df

                            st.success("Supplier AI insights & recommendations generated!")
                        except Exception as e:
                            st.error(f"Error generating Supplier AI outputs: {e}")

                # ----- DISPLAY INSIGHTS (Blue) -----
                st.markdown("#### Insights")
                if sup_ai_name:
                    st.caption(f"Insights & recommendations below refer to supplier: {sup_ai_name} ({sup_ai_country})")

                sup_ins_latest = st.session_state.get("supplier_insights_latest")

                if isinstance(sup_ins_latest, pd.DataFrame) and not sup_ins_latest.empty:
                    bullets = []
                    for _, row in sup_ins_latest.iterrows():
                        desc = str(row.get("insight_desc") or "").strip()
                        if desc and desc.lower() not in ["nan", "n/a", "na"]:
                            bullets.append(f"- {desc}")
                    if bullets:
                        # BLUE BOX
                        st.info("\n".join(bullets))
                    else:
                        st.info("AI returned no usable insights. Try generating again.")
                else:
                    st.info("Click the button above to generate supplier AI insights.")

                # ----- DISPLAY RECOMMENDATIONS (Yellow) -----
                st.markdown("#### Recommendations")

                sup_recs_latest = st.session_state.get("supplier_recs_latest")

                if isinstance(sup_recs_latest, pd.DataFrame) and not sup_recs_latest.empty:
                    bullets = []
                    text_col = next(
                        (col for col in ["recommend_desc", "recommendation_text", "recommend_text", "text"]
                         if col in sup_recs_latest.columns),
                        "recommend_desc",
                    )
                    for _, row in sup_recs_latest.iterrows():
                        desc = str(row.get(text_col) or "").strip()
                        if desc and desc.lower() not in ["", "nan", "n/a", "na"]:
                            bullets.append(f"- {desc}")
                    if bullets:
                        # YELLOW BOX
                        st.warning("\n".join(bullets))
                    else:
                        st.info("AI returned no usable recommendations.")
                else:
                    st.info("No recommendations yet. Generate insights first.")

    st.markdown("---")
    
    # SUPPLIER DRILL-DOWN - NOW COLLAPSIBLE
    with st.expander(" **Supplier Drill-Down**", expanded=False):
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
                    if st.button(" Prev"):
                        if current_idx > 0:
                            selected_supplier_id = supplier_list[current_idx - 1]
                            st.rerun()
                with col_next:
                    if st.button("Next "):
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
                
                # Radar chart and average price per category bar chart
                col_radar, col_cat = st.columns([1, 1])
                
                with col_radar:
                    radar_fig = create_radar_chart(selected_supplier_id, df_sup_f)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                
                with col_cat:
                    st.markdown("#### Average Price per Category")
                    supplier_trades = df_trade_f[df_trade_f['supplier_id'] == selected_supplier_id]
                    
                    if not supplier_trades.empty and 'product' in supplier_trades.columns:
                        # Calculate average unit price per category
                        cat_prices = supplier_trades.groupby('product').agg({
                            'unit_price': 'mean',
                            'volume': 'sum'
                        }).reset_index()
                        cat_prices.columns = ['Category', 'Avg Price', 'Total Volume']
                        cat_prices = cat_prices.sort_values('Avg Price', ascending=False)
                        
                        # Create bar chart
                        bar_chart = alt.Chart(cat_prices).mark_bar().encode(
                            x=alt.X('Avg Price:Q', title='Average Unit Price ($)', 
                                   axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
                            y=alt.Y('Category:N', title='Product Category', 
                                   sort='-x',
                                   axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
                            color=alt.Color('Avg Price:Q', 
                                          scale=alt.Scale(scheme='blues'),
                                          legend=None),
                            tooltip=[
                                alt.Tooltip('Category:N', title='Category'),
                                alt.Tooltip('Avg Price:Q', title='Avg Price', format='$,.2f'),
                                alt.Tooltip('Total Volume:Q', title='Volume (MT)', format=',.0f')
                            ]
                        ).properties(height=400)
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.info("No price data available for this supplier.")
                
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
                    st.caption(" Blue = Volume | Orange = Value")
                else:
                    st.info("No trade history available for this supplier.")
        else:
            st.info("No suppliers available with current filters.")
    
    st.markdown("---")
    
    # SUPPLIER DOCUMENTS STATUS TABLE (below drill-down as requested)
    with st.expander(" **Supplier Document Status**", expanded=False):
        
        doc_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
        docs_table = df_sup_f[['supplier_id', 'supplier_name', 'country', 'supplier_score'] + 
                                [col for col in doc_cols if col in df_sup_f.columns]].copy()
        
        if not docs_table.empty:
            # Format supplier score
            if 'supplier_score' in docs_table.columns:
                docs_table['supplier_score'] = docs_table['supplier_score'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
            
            # Convert certification columns to tick/cross
            for col in doc_cols:
                if col in docs_table.columns:
                    cert_data = docs_table[col].fillna('N').astype(str)
                    docs_table[col] = cert_data.apply(
                        lambda x: '✓' if x.strip().upper() == 'Y' else '✗'
                    )
            
            # Rename columns
            rename_map = {
                'supplier_id': 'Supplier ID', 
                'supplier_name': 'Supplier Name', 
                'country': 'Country',
                'supplier_score': 'Supplier Score',
                'cert_gots': 'GOTS', 
                'cert_oeko': 'OEKO-TEX', 
                'cert_grs': 'GRS',
                'iso_14001': 'ISO 14001', 
                'iso_45001': 'ISO 45001'
            }
            docs_table = docs_table.rename(columns=rename_map)
            
            # Apply color styling to checkmarks
            def style_certs(val):
                if val == '✓':
                    return 'color: green; font-weight: bold; font-size: 16px'
                elif val == '✗':
                    return 'color: red; font-weight: bold; font-size: 16px'
                return ''
            
            cert_columns = ['GOTS', 'OEKO-TEX', 'GRS', 'ISO 14001', 'ISO 45001']
            existing_cert_cols = [col for col in cert_columns if col in docs_table.columns]
            
            if existing_cert_cols:
                styled_df = docs_table.style.applymap(style_certs, subset=existing_cert_cols)
                st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
            else:
                st.dataframe(docs_table, use_container_width=True, height=400, hide_index=True)
            
            st.info(f"Showing {len(docs_table)} suppliers")
        else:
            st.info("No suppliers to display with current filters.")

    # # Overall AI widgets for all suppliers in view
    # col_insight_all, col_rec_all = st.columns(2)
    
    # with col_insight_all:
    #     st.markdown("### Insights (All Suppliers)")
        
    #     if not df_sup_f.empty:
    #         avg_score = df_sup_f['supplier_score'].mean()
    #         top_country = df_sup_f.groupby('country')['supplier_score'].mean().idxmax()
    #         top_score = df_sup_f.groupby('country')['supplier_score'].mean().max()
            
    #         insights_text = f"""
    #         **Performance Overview:**
            
    #          Average score: **{avg_score:.2f}**
            
    #          Top country: **{top_country}** ({top_score:.2f})
            
    #          Total suppliers: **{len(df_sup_f)}**
    #         """
    #         st.info(insights_text)
    #     else:
    #         st.info("Adjust filters to see insights.")
    
    # with col_rec_all:
    #     st.markdown("### Recommendations (All Suppliers)")
        
    #     if not df_sup_f.empty:
    #         recommendations = []
            
    #         # Risk distribution
    #         high_risk_pct = (df_sup_f['risk_rating'] == 'High').sum() / len(df_sup_f) * 100
    #         if high_risk_pct > 20:
    #             recommendations.append(f" {high_risk_pct:.0f}% high-risk suppliers - review portfolio")
            
    #         # Geographic concentration
    #         top_country_count = df_sup_f['country'].value_counts().iloc[0]
    #         concentration = (top_country_count / len(df_sup_f) * 100)
    #         if concentration > 40:
    #             top_conc_country = df_sup_f['country'].value_counts().index[0]
    #             recommendations.append(f" {concentration:.0f}% concentrated in {top_conc_country} - diversify")
            
    #         # Low performers
    #         low_performers = (df_sup_f['supplier_score'] < 50).sum()
    #         if low_performers > 0:
    #             recommendations.append(f" {low_performers} suppliers below 50 score - capability assessment needed")
            
    #         # Performance trends
    #         if len(df_sup_f) >= 10:
    #             recommendations.append(" Sufficient supplier base for benchmarking")
    #         else:
    #             recommendations.append(" Expand supplier base for better risk distribution")
            
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
    
    # Combined metrics row - Total Suppliers + Certification percentages
    cert_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
    cols = st.columns(6)
    
    total_all_suppliers = len(df_sup)  # Total in entire dataset, not just filtered
    
    # First box: Total Suppliers
    with cols[0]:
        st.metric("Total Suppliers", f"{len(df_sup_f)}")
    
    # Next 5 boxes: Certifications
    for idx, cert in enumerate(cert_cols):
        with cols[idx + 1]:
            if cert in df_sup_f.columns:
                cert_data = df_sup_f[cert].fillna('N')
                if cert_data.dtype == 'object':
                    cert_count = (cert_data.str.strip().str.upper() == 'Y').sum()
                else:
                    cert_count = (pd.to_numeric(cert_data, errors='coerce') > 0).sum()
                
                # Calculate percentage of ALL suppliers
                pct = (cert_count / total_all_suppliers * 100) if total_all_suppliers > 0 else 0
                label = cert.replace('cert_', '').replace('iso_', 'ISO ').upper()
                st.metric(label, f"{pct:.1f}%")
            else:
                label = cert.replace('cert_', '').replace('iso_', 'ISO ').upper()
                st.metric(label, "N/A")
    st.markdown("---")
    
    # RISK DISTRIBUTION BY COUNTRY - EXPANDED BY DEFAULT (matching Tab 1 layout)
    col_chart, col_ai = st.columns([2, 1])
    
    with col_chart:
        # Risk Distribution Bar Chart
        with st.expander(" **Risk Distribution by Country**", expanded=True):
            if not df_sup_f.empty:
                # Aggregate data by country and risk level
                risk_by_country = df_sup_f.groupby(['country', 'risk_rating']).agg(
                    num_suppliers=("supplier_id", "nunique")
                ).reset_index()
                
                # Merge with trade data to get total price and volume per country-risk combination
                if not df_trade_f.empty:
                    trade_by_country_risk = df_sup_f[['supplier_id', 'country', 'risk_rating']].merge(
                        df_trade_f[['supplier_id', 'value', 'volume']], 
                        on='supplier_id', 
                        how='left'
                    )
                    
                    trade_agg = trade_by_country_risk.groupby(['country', 'risk_rating']).agg(
                        total_value=('value', 'sum'),
                        total_volume=('volume', 'sum')
                    ).reset_index()
                    
                    risk_by_country = risk_by_country.merge(trade_agg, on=['country', 'risk_rating'], how='left')
                    risk_by_country['total_value'] = risk_by_country['total_value'].fillna(0)
                    risk_by_country['total_volume'] = risk_by_country['total_volume'].fillna(0)
                else:
                    risk_by_country['total_value'] = 0
                    risk_by_country['total_volume'] = 0
                
                # Convert to millions for display
                risk_by_country['total_value_m'] = risk_by_country['total_value'] / 1_000_000
                
                # Add numeric risk order for proper sorting (Low=1, Medium=2, High=3)
                risk_order = {'Low': 1, 'Medium': 2, 'High': 3}
                risk_by_country['risk_order'] = risk_by_country['risk_rating'].map(risk_order)
                
                # Calculate total suppliers by country for sorting
                country_totals = risk_by_country.groupby('country')['num_suppliers'].sum().reset_index()
                country_totals.columns = ['country', 'total_suppliers']
                risk_by_country = risk_by_country.merge(country_totals, on='country')
                
                # Create stacked bar chart - Low (green) at left, Medium (orange) middle, High (red) at right
                stacked_chart = alt.Chart(risk_by_country).mark_bar().encode(
                    x=alt.X('num_suppliers:Q', title='Number of Suppliers',
                           axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                    y=alt.Y('country:N', title='Country',
                           sort=alt.EncodingSortField(field='total_suppliers', order='descending'),
                           axis=alt.Axis(labelFontSize=14, titleFontSize=18)),
                    color=alt.Color('risk_rating:N', 
                                   scale=alt.Scale(domain=['Low', 'Medium', 'High'], 
                                                 range=['green', 'orange', 'red']),
                                   legend=alt.Legend(title='Risk Level', titleFontSize=16, labelFontSize=14)),
                    order=alt.Order('risk_order:Q', sort='ascending'),  # Use numeric order for proper stacking
                    tooltip=[
                        alt.Tooltip('country:N', title='Country'),
                        alt.Tooltip('risk_rating:N', title='Risk Level'),
                        alt.Tooltip('num_suppliers:Q', title='Number of Suppliers', format=','),
                        alt.Tooltip('total_value_m:Q', title='Total Price ($M)', format=',.2f'),
                        alt.Tooltip('total_volume:Q', title='Total Volume (MT)', format=',.0f')
                    ]
                ).properties(height=450).configure_axis(
                    labelFontSize=14,
                    titleFontSize=18
                )
                
                st.altair_chart(stacked_chart, use_container_width=True)
                st.caption(" Stacked bars show risk distribution by country")
            else:
                st.info("No data available.")

                
    with col_ai:
        # AI INSIGHTS & RECOMMENDATIONS (Compliance AI)
        with st.expander(" **Insights & Recommendations**", expanded=True):

            # -------- pick ONE supplier for Compliance focus (highest risk, then lowest score) --------
            comp_sup_id = comp_sup_name = comp_sup_region = comp_sup_country = None
            if not df_sup_f.empty:
                risk_order = {"High": 3, "Medium": 2, "Low": 1}
                df_sup_sort = df_sup_f.copy()
                df_sup_sort["risk_order"] = df_sup_sort["risk_rating"].map(risk_order).fillna(0)
                df_sup_sort = df_sup_sort.sort_values(
                    ["risk_order", "supplier_score"],
                    ascending=[False, True]
                )
                row_comp = df_sup_sort.iloc[0]
                comp_sup_id = str(row_comp.get("supplier_id", ""))
                comp_sup_name = str(row_comp.get("supplier_name", ""))
                comp_sup_region = str(row_comp.get("region", ""))
                comp_sup_country = str(row_comp.get("country", ""))

            if not comp_sup_id:
                st.info("No suppliers in current view for Compliance AI. Adjust filters and try again.")
            else:
                # ----- build label strings to pass into the prompt (same style as Regional / Supplier AI) -----
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

                # ----- filter news for Compliance AI (similar pattern to other tabs) -----
                news_f = df_news.copy()
                if "news_date_dt" not in news_f.columns:
                    news_f["news_date_dt"] = pd.to_datetime(
                        news_f["news_date"], errors="coerce"
                    )

                countries_in_scope = set()
                if "exporter" in df_tab.columns:
                    countries_in_scope.update(df_tab["exporter"].dropna().unique())
                if "importer" in df_tab.columns:
                    countries_in_scope.update(df_tab["importer"].dropna().unique())
                # always keep supplier country in scope
                if comp_sup_country:
                    countries_in_scope.add(comp_sup_country)

                if countries_in_scope:
                    news_f = news_f[news_f["impacted_country"].isin(countries_in_scope)]

                # apply the same date_range filter
                if date_range and len(date_range) == 2:
                    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    news_f = news_f[
                        (news_f["news_date_dt"] >= start_dt) &
                        (news_f["news_date_dt"] <= end_dt)
                    ]

                # if filters kill everything, fall back to all news
                if news_f.empty:
                    news_f = df_news.copy()

                # ----- Compliance AI CSV paths -----
                comp_insight_path = os.path.join(data_path, "compliance_insight.csv")
                comp_recommend_path = os.path.join(data_path, "compliance_recommend.csv")

                # ----- BUTTON: run complianceInsights then complianceRecommend -----
                if st.button(
                    "Generate Insights & Recommendations",
                    type="primary",
                    use_container_width=True,
                    key="tab3_gen_ai",
                ):
                    with st.spinner(f"Generating Compliance AI insights for {comp_sup_name}..."):
                        try:
                            # 1) Insights
                            comp_ins_df = complianceInsights(
                                news_df=news_f,
                                region=reg_label,
                                save_path=comp_insight_path,
                                supplier_name=[comp_sup_name],
                                supplier_id=comp_sup_id,
                                supplier_region=comp_sup_region,
                                supplier_country=comp_sup_country,
                            )

                            # 2) Recommendations based on latest Compliance insights
                            comp_recs_df = complianceRecommend(
                                region=reg_label,
                                supplier_name=comp_sup_name,
                                insights_path=comp_insight_path,
                                save_path=comp_recommend_path,
                            )

                            st.session_state["comp_insights_latest"] = comp_ins_df
                            st.session_state["comp_recs_latest"] = comp_recs_df

                            st.success("Compliance AI insights & recommendations generated!")
                        except Exception as e:
                            st.error(f"Error generating Compliance AI outputs: {e}")

                # ===== DISPLAY INSIGHTS (BLUE BOX) =====
                st.markdown("#### Insights")
                st.caption(
                    f"Focus supplier for Compliance AI: {comp_sup_name} "
                    f"({comp_sup_country}, {comp_sup_region})"
                )

                comp_ins_latest = st.session_state.get("comp_insights_latest")

                if isinstance(comp_ins_latest, pd.DataFrame) and not comp_ins_latest.empty:
                    bullets = []
                    for _, row in comp_ins_latest.iterrows():
                        desc = str(row.get("insight_desc") or "").strip()
                        if desc and desc.lower() not in ["", "nan", "n/a", "na"]:
                            bullets.append(f"- {desc}")
                    if bullets:
                        # BLUE insight box
                        st.info("\n".join(bullets))
                    else:
                        st.info("Compliance AI returned no usable insights. Try generating again.")
                else:
                    st.info("Click the button above to generate Compliance AI insights.")

                # ===== DISPLAY RECOMMENDATIONS (YELLOW BOX) =====
                st.markdown("#### Recommendations")

                comp_recs_latest = st.session_state.get("comp_recs_latest")

                if isinstance(comp_recs_latest, pd.DataFrame) and not comp_recs_latest.empty:
                    text_col = next(
                        (c for c in ["recommend_desc", "recommendation_text", "recommend_text", "text"]
                         if c in comp_recs_latest.columns),
                        "recommend_desc",
                    )
                    bullets = []
                    for _, row in comp_recs_latest.iterrows():
                        desc = str(row.get(text_col) or "").strip()
                        if desc and desc.lower() not in ["", "nan", "n/a", "na"]:
                            bullets.append(f"- {desc}")
                    if bullets:
                        # YELLOW recommendation box
                        st.warning("\n".join(bullets))
                    else:
                        st.info("Compliance AI returned no usable recommendations.")
                else:
                    st.info("No recommendations yet. Generate insights first.")


    # News Feed Widget (cloned from Tab 1)
    st.markdown("---")
    st.markdown("####  Compliance & Risk News")
    
    if not df_news.empty and 'news_date' in df_news.columns:
        # Filter news based on selected export countries
        df_news_filtered = df_news.copy()
        if selected_export and 'All' not in selected_export:
            df_news_filtered = df_news_filtered[df_news_filtered['impacted_country'].isin(selected_export)]
        
        # Sort by date and get top 5
        df_news_sorted = df_news_filtered.sort_values('news_date', ascending=False).head(5)
        
        if not df_news_sorted.empty:
            for _, news in df_news_sorted.iterrows():
                with st.expander(f" {news['news_title']}", expanded=False):
                    st.markdown(f"**Date:** {news['news_date'].strftime('%Y-%m-%d') if pd.notna(news['news_date']) else 'N/A'}")
                    st.markdown(f"**Country:** {news['impacted_country']}")
                    st.markdown(f"**Category:** {news['news_category']}")
                    st.markdown(f"**Summary:** {news['news_summary']}")
        else:
            st.info("No recent news available for selected filters.")
    else:
        st.info("No news data available.")
    
    st.markdown("---")

    # Missing documents section - now collapsible
    with st.expander(" **Supplier Document Status**", expanded=False):
        
        doc_cols = ['cert_gots', 'cert_oeko', 'cert_grs', 'iso_14001', 'iso_45001']
        missing_docs = df_sup_f[['supplier_id', 'supplier_name', 'country'] + 
                                [col for col in doc_cols if col in df_sup_f.columns]].copy()
        
        # Initialize session state for showing scores
        if 'show_supplier_scores' not in st.session_state:
            st.session_state['show_supplier_scores'] = False
        
        # Supplier score column is blank by default
        missing_docs['supplier_score'] = ''
        
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
            
            display_cols = ['supplier_id', 'supplier_name', 'country'] + \
                          [col for col in doc_cols if col in df_sup_f.columns] + ['supplier_score']
            
            display_df = suppliers_with_missing[display_cols].copy()
            
            # NEW: Add Generate Score button near the table
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button(" Generate Scores", type="primary", key="gen_scores_btn"):
                    st.session_state['show_supplier_scores'] = True
                    st.rerun()
            
            # Fill in the supplier scores if button was clicked
            if st.session_state['show_supplier_scores'] and 'supplier_score' in df_sup_f.columns:
                score_map = df_sup_f.set_index('supplier_id')['supplier_score'].to_dict()
                display_df['supplier_score'] = display_df['supplier_id'].map(score_map).apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ''
                )
            
            # Convert certification columns to tick/cross
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
            
            # Apply color styling to tick/cross
            def style_certs(val):
                if val == '✓':
                    return 'color: green; font-weight: bold; font-size: 16px'
                elif val == '✗':
                    return 'color: red; font-weight: bold; font-size: 16px'
                return ''
            
            cert_columns = ['GOTS', 'OEKO-TEX', 'GRS', 'ISO 14001', 'ISO 45001']
            existing_cert_cols = [col for col in cert_columns if col in display_df.columns]
            
            if existing_cert_cols:
                styled_df = display_df.style.applymap(style_certs, subset=existing_cert_cols)
                st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
            else:
                st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            
            st.info(f"Showing {len(display_df)} suppliers")
        else:
            st.success("All suppliers have submitted required documents!")
    
    st.markdown("---")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("Sustainable Supplier Intelligence by Team PROTΞGΞ")