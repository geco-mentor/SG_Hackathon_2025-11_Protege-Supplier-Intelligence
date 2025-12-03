# api/regional_ai.py  (or same folder as supplier_scoring.py)

import io
import pandas as pd
import requests
from datetime import datetime

from api.config import BITDEER_API_KEY, BITDEER_API_URL, BITDEER_MODEL

def _parse_model_csv(model_output, expected_cols):
    """
    Parse a semi-colon separated CSV string from the model.
    If a row has too many semicolons, merge the extras into the last column.
    """
    lines = [l.strip() for l in model_output.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Model returned empty output.")

    # Header
    header_parts = [h.strip() for h in lines[0].split(";")]
    if len(header_parts) == len(expected_cols):
        cols = header_parts
    else:
        cols = expected_cols

    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(";")]

        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            # Merge extra pieces into last column (usually the text)
            parts = parts[: len(cols) - 1] + [";".join(parts[len(cols) - 1 :])]

        rows.append(parts)

    return pd.DataFrame(rows, columns=cols)


def regionalInsights(
    news_df: pd.DataFrame,
    category: str,
    region: str,
    import_country: str,
    export_country: str,
    save_path: str = "regional_insight.csv",  # e.g. "./data/regional_insight.csv"
) -> pd.DataFrame:

    """
    Use GPT-OSS (Bitdeer OpenAI-compatible API) to generate 5 ESG & Trade
    insights for the current Regional Sourcing view.

    - Takes the current filtered news_df from the dashboard
    - Uses region + import/export country filters as context
    - Converts news_df to semi-colon CSV
    - Injects that CSV + context into the prompt
    - Parses the returned CSV (with ';' delimiter)
    - Appends the new insights into regional_insight.csv (if save_path not None)

    Expected output columns:
        insight_id;timestamp;region;country;insight_desc
    """

    if news_df is None or news_df.empty:
        raise ValueError("news_df is empty – there is no news to generate insights from.")

    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    required_cols = [
        "news_id",
        "news_date",
        "impacted_country",
        "news_category",
        "news_title",
        "news_summary",
    ]
    missing = [c for c in required_cols if c not in news_df.columns]
    if missing:
        raise ValueError(f"news_df is missing required columns: {missing}")

    # ---- 1. Prepare subset and CSV ----
    subset_df = news_df[required_cols].copy()
    subset_df["news_date"] = subset_df["news_date"].astype(str)
    news_csv = subset_df.to_csv(index=False, sep=";")

    # Region / import / export shown to the model as context
    filter_context = (
        f"Product Category: {category or 'All'}, "
        f"Region: {region or 'All'}, "
        f"Import Country: {import_country or 'All'}, "
        f"Export Country: {export_country or 'All'}"
    )

    # ---- 2. Inline prompt template ----
    prompt_template = """
You are a Regional Sourcing Manager with over 10 years of Professional Experience in the 
Supply Chain Management and Textile Manufacturing Business Spaces.

You are working under the employment of a Supply Chain Company that deals with 
the Sourcing of Raw Materials & Intermediate Goods relating to the Textile Manufacturing 
space across the globe.

Your counterparties within the organization include:

1. Sourcing Analyst: Whenever a Prospective Supplier Name has been identified for 
onboarding, you will send an e-mail request for this person to assist with data maintenance 
and market price analysis relating to the Prospective Supplier's offering.

2. Sustainability Compliance Officer: Whenever a Prospective Supplier Name is ready for 
onboarding, your Sourcing Analyst colleague will send an e-mail request to the Compliance 
Team for Risk Assessment, Insight Generation, and Recommendations on Corrective Actions to 
help improve the Supplier Score.

Your counterparties outside the organization include Textile Material Suppliers from across 
the globe.

Dashboard Filters that are currently applied by the user:
{{current_filter_context}}

Filtered Records from the "news_master" table in CSV format below 
with semi-colon (;) as the delimiter:
{{current_filtered_news_dataframe}}

Each record in the "news_master" table contains 1 news article.
You MUST analyze based on the data on this table only.
Do NOT invent any fictitious events or details.

Your Task:
1. Analyze the news records and identify key topics relating to that matter for
   textile sourcing in the current region / country.
2. Generate EXACTLY 5 concise INSIGHTS that summarise "what is going on" from
   the perspective of a Regional Sourcing Manager.
   - Each insight must be at most 50 words.
   - Where possible, combine multiple related articles into one insight
     (trend-level summary).
3. For each insight:
   - Use the most relevant impacted_country as the "country" value.
     If the insight clearly spans several countries, use "Multiple".
   - Use the region from the filter context as the "region".
   - Use today's date in YYYY-MM-DD format as the "timestamp".
4. In the event that there is insufficient data available to generate Insights for the currently filtered region and/or country, search the public internet for the latest available information. Afterwards, you MUST still return 5 insights.
   You MUST NOT leave any insight_desc empty, or write "N/A", "NA" or "nan".

OUTPUT FORMAT (VERY IMPORTANT):
Return ONLY a CSV table, using semi-colon (;) as the delimiter, with the
following columns:

insight_id;timestamp;region;country;insight_desc

Column rules:
- insight_id: integers 1 to 5.
- timestamp: today's date in YYYY-MM-DD format.
- region: region name from the filter context (or "All" if not specified).
- country: the main impacted country for that insight, or "Multiple".
- insight_desc: the business insight text (max 50 words). This field
  must NEVER be empty and must NEVER contain "N/A", "NA" or "nan".

Do NOT return any explanation, notes or commentary outside of the CSV table.
"""

    # Inject the CSV + filter context into the placeholder
    prompt = (
        prompt_template.replace("{{current_filtered_news_dataframe}}", news_csv)
        .replace("{{current_filter_context}}", filter_context)
    )

    # ---- 3. Call GPT-OSS via Bitdeer API (OpenAI chat format) ----
    payload = {
        "model": BITDEER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 15360,
        "temperature": 0.2,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {BITDEER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        BITDEER_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    try:
        model_output = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Bitdeer response format: {data}") from exc

    # ---- 4. Parse model CSV output ----
    expected_cols = ["insight_id", "timestamp", "region", "country", "insight_desc"]
    new_insights = _parse_model_csv(model_output, expected_cols)

    # Clean and handle insufficient-data blanks
    if "insight_desc" in new_insights.columns:
        new_insights["insight_desc"] = (
            new_insights["insight_desc"].astype(str).str.strip()
        )
        mask_blank = new_insights["insight_desc"].isin(
            ["", "nan", "NaN", "N/A", "NA"]
        )
        new_insights.loc[mask_blank, "insight_desc"] = (
            "Limited recent ESG & trade news in this view; maintain routine "
            "monitoring and supplier risk checks for this region/country."
        )

    # Safety backfill if model omits fields
    if "timestamp" not in new_insights.columns:
        new_insights["timestamp"] = datetime.utcnow().date().isoformat()
    if "region" not in new_insights.columns:
        new_insights["region"] = region or "All"
    if "country" not in new_insights.columns:
        # Fallback: use a specific import/export country if available, else "Multiple"
        if import_country and import_country != "Multiple":
            default_country = import_country
        elif export_country and export_country != "Multiple":
            default_country = export_country
        else:
            default_country = "Multiple"
        new_insights["country"] = default_country

    # ---- 5. Append to CSV if requested ----
    if save_path is not None:
        try:
            existing = pd.read_csv(save_path, sep=";")
            combined = pd.concat([existing, new_insights], ignore_index=True)
        except FileNotFoundError:
            combined = new_insights.copy()

        combined.to_csv(save_path, index=False, sep=";")

    return new_insights



def regionalRecommend(
    news_df: pd.DataFrame,        # currently unused, kept for symmetry / future
    category: str,
    region: str,
    import_country: str,
    export_country: str,
    insights_path: str = "regional_insight.csv",
    save_path: str = "regional_recommend.csv",
) -> pd.DataFrame:

    """
    Use GPT-OSS to generate 5 sourcing recommendations based on the LATEST
    batch of insights for the selected region / country.

    - Reads regional_insight.csv
    - Filters by region/country (context also includes import/export)
    - Takes the latest timestamp batch
    - Sends that as CSV into the Recommendations prompt
    - Appends new recommendations into regional_recommend.csv

    Expected output columns:
        recommend_id;timestamp;region;country;recommend_desc
    """

    # ---- 1. Read existing insights & get latest batch ----
    try:
        insights_df = pd.read_csv(insights_path, sep=";")
    except FileNotFoundError:
        raise ValueError("regional_insight.csv not found - generate insights first.")

    if insights_df.empty:
        raise ValueError("regional_insight.csv is empty - generate insights first.")

    df = insights_df.copy()
    if "region" in df.columns and region and region != "All":
        df = df[df["region"] == region]
    # Country column in insights = impacted_country / Multiple (we keep same logic)
    if "country" in df.columns and import_country and import_country not in ("All", "Multiple"):
        df = df[df["country"] == import_country]

    if df.empty:
        raise ValueError("No insights found for the specified region/country.")

    if "timestamp" not in df.columns:
        raise ValueError("Insights table must contain a 'timestamp' column.")

    # 🔧 NEW: make timestamp comparable (avoid float/str mix)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].isna().all():
        raise ValueError("No valid timestamps available in insights table.")

    latest_ts = df["timestamp"].max()
    latest_batch = df[df["timestamp"] == latest_ts].copy()

    insights_csv = latest_batch.to_csv(index=False, sep=";")
    filter_context = (
        f"Product Category: {category or 'All'}, "
        f"Region: {region or 'All'}, "
        f"Import Country: {import_country or 'All'}, "
        f"Export Country: {export_country or 'All'}, "
        f"Latest insights timestamp: {latest_ts}"
    )


    # ---- 2. Prompt template for recommendations ----
    prompt_template = """
You are a Regional Sourcing Manager with over 10 years of Professional Experience in the 
Supply Chain Management and Textile Manufacturing Business Spaces.

You are working under the employment of a Supply Chain Company that deals with 
the Sourcing of Raw Materials & Intermediate Goods relating to the Textile Manufacturing 
space across the globe.

Your counterparties within the organization include:

1. Sourcing Analyst: Whenever a Prospective Supplier Name has been identified for 
onboarding, you will send an e-mail request for this person to assist with data maintenance 
and market price analysis relating to the Prospective Supplier's offering.

2. Sustainability Compliance Officer: Whenever a Prospective Supplier Name is ready for 
onboarding, your Sourcing Analyst colleague will send an e-mail request to the Compliance 
Team for Risk Assessment, Insight Generation, and Recommendations on Corrective Actions to 
help improve the Supplier Score.

Your counterparties outside the organization include Textile Material Suppliers from across 
the globe.

The current dashboard context is:
{{current_filter_context}}

The latest insights are provided below as a CSV table with semi-colon (;)
as the delimiter:
{{latest_insights_dataframe}}

You MUST base your recommendations ONLY on these insights.
Do NOT assume additional events or data beyond what is shown.

YOUR TASKS:
1. Derive what actions a Textile Regional Sourcing Manager should take to
   manage risk and secure supply (e.g. proceed, proceed with caution, pause,
   diversify, step up monitoring, renegotiate contracts).
2. Generate EXACTLY 5 RECOMMENDATIONS.
   - Each recommendation must be at most 50 words.
   - Recommendations should be practical and business-focused
     (not generic AI statements).
3. In the event that there is insufficient data available to generate Recommendations for the currently filtered region and/or country, search the public internet for the latest available information. Afterwards, you MUST still return 5 Recommendations.
   You MUST NOT leave any recommend_desc empty, or write "N/A", "NA" or "nan".

4. For each recommendation:
   - Use the region and country values that are most relevant from the insights.
   - Use today's date in YYYY-MM-DD format as the timestamp.

OUTPUT FORMAT (VERY IMPORTANT):
Return ONLY a CSV table, using semi-colon (;) as the delimiter, with the
following columns:

recommend_id;timestamp;region;country;recommend_desc

Column rules:
- recommend_id: integers 1 to 5.
- timestamp: today's date in YYYY-MM-DD format.
- region: region name (may repeat across rows).
- country: country name (may repeat across rows).
- recommend_desc: the recommended action (max 35 words). This field
  must NEVER be empty and must NEVER contain "N/A", "NA" or "nan".

Do NOT return any explanation, notes or commentary outside of the CSV table.
"""

    prompt = (
        prompt_template.replace("{{latest_insights_dataframe}}", insights_csv)
        .replace("{{current_filter_context}}", filter_context)
    )

    # ---- 3. Call GPT-OSS ----
    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    payload = {
        "model": BITDEER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 15360,
        "temperature": 0.2,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {BITDEER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        BITDEER_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    try:
        model_output = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Bitdeer response format: {data}") from exc

    # ---- 4. Parse output and append ----
    expected_cols = ["recommend_id", "timestamp", "region", "country", "recommend_desc"]
    new_recs = _parse_model_csv(model_output, expected_cols)

    # Clean and handle blanks in recommendations
    if "recommend_desc" in new_recs.columns:
        new_recs["recommend_desc"] = (
            new_recs["recommend_desc"].astype(str).str.strip()
        )
        mask_blank = new_recs["recommend_desc"].isin(
            ["", "nan", "NaN", "N/A", "NA"]
        )
        new_recs.loc[mask_blank, "recommend_desc"] = (
            "Limited or low-clarity risk signals; continue monitoring, validate "
            "supplier data, and keep contingency sourcing options ready for this region."
        )

    if "timestamp" not in new_recs.columns:
        new_recs["timestamp"] = datetime.utcnow().date().isoformat()
    if "region" not in new_recs.columns:
        new_recs["region"] = region or "All"
    if "country" not in new_recs.columns:
        # Same fallback logic as insights
        if import_country and import_country != "Multiple":
            default_country = import_country
        elif export_country and export_country != "Multiple":
            default_country = export_country
        else:
            default_country = "Multiple"
        new_recs["country"] = default_country

    if save_path is not None:
        try:
            existing_recs = pd.read_csv(save_path, sep=";")
            combined = pd.concat([existing_recs, new_recs], ignore_index=True)
        except FileNotFoundError:
            combined = new_recs.copy()

        combined.to_csv(save_path, index=False, sep=";")

    return new_recs

