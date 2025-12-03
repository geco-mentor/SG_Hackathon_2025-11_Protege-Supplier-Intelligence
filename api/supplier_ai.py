# api/supplier_ai.py

import io
from datetime import datetime

import pandas as pd
import requests

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


def supplierInsights(
    supplier_id: int,
    supplier_master: pd.DataFrame,
    supplier_products: pd.DataFrame,
    news_df: pd.DataFrame,
    tariff_df: pd.DataFrame,
    category: str,
    region: str,
    import_country: str,
    export_country: str,
    save_path: str = "supplier_esg_insight.csv",
) -> pd.DataFrame:
    """
    Generate 5 ESG & Trade INSIGHTS for a single supplier from the perspective
    of a Sourcing Analyst, using the same CSV-style pattern as regional_ai.py.

    Expected output columns:
        insight_id;timestamp;supplier_id;supplier_name;region;country;product_category;insight_desc
    """

    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    # ---- 1. Look up the supplier + basic validation ----
    req_cols_master = [
        "supplier_id",
        "supplier_name",
        "country",
        "region",
        "certified_audited",
        "code_conduct",
        "timely_delivery",
        "product_quality",
        "service_reliability",
        "esg_environment",
        "esg_social",
        "esg_governance",
        "supplier_score",
        "risk_rating",
    ]
    missing_master = [c for c in req_cols_master if c not in supplier_master.columns]
    if missing_master:
        raise ValueError(f"supplier_master is missing required columns: {missing_master}")

    sup_row_df = supplier_master[supplier_master["supplier_id"] == supplier_id]
    if sup_row_df.empty:
        raise ValueError(f"No supplier found with supplier_id={supplier_id}")

    sup_row = sup_row_df.iloc[0]
    sup_region = sup_row["region"]
    sup_country = sup_row["country"]
    sup_name = sup_row["supplier_name"]

    # Single supplier row as CSV
    supplier_profile_csv = sup_row_df[req_cols_master].to_csv(index=False, sep=";")

    # ---- 2. Products for this supplier ----
    if not supplier_products.empty:
        if "supplier_id" not in supplier_products.columns or "product_category" not in supplier_products.columns:
            raise ValueError("supplier_products must contain 'supplier_id' and 'product_category' columns.")
        prod_subset = supplier_products[supplier_products["supplier_id"] == supplier_id].copy()
    else:
        prod_subset = pd.DataFrame(columns=["supplier_id", "product_category"])

    products_csv = prod_subset.to_csv(index=False, sep=";")

    # ---- 3. ESG & trade news (already filtered by dashboard) ----
    if news_df is None or news_df.empty:
        raise ValueError("news_df is empty – there is no news to generate supplier insights from.")

    req_cols_news = [
        "news_id",
        "news_date",
        "impacted_country",
        "news_category",
        "news_title",
        "news_summary",
    ]
    missing_news = [c for c in req_cols_news if c not in news_df.columns]
    if missing_news:
        raise ValueError(f"news_df is missing required columns: {missing_news}")

    news_subset = news_df[req_cols_news].copy()
    news_subset["news_date"] = news_subset["news_date"].astype(str)
    news_csv = news_subset.to_csv(index=False, sep=";")

    # ---- 4. Tariff rows relevant for this supplier (optional) ----
    tariff_csv = ""
    if tariff_df is not None and not tariff_df.empty:
        req_cols_tariff = [
            "tariff_id",
            "start_date",
            "importing_country",
            "product_category",
            "exporting_country",
            "tariff_pct",
        ]
        missing_tariff = [c for c in req_cols_tariff if c not in tariff_df.columns]
        if missing_tariff:
            raise ValueError(f"tariff_df is missing required columns: {missing_tariff}")

        # Heuristic: tariffs where exporting_country == supplier country
        subset_tariff = tariff_df[tariff_df["exporting_country"] == sup_country].copy()
        subset_tariff["start_date"] = subset_tariff["start_date"].astype(str)
        tariff_csv = subset_tariff.to_csv(index=False, sep=";")
    else:
        subset_tariff = pd.DataFrame(
            columns=["tariff_id", "start_date", "importing_country", "product_category", "exporting_country", "tariff_pct"]
        )

    # ---- 5. Filter context shown to the model ----
    filter_context = (
        f"Selected Supplier: {sup_name} (ID {supplier_id}), "
        f"Supplier Country: {sup_country}, Supplier Region: {sup_region}, "
        f"Focus Product Category: {category or 'All'}, "
        f"Dashboard Region Filter: {region or 'All'}, "
        f"Import Country: {import_country or 'All'}, "
        f"Export Country: {export_country or 'All'}"
    )

    # ---- 6. Prompt template – SUPPLIER ESG INSIGHTS (Sourcing Analyst lens) ----
    prompt_template = """
You are a Senior Sourcing Analyst Officer with over 5 years of Professional Experience in 
the Supply Chain Management and Textile Manufacturing Business Spaces.

You are working under the employment of a Supply Chain Company that deals with 
the Sourcing of Raw Materials & Intermediate Goods relating to the Textile Manufacturing 
space across the globe.

The following high-level dashboard context is applied:
{{current_filter_context}}

You are given FOUR related data tables, all already filtered by the dashboard:

1) "supplier_master" – ONE row for the currently selected supplier.
   Columns include:
   - supplier_id
   - supplier_name
   - country
   - region
   - certified_audited
   - code_conduct
   - timely_delivery
   - product_quality
   - service_reliability
   - esg_environment
   - esg_social
   - esg_governance
   - supplier_score
   - risk_rating

   The selected supplier row in semi-colon CSV format:
   {{current_supplier_profile}}

2) "supplier_products" – list of product categories for this supplier.
   Columns:
   - supplier_id
   - product_category

   The products for this supplier in semi-colon CSV format:
   {{current_supplier_products}}

3) "news_master" – ESG & Trade news related to textiles.
   Columns:
   - news_id
   - news_date (YYYY.MM.DD or YYYY-MM-DD)
   - impacted_country
   - news_category
   - news_title
   - news_summary

   The filtered ESG & Trade news in semi-colon CSV format:
   {{current_filtered_news}}

4) "trade_tariff" – tariff % on textile products across importing/exporting countries.
   Columns:
   - tariff_id
   - start_date (YYYY.MM.DD)
   - importing_country
   - product_category
   - exporting_country
   - tariff_pct

   The relevant tariff rows in semi-colon CSV format (may be empty):
   {{current_tariff_rows}}

Each row you see is REAL data from the dashboard. As the 3 roles working together,
you MUST base your analysis ONLY on these tables. Do NOT invent additional events,
dates, countries or metrics that are not clearly present in the data.

----------------------------------------------------------------------
YOUR JOINT TASK – SUPPLIER ESG & TRADE INSIGHTS (5 LINES)
----------------------------------------------------------------------
From the viewpoint of a Sourcing Analyst, and using the support of the
Regional Sourcing Manager and Compliance Officer:

1. Read the supplier profile, supplier products, ESG & trade news and tariff rows.
2. Identify ESG and trade THEMES that matter for this supplier, such as:
   - Certification status and audit gaps
   - Delivery, quality and service reliability
   - Environmental, social or governance weaknesses
   - Country-level labour unrest or natural disasters
   - Tariff exposure and price competitiveness
3. Generate EXACTLY 5 concise INSIGHTS that explain:
   "What have you understood about this supplier's risk, ESG profile and trade position?"

For each INSIGHT:

- Use today's date in YYYY-MM-DD format as the "timestamp".
- Use the supplier_id and supplier_name from the supplier profile.
- Use the supplier's country and region from the supplier profile.
- For product_category:
  - If a specific product is clearly the focus of the insight (e.g. Denim),
    use that product name.
  - Otherwise, use "All".
- Keep "insight_desc" to a maximum of 50 words and make it factual.
- Where possible, combine multiple data points into a single trend-level insight
  (e.g. low audit scores + negative news + tariff changes).

----------------------------------------------------------------------
INSUFFICIENT DATA RULE
----------------------------------------------------------------------
If there is very little or no data available for this supplier in the
tables above (e.g. no news or no tariffs), you MUST:
1) Search the public internet for the latest available information that is
   relevant to the supplier's country, region and ESG/trade context.
2) STILL return 5 INSIGHTS.

You MUST NOT leave any insight_desc empty, or write "N/A", "NA" or "nan".

----------------------------------------------------------------------
OUTPUT FORMAT (VERY IMPORTANT)
----------------------------------------------------------------------
Return ONLY a CSV table, using semi-colon (;) as the delimiter, with the
following columns:

insight_id;timestamp;supplier_id;supplier_name;region;country;product_category;insight_desc

Column rules:
- insight_id: integers 1 to 5.
- timestamp: today's date in YYYY-MM-DD format.
- supplier_id: ID of the supplier in the supplier_master table.
- supplier_name: name of the supplier in the supplier_master table.
- region: supplier region from the supplier_master table.
- country: supplier country from the supplier_master table.
- product_category: key product in focus for the insight, or "All".
- insight_desc: the supplier ESG & trade insight text (max 50 words).
  This field must NEVER be empty and must NEVER contain "N/A", "NA" or "nan".

Do NOT return any explanation, notes or commentary outside of the CSV table.
"""

    prompt = (
        prompt_template.replace("{{current_filter_context}}", filter_context)
        .replace("{{current_supplier_profile}}", supplier_profile_csv)
        .replace("{{current_supplier_products}}", products_csv)
        .replace("{{current_filtered_news}}", news_csv)
        .replace("{{current_tariff_rows}}", tariff_csv)
    )

    # ---- 7. Call GPT-OSS via Bitdeer API (OpenAI chat format) ----
    payload = {
        "model": BITDEER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
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

    expected_cols = [
        "insight_id",
        "timestamp",
        "supplier_id",
        "supplier_name",
        "region",
        "country",
        "product_category",
        "insight_desc",
    ]
    new_insights = _parse_model_csv(model_output, expected_cols)

    # Clean and handle blanks
    if "insight_desc" in new_insights.columns:
        new_insights["insight_desc"] = new_insights["insight_desc"].astype(str).str.strip()
        mask_blank = new_insights["insight_desc"].isin(["", "nan", "NaN", "N/A", "NA"])
        new_insights.loc[mask_blank, "insight_desc"] = (
            "Limited structured data for this supplier; maintain close monitoring and "
            "request updated ESG, certification and performance information before "
            "increasing order volumes."
        )

    # Backfill metadata if model omitted some columns
    if "timestamp" not in new_insights.columns:
        new_insights["timestamp"] = datetime.utcnow().date().isoformat()
    if "supplier_id" not in new_insights.columns:
        new_insights["supplier_id"] = supplier_id
    if "supplier_name" not in new_insights.columns:
        new_insights["supplier_name"] = sup_name
    if "region" not in new_insights.columns:
        new_insights["region"] = sup_region
    if "country" not in new_insights.columns:
        new_insights["country"] = sup_country
    if "product_category" not in new_insights.columns:
        new_insights["product_category"] = category or "All"

    # ---- 8. Append to CSV if requested ----
    if save_path is not None:
        try:
            existing = pd.read_csv(save_path, sep=";")
            combined = pd.concat([existing, new_insights], ignore_index=True)
        except FileNotFoundError:
            combined = new_insights.copy()

        combined.to_csv(save_path, index=False, sep=";")

    return new_insights


def supplierRecommend(
    supplier_id: int,
    region: str,
    import_country: str,
    export_country: str,
    insights_path: str = "supplier_esg_insight.csv",
    save_path: str = "supplier_recommend.csv",
) -> pd.DataFrame:
    """
    Generate 5 sourcing RECOMMENDATIONS for a single supplier based on the
    latest batch of supplier ESG & trade insights.

    Expected output columns:
        recommend_id;timestamp;supplier_id;supplier_name;region;country;recommend_desc
    """

    # ---- 1. Read existing supplier insights & get latest batch for this supplier ----
    try:
        insights_df = pd.read_csv(insights_path, sep=";")
    except FileNotFoundError:
        raise ValueError("supplier_esg_insight.csv not found - generate insights first.")

    if insights_df.empty:
        raise ValueError("supplier_esg_insight.csv is empty - generate insights first.")

    if "supplier_id" not in insights_df.columns:
        raise ValueError("Insights table must contain 'supplier_id' column.")

    df = insights_df[insights_df["supplier_id"] == supplier_id].copy()
    if df.empty:
        raise ValueError(f"No insights found for supplier_id={supplier_id}.")

    if "timestamp" not in df.columns:
        raise ValueError("Insights table must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().all():
        raise ValueError("No valid timestamps available in supplier insights table.")

    latest_ts = df["timestamp"].max()
    latest_batch = df[df["timestamp"] == latest_ts].copy()

    # Pull metadata for filter context
    sup_name = latest_batch["supplier_name"].iloc[0] if "supplier_name" in latest_batch.columns else "Unknown"
    sup_region = latest_batch["region"].iloc[0] if "region" in latest_batch.columns else region or "Unknown"
    sup_country = latest_batch["country"].iloc[0] if "country" in latest_batch.columns else "Multiple"

    insights_csv = latest_batch.to_csv(index=False, sep=";")

    filter_context = (
        f"Selected Supplier: {sup_name} (ID {supplier_id}), "
        f"Supplier Region: {sup_region}, Supplier Country: {sup_country}, "
        f"Dashboard Region Filter: {region or 'All'}, "
        f"Import Country: {import_country or 'All'}, "
        f"Export Country: {export_country or 'All'}, "
        f"Latest insights timestamp: {latest_ts.date()}"
    )

    # ---- 2. Prompt template – SUPPLIER RECOMMENDATIONS ----
    prompt_template = """
You are an AI co-pilot made up of three virtual roles working together:
1) Sourcing Analyst – evaluates supplier performance, ESG profile and tariffs
   to recommend sourcing actions.
2) Regional Sourcing Manager (Textiles) – owns supplier portfolio strategy
   for the region.
3) Compliance Officer – ensures ESG and regulatory expectations are met.

You are given the LATEST set of ESG & Trade insights for ONE supplier.

The current dashboard context is:
{{current_filter_context}}

The latest supplier insights are provided below as a CSV table with
semi-colon (;) as the delimiter:

{{latest_supplier_insights}}

Each row has the columns:
- insight_id
- timestamp
- supplier_id
- supplier_name
- region
- country
- product_category
- insight_desc

As the 3 roles working together, you MUST base your recommendations ONLY on
these insights. Do NOT assume additional data beyond what is shown, unless
you are explicitly told to search the public internet.

----------------------------------------------------------------------
YOUR JOINT TASK – SUPPLIER RECOMMENDATIONS (5 LINES)
----------------------------------------------------------------------
From the viewpoint of a Sourcing Analyst advising a Regional Sourcing Manager:

1. Decide how to handle this supplier in the short to medium term, considering:
   - certification and ESG gaps,
   - delivery / quality / reliability performance,
   - country and tariff risks,
   - price and competitiveness versus alternatives (where visible).
2. Generate EXACTLY 5 RECOMMENDATIONS.
   - Each recommendation must be at most 50 words.
   - Recommendations should be practical and business-focused
     (not generic AI statements).
   - Where appropriate, suggest how to help the supplier improve
     (e.g. corrective action plans) while protecting the buyer's risk profile.

----------------------------------------------------------------------
INSUFFICIENT DATA RULE
----------------------------------------------------------------------
If there is insufficient insight data for this supplier, you MUST:
1) Search the public internet for the latest available information relevant
   to this supplier's country, region and ESG/trade context.
2) STILL return 5 RECOMMENDATIONS.

You MUST NOT leave any recommend_desc empty, or write "N/A", "NA" or "nan".

----------------------------------------------------------------------
OUTPUT FORMAT (VERY IMPORTANT)
----------------------------------------------------------------------
Return ONLY a CSV table, using semi-colon (;) as the delimiter, with the
following columns:

recommend_id;timestamp;supplier_id;supplier_name;region;country;recommend_desc

Column rules:
- recommend_id: integers 1 to 5.
- timestamp: today's date in YYYY-MM-DD format.
- supplier_id: ID of the supplier that these recommendations refer to.
- supplier_name: supplier name.
- region: supplier region.
- country: supplier country.
- recommend_desc: the recommended sourcing / remediation action (max 50 words).
  This field must NEVER be empty and must NEVER contain "N/A", "NA" or "nan".

Do NOT return any explanation, notes or commentary outside of the CSV table.
"""

    prompt = (
        prompt_template.replace("{{current_filter_context}}", filter_context)
        .replace("{{latest_supplier_insights}}", insights_csv)
    )

    # ---- 3. Call GPT-OSS via Bitdeer API ----
    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    payload = {
        "model": BITDEER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
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

    expected_cols = [
        "recommend_id",
        "timestamp",
        "supplier_id",
        "supplier_name",
        "region",
        "country",
        "recommend_desc",
    ]
    new_recs = _parse_model_csv(model_output, expected_cols)

    # Clean and handle blanks in recommendations
    if "recommend_desc" in new_recs.columns:
        new_recs["recommend_desc"] = new_recs["recommend_desc"].astype(str).str.strip()
        mask_blank = new_recs["recommend_desc"].isin(["", "nan", "NaN", "N/A", "NA"])
        new_recs.loc[mask_blank, "recommend_desc"] = (
            "Information is limited; keep this supplier under monitoring, "
            "maintain contingency options, and request a clear improvement "
            "plan on certification, ESG and delivery performance before "
            "increasing order volumes."
        )

    # Backfill any missing metadata
    if "timestamp" not in new_recs.columns:
        new_recs["timestamp"] = datetime.utcnow().date().isoformat()
    if "supplier_id" not in new_recs.columns:
        new_recs["supplier_id"] = supplier_id
    if "supplier_name" not in new_recs.columns:
        new_recs["supplier_name"] = sup_name
    if "region" not in new_recs.columns:
        new_recs["region"] = sup_region
    if "country" not in new_recs.columns:
        new_recs["country"] = sup_country

    # ---- 4. Append to CSV if requested ----
    if save_path is not None:
        try:
            existing_recs = pd.read_csv(save_path, sep=";")
            combined = pd.concat([existing_recs, new_recs], ignore_index=True)
        except FileNotFoundError:
            combined = new_recs.copy()

        combined.to_csv(save_path, index=False, sep=";")

    return new_recs
