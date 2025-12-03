# api/compliance_ai.py

import os
import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional
from api.config import BITDEER_API_KEY, BITDEER_API_URL, BITDEER_MODEL


def _parse_model_csv(model_output: str, expected_cols):
    # Parse a semi-colon separated CSV string from the model.
    # If a row has too many semicolons, merge the extras into the last column.
    lines = [l.strip() for l in model_output.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Model returned empty output.")

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
            # Merge extra pieces into last column (usually the text field)
            parts = parts[: len(cols) - 1] + [";".join(parts[len(cols) - 1 :])]
        rows.append(parts)

    return pd.DataFrame(rows, columns=cols)


# 1. Risk & Compliance Insights #
# ----------------------------- #

def complianceInsights(
    news_df: pd.DataFrame,
    region: Optional[str],
    save_path: str = "risk_compliance_insight.csv",
    supplier_name: Optional[List[str]] = None,
    supplier_id: Optional[str] = None,
    supplier_region: Optional[str] = None,
    supplier_country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate 5 Sustainability Risk & Compliance INSIGHTS based on Global News,
    from the perspective of a Compliance Officer.

    Mirrors regional_ai.newsInsights but with Compliance persona & wording.

    Output columns:
        insight_id;timestamp;region;country;risk_theme;severity;insight_desc
    """

    supplier_text = supplier_name or "Not specified (Compliance scanning risk environment)"

    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    if news_df is None or news_df.empty:
        raise ValueError("news_df is empty – there is no news to generate insights from.")

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

    # Prepare CSV snippet for the model
    df_news = news_df[req_cols_news].copy()
    df_news["news_date"] = df_news["news_date"].astype(str)
    news_csv = df_news.to_csv(index=False, sep=";")

    # Dashboard / storyline context
    supplier_text = supplier_name or "Not specified (Compliance scanning risk environment)"

    filter_context = (
        f"Current dashboard region filter (Risk & Compliance Tab): {region or 'All regions'}. "
        f"Supplier Names passed from Sourcing Analyst for review: {supplier_text}. "
        "Compliance Officer is assessing Sustainability Risk, ESG, labour issues, "
        "regulatory changes and reputational risk that may impact these suppliers."
    )

    # -------- Prompt template for 5 Compliance Insights ----------
    prompt_template = """
You are a Business Code of Conduct & ESG Sustainability Compliance Senior Officer with over 
10 years of Professional Experience in the Supply Chain Management and Textile 
Manufacturing Business Spaces.

You are working under the employment of a Supply Chain Company that deals with 
the Sourcing of Raw Materials & Intermediate Goods relating to the Textile Manufacturing 
space across the globe.

Your counterparties within the organization include:

1. Sourcing Analyst: On a daily basis, they will send a list of Existing & Prospective 
Supplier Names via E-mail with a Service Request to assess the Suppliers' Business Profiles, 
Positive & Negative News relating to the Specified Supplier Names, and to provide Insights 
for each Specified Supplier Name.

2. Regional Sourcing Manager: On a daily basis, monitor their Portfolio of Existing & 
Prospective Supplier Names for the purpose of optimizing the Best Sourcing Strategies, 
Reduce Geographic Concentration Risks, and Mitigate Reputational Risks arising from 
Third Party Allegations that the Organization is believe to be supportive of the 
Unethical Suppliers' actions.

RISK & COMPLIANCE INSIGHTS GENERATION — INPUT / PROCESS / OUTPUT

1. Input: Dashboard Filter, Specific Supplier Details (1 row only), News Table (everything).
2. Process: Analyze all the input, use public internet as guidance on ESG and Code of Conduct rules.
3. Output: Specific Supplier Details, 5 lines of Insights (based on understanding of supplier details, news table, and public internet).

The Data Sources provided to you are as follow:

1. Business Profile of the Specified Supplier Name in CSV Format: 
{{current_supplier_profile}}

2. Latest News Feed from the Centralized Database in CSV Format: 
{{current_filtered_news}}

3. The Current Selected Dashboard Filter Options: 
{{current_filter_context}}

Your task is to write the Top 5 Insights based on what you have assessed and understood 
from the above-mentioned Data Sources (e.g. Corruption Scandal, Launched a Green Initiative).

For each Insight:
- Use the exact "supplier_name" as defined in the Data Sources.
- Use the Supplier's Region and Country from the Supplier Profile.
- Assign a "theme" from the following list of options:
  "Certification & Audit", "Business Code of Conduct", "Delivery Performance", 
  "Product Quality", "Workplace Safety & Health", 
  "ESG Environment", "ESG Social", "ESG Governance",
- Assign a "severity" rating: "Low", "Medium" or "High".
- Always group similar issues under the same Insight.
- Maximum Limit of 50 Words.

Exception Handling: If there insufficient information available from the above-mentioned 
Data Sources for writing meaningful insights, you may search the Public Internet for 
recent news and development relating to ESG Sustainability Compliance and 
Textile Manufacturing.

Requirement for the Output: Return a CSV file, using semi-colon (;) as delimiter.

Column Names & Rules: 
1. "insight_id" -- Incremental starting from 1.
2. "timestamp" -- Today's Date in YYYY.MM.DD format.
3. "supplier_id" -- Unique ID of the Specified Supplier.
4. "supplier_name" -- Name of the Specified Supplier.
5. "region" -- Region of the Specified Supplier.
6. "country" -- Country of the Specified Supplier.
7. "theme" -- One of the Most Suitable Theme Names as explained above.
8. "severity" -- Only one value: Low / Medium / High
9. "insight_desc" -- Text of the Generated Insight.

Do NOT return any blank, "N/A", "NA", or "NaN" values.
Do NOT return any greeting, explanation, note, or comment outside of the CSV table.
"""

    prompt = (
        prompt_template.replace("{{current_filter_context}}", filter_context)
        .replace("{{current_filtered_news}}", news_csv)
    )

    payload = {
        "model": BITDEER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30720,
        "temperature": 0.2,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {BITDEER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(BITDEER_API_URL, headers=headers, json=payload, timeout=60)
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
        "theme",
        "severity",
        "insight_desc",
    ]
    new_insights = _parse_model_csv(model_output, expected_cols)


    # Clean & handle blanks in description
    if "insight_desc" in new_insights.columns:
        new_insights["insight_desc"] = new_insights["insight_desc"].astype(str).str.strip()
        mask_blank = new_insights["insight_desc"].isin(["", "nan", "NaN", "N/A", "NA"])
        new_insights.loc[mask_blank, "insight_desc"] = (
            "Limited recent Sustainability Risk & Compliance news in this view; "
            "maintain routine ESG monitoring, supplier due diligence and "
            "incident reporting for this region/country."
        )

    # ---- Force supplier profile metadata onto every row ----
    # Timestamp fallback
    if "timestamp" not in new_insights.columns:
        new_insights["timestamp"] = datetime.utcnow().date().isoformat()

    # Supplier ID
    if supplier_id:
        new_insights["supplier_id"] = supplier_id
    elif "supplier_id" not in new_insights.columns:
        new_insights["supplier_id"] = ""

    # Supplier name
    if supplier_name:
        if len(supplier_name) == 1:
            sup_text = supplier_name[0]
        else:
            sup_text = ", ".join(sorted(set(supplier_name)))
        new_insights["supplier_name"] = sup_text
    elif "supplier_name" not in new_insights.columns:
        new_insights["supplier_name"] = ""

    # Region – prefer the supplier's region if provided
    if supplier_region:
        new_insights["region"] = supplier_region
    elif "region" not in new_insights.columns:
        new_insights["region"] = region or "All"

    # Country – ALWAYS lock to supplier profile if available
    if supplier_country:
        new_insights["country"] = supplier_country
    elif "country" not in new_insights.columns:
        new_insights["country"] = "Multiple"

    # Theme + severity defaults / normalization
    if "theme" not in new_insights.columns and "risk_theme" in new_insights.columns:
        new_insights.rename(columns={"risk_theme": "theme"}, inplace=True)
    if "theme" not in new_insights.columns:
        new_insights["theme"] = "Other ESG issue"
    if "severity" not in new_insights.columns:
        new_insights["severity"] = "Medium"


    # Append (or create) the Risk & Compliance Insights Data Table
    if save_path is not None:
        file_exists = os.path.exists(save_path)
        new_insights.to_csv(
            save_path,
            index=False,
            sep=";",
            mode="a" if file_exists else "w",
            header=not file_exists,
        )

    return new_insights


# -------------------------------------------------------------------
# 2) COMPLIANCE RECOMMENDATIONS (Corrective Actions)
# -------------------------------------------------------------------

def complianceRecommend(
    region: Optional[str],
    supplier_name: Optional[str] = None,
    insights_path: str = "compliance_insight.csv",
    save_path: str = "compliance_recommend.csv",
) -> pd.DataFrame:
    """
    Generate 5 Compliance RECOMMENDATIONS based on the latest batch of
    Sustainability Risk & Compliance insights.

    These recommendations follow the 2 compliance objectives:
    1) Insights for Supplier Business Performance & Sustainability Conduct.
    2) Corrective Actions for Suppliers + guidance for Regional Sourcing Manager.

    Output columns:
        recommend_id;timestamp;region;country;owner;priority;recommend_desc
    """

    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    if insights_path is None:
        raise ValueError("insights_path must not be None.")

    try:
        insights_df = pd.read_csv(insights_path, sep=";")
    except FileNotFoundError:
        raise ValueError(
            "Risk & Compliance insights file not found. Run complianceInsights() first."
        )

    if insights_df.empty:
        raise ValueError("Risk & Compliance insights table is empty.")

    if "timestamp" not in insights_df.columns:
        raise ValueError("Risk & Compliance insights table must contain 'timestamp' column.")

    # Focus on LATEST batch of insights
    insights_df["timestamp"] = pd.to_datetime(insights_df["timestamp"], errors="coerce")
    if insights_df["timestamp"].isna().all():
        raise ValueError("No valid timestamps found in Risk & Compliance insights table.")

    latest_ts = insights_df["timestamp"].max()
    latest_batch = insights_df[insights_df["timestamp"] == latest_ts].copy()

    insights_csv = latest_batch.to_csv(index=False, sep=";")

    supplier_text = supplier_name or "Not specified – general compliance recommendations."

    filter_context = (
        f"Current dashboard region filter (Risk & Compliance Tab): {region or 'All regions'}. "
        f"Supplier Names provided by Sourcing Analyst for review: {supplier_text}. "
        f"Latest Risk & Compliance Insights timestamp: {latest_ts.date()}."
    )

    # -------- Prompt template for 5 Compliance Recommendations ----------
    prompt_template = """
You are a Business Code of Conduct & ESG Sustainability Compliance Senior Officer with over 
10 years of Professional Experience in the Supply Chain Management and Textile 
Manufacturing Business Spaces.

You are working under the employment of a Supply Chain Company that deals with 
the Sourcing of Raw Materials & Intermediate Goods relating to the Textile Manufacturing 
space across the globe.

Your counterparties within the organization include:

1. Sourcing Analyst: On a daily basis, they will send a list of Existing & Prospective 
Supplier Names via E-mail with a Service Request to assess the Suppliers' Business Profiles, 
Positive & Negative News relating to the Specified Supplier Names, and to provide 
Recommendations for each Specified Supplier Name.

2. Regional Sourcing Manager: On a daily basis, monitor their Portfolio of Existing & 
Prospective Supplier Names for the purpose of optimizing the Best Sourcing Strategies, 
Reduce Geographic Concentration Risks, and Mitigate Reputational Risks arising from 
Third Party Allegations that the Organization is believe to be supportive of the 
Unethical Suppliers' actions.

RISK & COMPLIANCE RECOMMENDATIONS GENERATION — INPUT / PROCESS / OUTPUT

1. Input: Recently Generated Insights of the Specified Supplier Details (5 rows), Current Dashboard Filters.
2. Process: Analyze all the input, use public internet as guidance on ESG and Code of Conduct corrective actions.
3. Output: Specific Supplier Details, 5 lines of Recommendations (based on understanding of supplier details, generated insights, and public internet).

The Data Sources provided to you are as follow:

1. Latest Insights of the Specified Supplier in CSV Format: 
{{latest_supplier_insights}}

2. The Current Selected Dashboard Filter Options: 
{{current_filter_context}}

Your task is to write the Top 5 Recommendations based on what you have assessed and understood 
from the above-mentioned Data Sources (e.g. Enforce Zero Tolerance for Bribery, Switch to 
Renewable Energy Sources, Improve Workplace Conditions).

For each Recommendation:
- It must be unique and linked to an Insight from the above-mentioned Data Sources.
- Assign a "priority":
  "P1" = Critical (Act Now)
  "P2" = Urgent (Act Soon)
  "P3" = Important (Act When Situation Permits)
  "P4" = Good to Have
  "P5" = Nice to Have
- For the "recommend_desc" column, describe (max 50 words):
  - The corrective action required by the Specified Supplier.
  - How this action may help improve the Specified Supplier's Score.

Exception Handling: If there insufficient information available from the above-mentioned 
Data Sources for writing meaningful recommendations, you may search the Public Internet for 
recommendation corrective actions and rationale relating to ESG Sustainability Compliance and 
Textile Manufacturing.

Requirement for the Output: Return a CSV file, using semi-colon (;) as delimiter.

Column Names & Rules: 
1. "recommend_id" -- Incremental starting from 1.
2. "timestamp" -- Today's Date in YYYY.MM.DD format.
3. "supplier_id" -- Unique ID of the Specified Supplier.
4. "supplier_name" -- Name of the Specified Supplier.
5. "region" -- Region of the Specified Supplier.
6. "country" -- Country of the Specified Supplier.
7. "recommend_desc" -- Text of the Generated Recommendation.
8. "priority" -- Priority Rating of the Recommendation.

Do NOT return any blank, "N/A", "NA", or "NaN" values.
Do NOT return any greeting, explanation, note, or comment outside of the CSV table.
"""

    # IMPORTANT: correctly inject insights + filter context into the template
    prompt = (
        prompt_template.replace("{{latest_supplier_insights}}", insights_csv)
        .replace("{{current_filter_context}}", filter_context)
    )

    payload = {
        "model": BITDEER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30720,
        "temperature": 0.2,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {BITDEER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(BITDEER_API_URL, headers=headers, json=payload, timeout=60)
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
        "priority",
    ]
    new_recs = _parse_model_csv(model_output, expected_cols)


    # Clean blanks
    if "recommend_desc" in new_recs.columns:
        new_recs["recommend_desc"] = new_recs["recommend_desc"].astype(str).str.strip()
        mask_blank = new_recs["recommend_desc"].isin(["", "nan", "NaN", "N/A", "NA"])
        new_recs.loc[mask_blank, "recommend_desc"] = (
            "No specific corrective action identified from current data; maintain standard "
            "supplier monitoring, periodic ESG audits and escalation procedures."
        )

    # Safety backfill if model omitted fields
    if "timestamp" not in new_recs.columns:
        new_recs["timestamp"] = latest_ts.date().isoformat()
    if "region" not in new_recs.columns:
        new_recs["region"] = region or "All"
    if "country" not in new_recs.columns:
        new_recs["country"] = "Multiple"
    if "priority" not in new_recs.columns:
        new_recs["priority"] = "P3"
    if "supplier_id" not in new_recs.columns:
        new_recs["supplier_id"] = ""
    if "supplier_name" not in new_recs.columns:
        new_recs["supplier_name"] = ""


    # Append (or create) the Compliance Recommendations Data Table
    if save_path is not None:
        file_exists = os.path.exists(save_path)
        new_recs.to_csv(
            save_path,
            index=False,
            sep=";",
            mode="a" if file_exists else "w",
            header=not file_exists,
        )

    return new_recs
