# api/supplier_scoring.py

import io
import pandas as pd
import requests

from config import BITDEER_API_KEY, BITDEER_API_URL, BITDEER_MODEL # import from config.py



def supplierScore(
    suppliers_df: pd.DataFrame) -> pd.DataFrame:


    save_path = './data/supplier_master.csv'  # <-- IMPORTANT, FILL UP LOCATION OF SAVE PATH HERE IF YOU WANT TO SAVE THE UPDATED DATAFRAME TO CSV, OTHERWISE LEAVE AS NONE

    """
    Use GPT-OSS (Bitdeer OpenAI-compatible API) to compute supplier_score and
    risk_rating for the first `max_rows` suppliers in the DataFrame.

    - Takes the current filtered suppliers_df from the dashboard
    - Converts the first `max_rows` rows to semi-colon CSV
    - Injects that CSV into the Supplier Score Determination prompt (inline)
    - Calls the OSS model
    - Parses the returned CSV (with ';' delimiter)
    - Copies 'supplier_score' and 'risk_rating' back into suppliers_df
    - Optionally writes the updated DataFrame to CSV (save_path)

    Parameters
    ----------
    suppliers_df : pd.DataFrame
        Filtered list of suppliers shown on the dashboard.

    Returns
    -------
    pd.DataFrame
        suppliers_df with supplier_score and risk_rating filled 
    """
    if suppliers_df is None or suppliers_df.empty:
        raise ValueError("suppliers_df is empty – there are no suppliers to score.")

    if not BITDEER_API_KEY:
        raise RuntimeError("BITDEER_API_KEY is not set in your .env file.")

    # ---- 1. Take first N suppliers and convert to CSV (semi-colon) ----
    subset_df = suppliers_df.head(7).copy()
    suppliers_csv = subset_df.to_csv(index=False, sep=";")

    # ---- 2. Inline prompt template (from Supplier Score Determination) ----
    prompt_template = """You are a Supply Chain Sustainability & Compliance Analyst worki...cs such as Service Reliability, and ISO Certification Standards.

List of Suppliers to be Assessed:
{{current_filtered_suppliers_dataframe}}

With reference to the above-mentioned List of Suppliers, your ta...Score" and "Risk Rating" (High / Medium / Low) of each Supplier.

ALL of the following Performance Metrics from the List of Suppli...ale each existing value to fit the new maximum value per column:
1. "certified_audited" - New Maximum Value: 30
2. "code_conduct" - New Maximum Value: 10
3. "timely_delivery" - New Maximum Value: 10
4. "product_quality" - New Maximum Value: 10
5. "service_reliability" - New Maximum Value: 10
6. "esg_environment" - New Maximum Value: 12
7. "esg_social" - New Maximum Value: 12
8. "esg_governance" - New Maximum Value: 6

Once the scaling is done, sum up all 8 performance metrics into ...the "supplier_score" column of each respective line of Supplier.

Rules for determination of Risk Rating based on Supplier Score:
"Low" = 75 to 100
"Medium" = 50 to 74
"High" = 0 to 49

Refer to the "supplier_score" column, and insert the assessed Ri...Rating under the "risk_rating" column for each line of Supplier.

Return the updated List of Suppliers in CSV Format, Delimiter to be semi-colon (;)."""

    # Inject the CSV into the placeholder
    prompt = prompt_template.replace(
        "{{current_filtered_suppliers_dataframe}}",
        suppliers_csv,
    )

    # ---- 3. Call GPT-OSS via Bitdeer API (OpenAI chat format) ----
    payload = {
        "model": BITDEER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
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

    # ---- 4. Parse the model's CSV output (semi-colon delimiter) ----
    buffer = io.StringIO(model_output)
    updated_subset_df = pd.read_csv(buffer, sep=";")

    # ---- 5. Copy supplier_score & risk_rating back into original df ----
    for col in ("supplier_score", "risk_rating"):
        if col in updated_subset_df.columns:
            suppliers_df.loc[subset_df.index, col] = updated_subset_df[col].values
        else:
            # If the model forgot the column, we just skip silently
            pass

    # ---- 6. Optional file write (for the other app.py Refresh Data button) ----
    if save_path is not None:
        suppliers_df.to_csv(save_path, index=False, sep=";")

    return suppliers_df

# supplierScore(pd.read_csv('supplier_master.csv', sep=';'))  # <-- for quick local testing only