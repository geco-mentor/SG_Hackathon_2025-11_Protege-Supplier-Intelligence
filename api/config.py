from dotenv import load_dotenv
import os

# Load environment variables from .env in the project root
load_dotenv()

# ---------- API KEYS ----------
BITDEER_API_KEY = os.getenv("BITDEER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UN_COMTRADE_TOKEN = os.getenv("UN_COMTRADE_TOKEN")

# ---------- BITDEER CHAT (OpenAI-compatible) ----------
# Endpoint + model taken from your original app.py.
# If Bitdeer console shows a different URL or model name, override in .env.
BITDEER_API_URL = os.getenv(
    "BITDEER_API_URL",
    "https://api-inference.bitdeer.ai/v1/chat/completions",
)
BITDEER_MODEL = os.getenv(
    "BITDEER_MODEL",
    "openai/gpt-oss-120b",
)

# ---------- UN COMTRADE ----------
COMTRADE_BASE_URL = os.getenv(
    "COMTRADE_BASE_URL",
    "https://comtradeapi.un.org/data/v1/get/C/A/HS"
)

# ---------- CENTRAL DB (CSV) ----------
TRADE_FLOW_CSV = os.getenv("TRADE_FLOW_CSV", "trade_flow.csv")
