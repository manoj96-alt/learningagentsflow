"""
config.py — Central configuration for the 6-agent YouTube Content System.
"""

import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str  = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str          = "claude-sonnet-4-20250514"
LLM_TEMPERATURE: float  = 0.7

NOTION_TOKEN: str       = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID: str = os.getenv("NOTION_DATABASE_ID", "")

VIDEOS_PER_WEEK: int    = 2
PUBLISH_DAYS: list[str] = ["Tuesday", "Friday"]

SUPPORTED_DOMAINS: list[str] = ["AI", "Data Governance", "SAP MDG"]
OUTPUT_DIR: str         = "output"
QA_MAX_RETRIES: int     = 2
QA_PASS_THRESHOLD: float = 0.75   # minimum score on all 4 QA dimensions
