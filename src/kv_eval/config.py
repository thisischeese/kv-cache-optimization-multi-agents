"""Fixed configuration for this evaluation run.

Targets and domain are fixed by the project brief, not user input.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "report.md"

TECH_IDS: tuple[str, str] = ("kivi", "infinigen")
DOMAIN_ID = "cloud_llm_serving"
PERSPECTIVES: tuple[str, ...] = ("trl", "market", "stakeholder", "domain")

# TODO: load LLM/model settings from .env once real agents replace the mocks.
