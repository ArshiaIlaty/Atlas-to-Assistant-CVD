"""Central configuration and paths for the revision experiments.

Everything is env-driven so the same code runs locally (free, offline retrieval
+ optional API generation) or against Azure once credentials are provisioned.
See experiments/README.md for the full list of environment variables.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.lower().startswith("export "):
                k = k[7:].strip()
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(REPO, ".env"))

# --- Data ---
CORPUS_CSV = os.getenv("CORPUS_CSV", os.path.join(REPO, "ETL", "pubmed_dataset.csv"))
TAXONOMY_CSV = os.getenv("TAXONOMY_CSV", os.path.join(REPO, "Taxonomy", "API_Inconsistency_o1.csv"))
BENCHMARK_XLSX = os.path.join(REPO, "Chatbot-Benchmark-Questions", "Q_A_evaluations.xlsx")
CACHE_DIR = os.path.join(REPO, "experiments", ".cache")
RESULTS_DIR = os.path.join(REPO, "experiments", "results")

# --- Embeddings (dense retrieval + taxonomy tagging) ---
# Local SBERT by default (no API key). The paper used all-MiniLM-L6-v2 for
# clustering and text-embedding-3-large for the Azure index; swap via EMBED_MODEL
# and EMBED_BACKEND when Azure OpenAI is available.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")        # local | openai | azure
# Model name is resolved per backend so flipping EMBED_BACKEND never leaves a
# mismatched model (e.g. asking local SBERT to load "text-embedding-3-large").
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
API_EMBED_MODEL = os.getenv("API_EMBED_MODEL", "text-embedding-3-large")
EMBED_MODEL = LOCAL_EMBED_MODEL if EMBED_BACKEND == "local" else API_EMBED_MODEL

# --- Generation LLM ---
# Backend auto-detected: azure if AZURE_OPENAI_ENDPOINT set, else openai if
# OPENAI_API_KEY set, else none (dry-run only).
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")      # OpenAI model id
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM-as-judge ---
# Default to the generation model, but STRONGLY prefer a DIFFERENT model family
# as judge (reduces self-preference bias, a known LLM-judge failure mode) and
# always treat the judge as a complement to the blinded human raters, not a
# replacement (reviewers R1#1/R1#7/R2#1).
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", CHAT_MODEL)
AZURE_JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", AZURE_CHAT_DEPLOYMENT)
# The judge model may live in a DIFFERENT Azure resource/region (newest models are
# region-limited). If a judge endpoint+key are set, the judge uses its own client;
# otherwise it reuses the main Azure OpenAI client with the judge deployment name.
AZURE_JUDGE_ENDPOINT = os.getenv("AZURE_OPENAI_JUDGE_ENDPOINT")
AZURE_JUDGE_API_KEY = os.getenv("AZURE_OPENAI_JUDGE_API_KEY")
AZURE_JUDGE_API_VERSION = os.getenv("AZURE_OPENAI_JUDGE_API_VERSION", AZURE_OPENAI_API_VERSION)

# --- Azure AI Search (production retrieval backend; optional) ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "pubmed-cvd")

# --- Retrieval defaults ---
TOP_K = int(os.getenv("TOP_K", "5"))
RRF_K0 = int(os.getenv("RRF_K0", "60"))

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def llm_backend():
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        return "azure"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


def judge_model():
    return AZURE_JUDGE_DEPLOYMENT if llm_backend() == "azure" else JUDGE_MODEL
