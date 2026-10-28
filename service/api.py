"""FastAPI service exposing the three conditions as a switchable API + a small UI.

    SERVICE_BACKEND=local  (default)  uvicorn service.api:app --host 0.0.0.0 --port 8000
    SERVICE_BACKEND=azure  (once Azure AI Search is reachable)

Endpoints
    GET  /            -> web UI
    GET  /health      -> backend + readiness
    POST /ask         -> {question, condition, k, retrieval_mode} -> one condition
    POST /compare     -> {question, k} -> all three conditions side by side
"""
from __future__ import annotations
import os, sys

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
sys.path.insert(0, EXP)

BACKEND = os.getenv("SERVICE_BACKEND", "local")
if BACKEND == "local":
    # local retriever uses cached SBERT vectors (fast, free); generation stays on Azure
    os.environ.setdefault("EMBED_BACKEND", "local")
    os.environ["EMBED_BACKEND"] = "local"

import config, conditions  # noqa: E402
from fastapi import FastAPI, Header, HTTPException, Depends  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="CVD RAG — three-condition assistant")
STATE = {"retriever": None, "tagger": None}

# Optional shared-secret auth: if SERVICE_API_KEY is set, POST calls must send a
# matching `x-api-key` header. Leave unset for keyless local use.
_API_KEY = os.getenv("SERVICE_API_KEY")


def auth(x_api_key: str = Header(default=None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


def _ensure_retriever():
    if STATE["retriever"] is not None:
        return
    if BACKEND == "azure":
        import azure_search
        STATE["retriever"] = azure_search.AzureSearchRetriever()
    else:
        import corpus, retrieval, taxonomy_tagger
        docs = corpus.load_corpus()
        STATE["retriever"] = retrieval.HybridRetriever(docs, cache_key="corpusall")
        STATE["tagger"] = taxonomy_tagger.TaxonomyTagger()


class Ask(BaseModel):
    question: str
    condition: str = "rag"            # alone | rag | rag_taxonomy
    k: int = 5
    retrieval_mode: str = "hybrid"    # hybrid | dense | bm25


class Compare(BaseModel):
    question: str
    k: int = 5


@app.get("/health")
def health():
    return {"backend": BACKEND, "llm_backend": config.llm_backend(),
            "retriever_ready": STATE["retriever"] is not None,
            "conditions": conditions.CONDITIONS}


@app.post("/ask", dependencies=[Depends(auth)])
def ask(req: Ask):
    if req.condition != "alone":
        _ensure_retriever()
    r = conditions.run_condition(req.question, req.condition,
                                 retriever=STATE["retriever"], tagger=STATE["tagger"],
                                 k=req.k, retrieval_mode=req.retrieval_mode)
    return r


# --- The three explicit condition APIs (what the advisor asked for) ---
# One switchable endpoint (/ask with condition=...) plus three fixed routes, so
# the conditions can be called by distinct URLs and swapped by Hossein's work.
class QReq(BaseModel):
    question: str
    k: int = 5
    retrieval_mode: str = "hybrid"


def _run(question, condition, k, mode):
    if condition != "alone":
        _ensure_retriever()
    return conditions.run_condition(question, condition, retriever=STATE["retriever"],
                                    tagger=STATE["tagger"], k=k, retrieval_mode=mode)


@app.post("/ask/alone", dependencies=[Depends(auth)])
def ask_alone(req: QReq):
    """Condition 1: GPT-4.1 with no retrieved context."""
    return _run(req.question, "alone", req.k, req.retrieval_mode)


@app.post("/ask/rag", dependencies=[Depends(auth)])
def ask_rag(req: QReq):
    """Condition 2: GPT-4.1 + hybrid PubMed retrieval (PMID-cited)."""
    return _run(req.question, "rag", req.k, req.retrieval_mode)


@app.post("/ask/rag_taxonomy", dependencies=[Depends(auth)])
def ask_rag_taxonomy(req: QReq):
    """Condition 3: GPT-4.1 + retrieval + taxonomy metadata."""
    return _run(req.question, "rag_taxonomy", req.k, req.retrieval_mode)


@app.post("/compare", dependencies=[Depends(auth)])
def compare(req: Compare):
    _ensure_retriever()
    out = []
    for c in conditions.CONDITIONS:
        out.append(conditions.run_condition(req.question, c,
                   retriever=STATE["retriever"], tagger=STATE["tagger"], k=req.k))
    return {"question": req.question, "results": out}


@app.get("/")
def ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
