"""Provider-agnostic chat wrapper: OpenAI or Azure OpenAI, chosen from env.

The paper generated answers with an Azure-hosted GPT-4.1 deployment and used the
OpenAI API for labeling/auditing; both are supported here so the same code runs
in either setting. With no credentials, callers should use dry-run mode.
"""
from __future__ import annotations
import config

_CLIENT = None
_IS_AZURE = None
_JUDGE_CLIENT = None
_JUDGE_IS_AZURE = None
LAST_USAGE = None   # token usage of the most recent chat() call, for cost/latency metrics


def get_judge_client():
    """Client for the judge. Separate Azure resource if AZURE_OPENAI_JUDGE_ENDPOINT
    + _API_KEY are set; otherwise reuse the main client (different deployment)."""
    global _JUDGE_CLIENT, _JUDGE_IS_AZURE
    if _JUDGE_CLIENT is not None:
        return _JUDGE_CLIENT, _JUDGE_IS_AZURE
    if config.AZURE_JUDGE_ENDPOINT and config.AZURE_JUDGE_API_KEY:
        from openai import AzureOpenAI
        from urllib.parse import urlparse
        ep = config.AZURE_JUDGE_ENDPOINT
        if "/openai" in ep:                     # user pasted the full target URI; keep base only
            u = urlparse(ep)
            ep = f"{u.scheme}://{u.netloc}/"
        _JUDGE_CLIENT = AzureOpenAI(azure_endpoint=ep, api_key=config.AZURE_JUDGE_API_KEY,
                                    api_version=config.AZURE_JUDGE_API_VERSION)
        _JUDGE_IS_AZURE = True
    else:
        _JUDGE_CLIENT, _JUDGE_IS_AZURE = get_client()
    return _JUDGE_CLIENT, _JUDGE_IS_AZURE


def get_client():
    global _CLIENT, _IS_AZURE
    if _CLIENT is not None:
        return _CLIENT, _IS_AZURE
    backend = config.llm_backend()
    if backend == "azure":
        from openai import AzureOpenAI
        _CLIENT = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        _IS_AZURE = True
    elif backend == "openai":
        from openai import OpenAI
        _CLIENT = OpenAI(api_key=config.OPENAI_API_KEY)
        _IS_AZURE = False
    else:
        raise RuntimeError(
            "No LLM credentials. Set OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT+"
            "AZURE_OPENAI_API_KEY, or run with --dry-run."
        )
    return _CLIENT, _IS_AZURE


def chat(system, user, temperature=0.0, max_tokens=800, model=None, judge=False):
    """Single-turn completion. Returns the assistant text.

    judge=True routes to the (possibly separate) judge client + judge deployment.
    model: optional explicit deployment override.
    Robust to reasoning models (o-series / GPT-5+ / gpt-6.x), which reject a custom
    `temperature` and require `max_completion_tokens` instead of `max_tokens`.
    """
    client, is_azure = get_judge_client() if judge else get_client()
    if model is None:
        if judge:
            model = config.AZURE_JUDGE_DEPLOYMENT if is_azure else config.JUDGE_MODEL
        else:
            model = config.AZURE_CHAT_DEPLOYMENT if is_azure else config.CHAT_MODEL
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def _call(**kw):
        return client.chat.completions.create(model=model, messages=messages, **kw)

    # Negotiate params: reasoning models (o-series / GPT-5+ / gpt-6.x) reject a
    # custom temperature and want max_completion_tokens instead of max_tokens.
    kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    for _ in range(4):
        try:
            resp = _call(**kwargs)
            break
        except Exception as e:
            m = str(e).lower()
            changed = False
            if "temperature" in m and "temperature" in kwargs:
                kwargs.pop("temperature"); changed = True
            if ("max_tokens" in m or "max_completion_tokens" in m) and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens"); changed = True
            if not changed:
                raise
    else:
        raise RuntimeError("chat parameter negotiation failed")
    global LAST_USAGE
    u = getattr(resp, "usage", None)
    LAST_USAGE = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                  "completion_tokens": getattr(u, "completion_tokens", None),
                  "total_tokens": getattr(u, "total_tokens", None)} if u else None
    return resp.choices[0].message.content
