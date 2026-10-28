"""Prompt templates for the three conditions.

The RAG template reproduces the paper's Appendix A.2 (context-constrained,
PMID-cited, explicit fallback). The taxonomy variant additionally exposes each
source's Domain/Subdomain/Topic so the model can organize the answer clinically.
The "alone" prompt gives GPT-4 the question with no retrieved context.
"""

ALONE_SYSTEM = (
    "You are a biomedical research assistant answering cardiovascular research "
    "questions. Answer accurately and concisely in a formal scientific tone. If "
    "you are uncertain, say so rather than fabricating specifics."
)

RAG_SYSTEM = (
    "You are a biomedical research assistant tasked with answering user queries "
    "using retrieved scientific evidence. Use ONLY the provided context.\n"
    "- Every factual statement must be directly supported by the context.\n"
    "- Include a PMID citation (from the context) immediately after each statement; "
    "use only PMIDs present in the context.\n"
    "- If the answer is not in the context, state exactly: "
    "\"The information is not available in the provided context.\"\n"
    "- Separate multiple PMIDs with semicolons. Maintain a formal scientific tone."
)

RAG_TAXONOMY_SYSTEM = RAG_SYSTEM + (
    "\n- Each source is annotated with a clinical taxonomy path "
    "(Domain > Subdomain > Topic). Use it to group related evidence and to keep "
    "the answer within the clinically relevant category; do not cite the taxonomy itself."
)


def format_context(docs, include_taxonomy=False):
    blocks = []
    for d in docs:
        head = f"[PMID: {d['pmid']}] {d['title']} ({d.get('year', 'n/a')}, {d.get('journal', '')})"
        if include_taxonomy and d.get("taxonomy"):
            t = d["taxonomy"]
            head += f"\nTaxonomy: {t['domain']} > {t['subdomain']} > {t['topic']}"
        blocks.append(f"{head}\nAbstract: {d['abstract']}")
    return "\n\n".join(blocks)


def rag_user_prompt(question, context):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
