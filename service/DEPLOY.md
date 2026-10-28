# Sharing the three-condition API with a colleague

The API (`service/api.py`) exposes the three conditions:
`POST /ask/alone`, `/ask/rag`, `/ask/rag_taxonomy`, plus switchable `POST /ask`
and `POST /compare`, and a web UI at `/`. Pick one way to share it.

## Auth (do this before exposing it)

Set a shared secret so only people with the key can spend your Azure OpenAI quota:

```bash
export SERVICE_API_KEY="pick-a-long-random-string"
```

Then callers send header `x-api-key: <that string>`; the web UI picks it up from
`?key=<that string>` in the URL. If unset, the API is open (fine for localhost only).

---

## Option A — Quick share from this EC2 box (fastest)

1. Open inbound TCP **8000** to Hossein's IP in the EC2 **security group** (not 0.0.0.0/0).
2. Run it (note `0.0.0.0` so it's reachable):
   ```bash
   SERVICE_BACKEND=azure SERVICE_API_KEY=... \
     uvicorn service.api:app --host 0.0.0.0 --port 8000
   ```
3. Share: `http://<ec2-public-ip>:8000/?key=...`  (UI) or the `/ask/*` endpoints for code.
   Pros: 2 minutes. Cons: tied to this box being up; remember to stop it after.

## Option B — Azure Container Apps (recommended, durable, same tenant as the data)

From Azure Cloud Shell (bash) at the repo root:

```bash
RG=cvd-rag-rg; LOC=eastus2; APP=cvd-rag-api
az containerapp up \
  --name $APP --resource-group $RG --location $LOC \
  --source . --ingress external --target-port 8000 \
  --env-vars SERVICE_BACKEND=azure \
    AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
    AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1 \
    API_EMBED_MODEL=text-embedding-3-large EMBED_BACKEND=azure \
    AZURE_SEARCH_ENDPOINT=$AZURE_SEARCH_ENDPOINT \
    AZURE_SEARCH_KEY=$AZURE_SEARCH_KEY AZURE_SEARCH_INDEX=pubmed-cvd \
    SERVICE_API_KEY=$SERVICE_API_KEY
```

`az containerapp up` builds the `service/Dockerfile`, pushes it, and returns a public
HTTPS URL (give Hossein `https://<url>/?key=...`). Store secrets as Container App
secrets rather than plain env for anything long-lived. (Also allow the Container App's
outbound IP on the AI Search firewall, or set Search public access to All networks.)

## Option C — Hand to Dr. Shirazi the repo

He clones it, creates his own `.env` (Azure OpenAI + Search keys), then
`pip install -r service/requirements.txt && SERVICE_BACKEND=azure uvicorn service.api:app`.
Best when he wants to run/modify it in his own environment for his broader work.

---

Recommendation: **Option A today** for a quick look; **Option B** once he wants a
stable URL to build against. Both keep generation on your Azure GPT-4.1 and
retrieval on the Azure AI Search index.
