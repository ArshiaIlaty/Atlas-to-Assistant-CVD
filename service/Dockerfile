# Lean image for the three-condition API, Azure-backed (AI Search + Azure OpenAI).
# Local SBERT backend is NOT included here (that path needs torch + the corpus CSV);
# set SERVICE_BACKEND=azure and provide the Azure env vars at runtime.
FROM python:3.12-slim
WORKDIR /app

COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments ./experiments
COPY service ./service

ENV SERVICE_BACKEND=azure
EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
