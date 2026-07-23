FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/serve.py ./scripts/serve.py
COPY outputs/deforestation_pipeline.pkl ./outputs/deforestation_pipeline.pkl

EXPOSE 8000

CMD ["uvicorn", "scripts.serve:app", "--host", "0.0.0.0", "--port", "8000"]
