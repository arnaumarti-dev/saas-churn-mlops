# Isolated localhost-only staging API. No public Render deployment.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.serve.txt requirements.saas.txt ./
RUN pip install --no-cache-dir -r requirements.serve.txt -r requirements.saas.txt \
    && useradd --create-home --uid 10001 churn
COPY src/ ./src/
RUN chown -R churn:churn /app
USER churn
EXPOSE 8000
CMD ["uvicorn", "churn_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
