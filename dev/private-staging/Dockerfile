# Local-only staging image. Train the real model using public demonstration
# data at BUILD TIME; runtime stores only aggregate synthetic run metadata.
# Never deploy this experimental OIDC service to public Render.
FROM python:3.11-slim AS trainer
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY scripts/download_data.py scripts/train_model.py ./scripts/
RUN python scripts/download_data.py && python scripts/train_model.py && \
    test -s models/best_model.joblib

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.serve.txt requirements.saas.txt ./
RUN pip install --no-cache-dir -r requirements.serve.txt -r requirements.saas.txt \
    && useradd --create-home --uid 10001 churn
COPY --from=trainer /app/models/best_model.joblib ./models/best_model.joblib
COPY src/ ./src/
RUN chown -R churn:churn /app
USER churn
EXPOSE 8000
CMD ["uvicorn", "churn_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
