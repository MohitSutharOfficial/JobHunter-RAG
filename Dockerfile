FROM python:3.11-slim

# Non-root user (uid 1000) required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser
ENV PYTHONPATH=/app/src \
    HOME=/home/appuser
EXPOSE 8000

CMD ["uvicorn", "job_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
