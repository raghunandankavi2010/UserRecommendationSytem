FROM python:3.12-slim
WORKDIR /srv
# CPU-only torch first, so sentence-transformers does not pull the CUDA build (~3 GB smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY ml/data/taxonomy.json ml/data/taxonomy.json
USER nobody
EXPOSE 8000
CMD ["uvicorn", "matcher.main:app", "--host", "0.0.0.0", "--port", "8000"]
