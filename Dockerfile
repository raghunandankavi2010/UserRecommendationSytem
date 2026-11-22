FROM python:3.12-slim
WORKDIR /srv
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
USER nobody
EXPOSE 8000
CMD ["uvicorn", "matcher.main:app", "--host", "0.0.0.0", "--port", "8000"]
