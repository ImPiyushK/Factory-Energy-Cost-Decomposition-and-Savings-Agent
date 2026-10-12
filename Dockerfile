FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src ./src
COPY evaluation ./evaluation
COPY tests ./tests
COPY scripts ./scripts
COPY data_card.json* README.md* ./

# judges may mount ./data; create the DB at startup automatically
ENV DB_PATH=/app/data/energy_agent.db \
    DATA_DIR=/app/data

EXPOSE 8000 8501

# default: API. docker-compose overrides the command for the UI service.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
