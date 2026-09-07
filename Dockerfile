FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first so that a source change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Detector dependencies are opt-in: they add ~2.5 GB and the registry runs
# without them. Build with --build-arg WITH_DETECTORS=1 for the full suite.
ARG WITH_DETECTORS=0
COPY requirements-detectors.txt .
RUN if [ "$WITH_DETECTORS" = "1" ]; then \
      pip install --no-cache-dir -r requirements-detectors.txt && \
      python -m spacy download en_core_web_lg ; \
    fi

COPY app/ ./app/
COPY detectors/ ./detectors/
COPY llm_judge/ ./llm_judge/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
