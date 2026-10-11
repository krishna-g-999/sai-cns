# SAI-CNS web server.
#
# Pinned to python:3.13-slim because the deployed estimators were fitted under scikit-learn 1.8.0 on
# that line; see requirements.txt for why the pin matters. The image installs dependencies in a
# separate layer from the application so that a code change does not rebuild the scientific stack,
# which is the slow part.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libxrender and libxext are needed by RDKit's molecule drawing; without them the structure image
# fails at runtime rather than at build time, which is the worst place to discover it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libsm6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py api.py serve.py model_fetch.py models_manifest.json ./
COPY src/ ./src/
COPY assets/ ./assets/
# results/ and docs/ are not copied in: they hold validation tables and write-ups for the paper,
# not anything the running server needs. Every place app.py/api.py reads from either is already
# guarded (an .exists() check or a try/except returning an empty result), verified by grep across
# both files before this line was removed, so their absence here degrades a few optional UI
# sections rather than failing the build or a request.

# The models are fetched at BUILD time from the published archive rather than copied from the build
# context or downloaded at start-up. Three reasons, in order of importance:
#
#   a build context does not have them   .gitignore excludes the binaries, so a build from a clone or
#                                        from Cloud Build has the code and none of the science
#   cold starts must be fast             a serverless host destroys idle instances, and fetching
#                                        0.79 GB on every cold start would make the first request
#                                        after each idle period take minutes and repeat the transfer
#   a build that cannot get them fails   here, rather than producing an image that starts and then
#                                        cannot answer
#
# The download verifies the archive checksum and every extracted file, so a corrupted layer cannot
# reach a user. SAI_CNS_SKIP_MODEL_FETCH then stops the running server repeating the check.
#
# SKIP_MODEL_FETCH exists for continuous integration, which needs to know that the Dockerfile and
# the dependency set are sound but has no published archive to fetch from and no use for 0.84 GB of
# estimators. An image built this way starts and serves the interface but cannot answer a query, so
# it is for build verification only and must never be deployed.
ARG SKIP_MODEL_FETCH=0
RUN if [ "$SKIP_MODEL_FETCH" = "1" ]; then \
        echo "SKIP_MODEL_FETCH=1: models not fetched. Build verification image only."; \
    else \
        python model_fetch.py && rm -f .model_fetch.lock; \
    fi
ENV SAI_CNS_SKIP_MODEL_FETCH=1

# Run as an unprivileged user. The application only ever reads from disk.
#
# The user id is 1000 rather than an arbitrary high number because Hugging Face Spaces runs every
# container as uid 1000, and a mismatch produces permission errors on the model directory that
# surface only after a slow image build. Any other host is indifferent to the number.
RUN useradd --create-home --uid 1000 sai_cns && chown -R sai_cns:sai_cns /app
USER sai_cns

# 8501 serves the interface, 8000 the REST API. Both are started by serve.py in one container
# because they share a single copy of the loaded models; running them as separate containers would
# double the memory for no benefit, the models being read-only.
EXPOSE 8501 8000

# Streamlit's own health endpoint, so an orchestrator restarts a wedged container rather than
# leaving it serving errors. The API's health endpoint is checked too, since the interface can be
# healthy while the API thread has died.
HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health && curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "serve.py"]
