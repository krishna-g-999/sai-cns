# Published container image for SAI-CNS. Built by .github/workflows/build-container.yml and
# pushed to ghcr.io, from where Azure Container Apps pulls it.
#
# Build context is the REPOSITORY ROOT, not this directory:
#     docker build -f deploy/container/Dockerfile .
#
# This exists separately from the repository's own Dockerfile because the two answer different
# questions. The root Dockerfile starts serve.py, which runs the interface and the REST API
# together on two ports. A Container App routes external traffic to exactly one port, so the
# second would be unreachable and the memory spent on it wasted. This image serves the interface.
FROM python:3.13-slim

# RDKit links against these at import time. Without them the failure is a bare
# "libXrender.so.1: cannot open shared object file" at run time, which reads like a broken
# install rather than a missing system package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libsm6 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a change to the application does not rebuild the scientific stack.
COPY deploy/container/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app.py api.py model_fetch.py models_manifest.json ./
COPY .streamlit ./.streamlit
COPY assets ./assets
COPY src ./src

# The six tables and one fingerprint the interface reads. They are small and they are not
# optional in practice: every read of them is guarded, and the guard substitutes a placeholder
# rather than failing, so without them the landing page quietly advertises "- molecular targets"
# instead of 54 and /version reports an empty knowledge-graph fingerprint. A deployment that is
# wrong in a way nobody notices is worse than one that refuses to start.
COPY results/tables/MODEL_INVENTORY.csv results/tables/inversion_validation.csv \
     results/tables/model_comparison.csv results/tables/noncns_specificity_summary.csv \
     results/tables/rf_cv_summary.csv results/tables/rf_temporal.csv ./results/tables/
COPY inversion/results/GRAPH_FINGERPRINT.json ./inversion/results/

# The models, 0.87 GB, are fetched at BUILD time from the published archive rather than copied
# from the build context, which does not have them, or downloaded at start-up, which would make
# every cold start pay the transfer. model_fetch.py verifies the archive checksum and then all
# 252 extracted files before the build is allowed to proceed, so a truncated download fails here
# rather than becoming predictions that are quietly wrong.
RUN python model_fetch.py && rm -f .model_fetch.lock

# Models are present and verified; stop the running server repeating the 252-file check.
ENV SAI_CNS_SKIP_MODEL_FETCH=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 1000 saicns && chown -R saicns:saicns /app
USER saicns

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8000", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
