# Deploying SAI-CNS

The server is a single container running two things that share one copy of the loaded models: the
Streamlit interface on `$PORT` (default 8501) and the REST API on `$API_PORT` (default 8000).

Everything below is prepared and tested locally. The one step that cannot be done from here is
creating an account with a host and pressing deploy, because that needs your credentials. Each option
is written so that it is a single command or a single form.

## What must be true before you deploy

```bash
brainsafe_env/Scripts/python.exe src/sai_cns/evaluation/app_health.py
```

Twenty checks, exit code 0 required. It verifies that every declared dependency resolves at its
pinned version, that all model artefacts load, that the knowledge graph is internally consistent,
that chemically unrelated compounds give distinct and directionally correct profiles, that every
export format is well formed and self-contained, that the coverage panel matches the deployed panel,
and that the API answers. Do not deploy a build that fails it.

## Option 1: Hugging Face Spaces (recommended, free, no card)

Best fit for a journal submission: free, no payment method, gives a permanent public URL, and the
model files are handled by git-lfs rather than needing separate hosting.

1. Create a Space at https://huggingface.co/new-space, SDK **Docker**, visibility **Public**.
2. Add the Space as a remote and push:

```bash
git remote add space https://huggingface.co/spaces/<your-user>/sai-cns
git push space main
```

The `Dockerfile` is used as-is. Spaces sets `$PORT`, which `serve.py` reads. The URL becomes
`https://<your-user>-sai-cns.hf.space` and goes in the manuscript.

### The model files, which are the real obstacle

`.gitignore` excludes `*.joblib`, so the repository tracks 64 metadata files and **no model
binaries**. A clone cannot run the server. This is a larger problem than not having a URL and must be
solved before any deployment succeeds.

The binaries are now 0.78 GB, down from 1.8 GB: `src/sai_cns/models/compress_models.py`
recompressed every model and verified per model that predictions are unchanged, checking the round
trip against each estimator's own run-to-run variation rather than demanding bit equality, because
these forests are not bit-deterministic between calls.

Two ways to get them to the host, in order of preference:

1. **Track them with git-lfs on the Space.** 0.78 GB fits comfortably within a Hugging Face Space.
   Do this in the Space clone, not in the GitHub repository, whose free LFS quota is 1 GB total:

   ```bash
   git lfs install
   git lfs track "models_rf/**/*.joblib" "models_rf/*.pkl"
   git add .gitattributes models_rf -f      # -f overrides .gitignore
   git commit -m "Add model binaries via LFS for deployment"
   git push space main
   ```

2. **Publish the archive and let the server fetch it.** This is now built and tested, and it is
   the route that makes the server deployable anywhere rather than only on a host that gives you a
   gigabyte of git storage. It also gives the manuscript a citable model archive.

   ```bash
   python src/sai_cns/models/package_models.py 1.0
   ```

   That writes `dist/sai_cns_models_v1.0.tar.gz` (0.79 GB) and `models_manifest.json`, which
   records the archive's SHA-256 and the size and checksum of all 195 files inside it. The manifest
   is committed; the archive is not.

   Upload the archive to Zenodo, then put its direct download URL in `models_manifest.json` under
   `"urls"` and the DOI under `"doi"`, and commit that one small file.

   From then on, any deployment that lacks the binaries downloads them on first start, verifies the
   archive checksum, extracts, and verifies every extracted file before a single model is loaded. A
   truncated transfer or a substituted archive is refused rather than quietly serving predictions
   that are wrong. A tree that already has the models does nothing: the check costs about a second.

   Tested end to end: fetch into an empty tree restored and verified all 195 files; a second run was
   a 1.3-second no-op; corrupting one file caused it to be detected and repaired.

   `SAI_CNS_SKIP_MODEL_FETCH=1` disables it for an image that bakes the models in.

## Option 2: Streamlit Community Cloud (free, simplest, interface only)

Fastest route to a URL, but it runs `app.py` directly and does **not** run the container, so the REST
API is not served. Acceptable if you only need the interface reviewed.

1. https://share.streamlit.io, connect the GitHub repository.
2. Main file `app.py`, Python 3.13, `requirements.txt` is picked up automatically.
3. **Set the branch to `main`, not `master`.** `master` is the repository default and holds the
   superseded v3/v5 engine, so a deployment left on the default serves the wrong tool. This is not a
   caching problem and rebooting will not fix it.
4. The models are fetched from the archive on first start, so this route needs `models_manifest.json`
   to carry a working URL. Without it the app starts and then reports that it cannot obtain the
   models, which is the correct behaviour but is not a working server.

## Option 3: any container host (Render, Railway, Fly.io, Cloud Run)

```bash
docker build -t sai-cns .
docker run -p 8501:8501 -p 8000:8000 sai-cns
```

Then push the image to the host's registry. All four read `$PORT`. Give the instance **at least 4 GB
of memory**. Measured on a warm server, resident set size is 2.35 GB: 164 MB for the interpreter and
imports, 2,059 MB for the 69 deployed estimators, 128 MB for the applicability-domain index, and
about 1 MB of working memory per query. A 2 GB instance will be killed during model loading, and a
3 GB instance leaves no headroom for concurrent sessions.

This figure is the single most important constraint on where the server can run. It rules out every
free tier built for small apps: Cloudflare Workers caps an isolate at 128 MB, Render's free web
service at 512 MB, and Streamlit Community Cloud at 2.7 GB, which is below the point at which
loading completes with room to serve. Hosts that fit are Hugging Face Spaces on CPU basic (16 GB),
Cloud Run or any container host configured with 4 GB, and Cloudflare Containers at `standard-1`
(4 GiB) or larger. Note that a 4 GB instance kept warm around the clock exceeds the always-free
allowance of every metered host, so continuous hosting at this footprint is a paid arrangement
unless the host gives the memory away, as Hugging Face does on CPU basic.

## After deploying

Confirm both halves are live:

```bash
curl -fsS https://<your-url>/_stcore/health
curl -fsS https://<your-url>:8000/health
```

Then put the URL in three places, all of which currently read `[URL]`:

- `manuscript/NAR_WebServer_SAI-CNS_draft.md`, the Abstract and the Data availability section
- `README.md`
- the `source` field returned by `GET /` on the API

## The REST API

```
GET  /                 documentation, as JSON
GET  /health           liveness
GET  /version          model counts and the knowledge-graph fingerprint
GET  /targets          every deployed endpoint with threshold and measured sensitivity
GET  /predict?q=...    full profile for one compound, by name or SMILES
POST /batch            {"compounds": [...]} up to 300
```

```bash
curl "https://<your-url>:8000/predict?q=rimegepant"
curl -X POST https://<your-url>:8000/batch \
     -H "Content-Type: application/json" \
     -d '{"compounds":["donepezil","fluoxetine","CC(=O)Nc1ccc(O)cc1"]}'
```

The API is built on the Python standard library rather than a web framework. That was forced by this
environment, where the network appliance blocks `pydantic_core` as an unscannable binary, but it is
also the better choice for a container whose value rests on a pinned, reproducible scientific stack:
a read-only JSON service over already-loaded models gains nothing from a framework and would gain a
dependency to pin.

## What is deliberately not here

No authentication, no rate limiting, no request logging of submitted structures. The service is
read-only and stateless, and a submitted structure is not written to disk. If you deploy behind an
institutional gateway, add rate limiting there rather than in the application.
