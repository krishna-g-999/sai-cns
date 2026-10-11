"""SAI-CNS REST API: programmatic access to the same models the web interface serves.

Built on the standard library rather than a web framework, for two reasons. The practical one is that
this deployment environment cannot install FastAPI: the network appliance blocks pydantic_core as an
unscannable binary wheel, and working around a security control to add a dependency is not a trade
worth making. The better reason is that it costs nothing to avoid: this is a read-only JSON service
over models that are already loaded and cached, so a framework would add pinning risk and image size
to a scientific container without adding capability.

Every endpoint calls the same functions as the interface. There is no second copy of the science
here, so the API cannot drift from what the web server reports; if a threshold changes, both change
together.

Endpoints
    GET  /                 this documentation, as JSON
    GET  /health           liveness, for an orchestrator
    GET  /version          model counts and the graph fingerprint the results were computed against
    GET  /targets          every deployed endpoint with its threshold and measured sensitivity
    GET  /predict?q=...    full profile for one compound, by name or SMILES
    POST /batch            {"compounds": ["aspirin", "CCO", ...]} up to 300, one summary row each

Run:  python api.py [port]        (default 8000)
"""
from __future__ import annotations

import json
import sys
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  the single source of truth for every number returned here

MAX_BATCH = 300
_MODELS = None


def models():
    """Loaded once, on the first request, so start-up is fast and health checks answer immediately.

    app.load_models() runs the model-fetch check first, so a deployment without the binaries
    downloads and verifies them here rather than returning wrong answers."""
    global _MODELS
    if _MODELS is None:
        _MODELS = app.load_models()
    return _MODELS


def _fingerprint():
    p = ROOT / "inversion" / "results" / "GRAPH_FINGERPRINT.json"
    return json.loads(p.read_text()) if p.exists() else {}


def doc():
    return {
        "name": "SAI-CNS", "version": app.APP_VERSION,
        "description": "Multi-endpoint prediction of small-molecule effects on the human brain. "
                       "Research decision-support for prioritisation and hypothesis generation. "
                       "Not for medical, diagnostic or treatment decisions.",
        "endpoints": {
            "GET /health": "liveness check",
            "GET /version": "model counts and knowledge-graph fingerprint",
            "GET /targets": "deployed endpoints with thresholds and measured sensitivity",
            "GET /predict?q=<name or SMILES>": "full profile for one compound",
            "POST /batch": f"body {{'compounds': [...]}}, up to {MAX_BATCH}, one summary row each",
        },
        "interpretation": {
            "binder_probability": "calibrated, per-target; NOT comparable between targets because "
                                  "each carries its own threshold and base rate",
            "engagement_signal": "distance above a target's own threshold; the only engagement "
                                 "quantity comparable across targets",
            "disease_score": "strongest engaged mechanism for that condition, gated by predicted "
                             "barrier penetration except for peripherally acting conditions",
            "silence": "no modelled mechanism cleared its threshold. This is not evidence of "
                       "inactivity: median deployed sensitivity is 0.77, falling to 0.26 at the "
                       "strictest endpoints",
            "independent_mechanisms": "engaged targets grouped by homology. Homologues are engaged "
                                      "together, so this is the number to weigh, not the raw count",
        },
        "source": "https://github.com/krishna-g-999/sai-cns",
    }


def targets_payload():
    modes = app.load_binder_modes()
    out = []
    for t in sorted(app.TARGET_KIND):
        if t == "NEURO":
            continue
        v = modes.get(t, {})
        if not v.get("deployed", True):
            continue
        out.append({
            "target": t, "label": app.MECH_LABEL.get(t, t),
            "kind": app.TARGET_KIND[t],
            "threshold": v.get("threshold"),
            "sensitivity_at_threshold": v.get("sensitivity_at_threshold"),
            "auroc_vs_measured_inactives": v.get("auroc_vs_measured_inactives"),
            "diseases": sorted({d for _p, _s, d, _w in app.KNOWLEDGE_GRAPH.get(t, [])}),
        })
    withdrawn = [{"target": t, "reason": v.get("withdrawn_reason")}
                 for t, v in sorted(modes.items()) if not v.get("deployed", True)]
    return {"n_deployed": len(out), "targets": out, "withdrawn": withdrawn,
            "conditions": list(app.DISEASE_ORDER),
            "peripheral_conditions_not_barrier_gated": sorted(app.PERIPHERAL_MECHANISM_DISEASES)}


def predict_one(q):
    smi, name, err = app.resolve_query(q)
    if err:
        return None, err
    r = app.predict_all(smi, models())
    if r is None:
        return None, "structure could not be featurized"
    return app.result_json(smi, name, r), None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"SAI-CNS/{app.APP_VERSION}"

    def log_message(self, fmt, *a):   # quieter than the default, and no client addresses recorded
        sys.stderr.write(f"{self.command} {self.path.split('?')[0]} {fmt % a}\n")

    def _send(self, obj, code=200):
        body = json.dumps(obj, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({}, 204)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/docs"):
                return self._send(doc())
            if u.path == "/health":
                return self._send({"status": "ok"})
            if u.path == "/version":
                return self._send({
                    "version": app.APP_VERSION,
                    "n_models": len(models()),
                    "n_graph_targets": len(app.KNOWLEDGE_GRAPH),
                    "n_conditions": len(app.DISEASE_ORDER),
                    "screening_mode": "standard",
                    "knowledge_graph": _fingerprint()})
            if u.path == "/targets":
                return self._send(targets_payload())
            if u.path == "/predict":
                q = (parse_qs(u.query).get("q") or [""])[0].strip()
                if not q:
                    return self._send({"error": "missing required parameter q"}, 400)
                res, err = predict_one(q)
                return self._send({"error": err, "query": q}, 422) if err else self._send(res)
            return self._send({"error": "not found", "see": "/"}, 404)
        except Exception:
            traceback.print_exc()
            return self._send({"error": "internal error"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/batch":
            return self._send({"error": "not found", "see": "/"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            items = payload.get("compounds")
            if not isinstance(items, list) or not items:
                return self._send({"error": "body must be {'compounds': [ ... ]}"}, 400)
            dropped = max(0, len(items) - MAX_BATCH)
            m = models()

            # Resolve first, then score every resolved compound in ONE vectorised pass, then build
            # the rows. Looping batch_row without `pre` would call each of the seventy estimators
            # once per compound and pay scikit-learn's per-call overhead every time, which measures
            # about fifty times slower. This mirrors render_batch in app.py deliberately: the two
            # batch paths must return the same numbers, so they take the same route to them.
            resolved = []
            for it in items[:MAX_BATCH]:
                smi, name, err = app.resolve_query(str(it))
                resolved.append((it, smi, name, err))

            canon = []
            for _it, smi, _nm, err in resolved:
                if err:
                    canon.append(None); continue
                mol = app.Chem.MolFromSmiles(smi)
                canon.append(app.Chem.MolToSmiles(mol) if mol is not None else None)

            scorable = [c for c in canon if c is not None]
            preds = app.predict_all_batch(scorable, m) if scorable else []
            by_smiles = dict(zip(scorable, preds))

            rows = []
            for (it, smi, name, err), cs in zip(resolved, canon):
                if err:
                    rows.append({"input": it, "status": err})
                elif cs is None:
                    rows.append({"input": it, "smiles": smi, "status": "unparseable"})
                else:
                    rows.append(app.batch_row(smi, name, m, pre=by_smiles.get(cs)))

            return self._send({"n_requested": len(items), "n_returned": len(rows),
                               "n_dropped_over_limit": dropped, "results": rows})
        except Exception:
            traceback.print_exc()
            return self._send({"error": "internal error"}, 500)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"SAI-CNS API on http://0.0.0.0:{port}  (GET / for documentation)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
