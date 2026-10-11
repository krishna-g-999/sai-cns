"""Pre-flight check on the deployed application.

Runs the checks that a browser session cannot make cheaply and that silently break a web server:
every declared import resolves, every model file the app asks for exists and loads, every artefact
referenced by the results panels is present, the knowledge graph is internally consistent, and a set
of chemically diverse compounds produces distinct and plausible output rather than one answer
repeated. The last of these exists because a shared answer across unrelated compounds was a real
defect once and would not raise an exception if it returned.

Exit status is 0 only if every check passes, so this can gate a deployment.

Run it with the interpreter that actually serves the application, not whichever Python happens to be
first on the path, or the version check will report drift that the deployed environment does not
have:

    brainsafe_env\\Scripts\\python.exe src/sai_cns/evaluation/app_health.py

Read-only. Writes nothing.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "sai_cns"))

# chemically unrelated, spanning CNS drugs, peripheral drugs and a natural product
PROBES = {
    "donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "levodopa": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)NC(N)=N",
    "fostamatinib": "COc1cc(Nc2ncc(F)c(Nc3ccccc3)n2)cc(OC)c1OC",
    "resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
    "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
}

fails, warns = [], []


def check(name, fn):
    try:
        msg = fn()
        print(f"  [ok]   {name}{(': ' + msg) if msg else ''}", flush=True)
    except AssertionError as e:
        fails.append(f"{name}: {e}")
        print(f"  [FAIL] {name}: {e}", flush=True)
    except Exception:
        fails.append(f"{name}: {traceback.format_exc(limit=2).strip()}")
        print(f"  [FAIL] {name}:\n{traceback.format_exc(limit=3)}", flush=True)


def main():
    print("1. imports")

    def _imports():
        missing = []
        for mod in ["streamlit", "rdkit", "sklearn", "numpy", "pandas", "joblib",
                    "plotly", "requests", "matplotlib"]:
            try:
                importlib.import_module(mod)
            except ImportError:
                missing.append(mod)
        assert not missing, f"not installed: {', '.join(missing)}"
        return "all third-party imports resolve"
    check("third-party dependencies", _imports)

    # every requirement the app declares must actually be importable, and vice versa
    def _declared():
        req = (ROOT / "requirements.txt").read_text().splitlines()
        names = {ln.split("==")[0].split(">=")[0].strip().lower()
                 for ln in req if ln.strip() and not ln.startswith("#")}
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        for mod, pkg in [("plotly", "plotly"), ("streamlit", "streamlit"),
                         ("joblib", "joblib"), ("rdkit", "rdkit")]:
            if f"import {mod}" in src:
                assert pkg in names, f"app.py imports {mod} but requirements.txt does not list it"
        return f"{len(names)} declared dependencies, all app imports covered"
    check("requirements.txt covers app.py imports", _declared)

    def _versions():
        """The pinned versions are the ones the models were fitted and validated under. Running on
        anything else means the deployed numbers were produced by an unvalidated stack, and for
        pickled estimators it can change predictions silently."""
        import importlib.metadata as md

        def norm(v):
            # rdkit reports 2026.3.2 for a pin written 2026.03.2; strip leading zeros per component
            # so equal versions do not read as drift
            return tuple(p.lstrip("0") or "0" for p in v.strip().split("."))

        drift = []
        for ln in (ROOT / "requirements.txt").read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "==" not in ln:
                continue
            pkg, want = ln.split("==")
            try:
                have = md.version(pkg.strip())
            except md.PackageNotFoundError:
                drift.append(f"{pkg} not installed (pinned {want})")
                continue
            if norm(have) != norm(want):
                drift.append(f"{pkg}: pinned {want.strip()}, installed {have}")
        if drift:
            warns.extend(drift)
            return "WARNING: " + "; ".join(drift)
        return "installed versions match every pin"
    check("environment matches the pinned versions", _versions)

    print("2. application module")
    import app  # noqa: E402  - deliberately after the import check

    def _module():
        for fn in ["load_models", "predict_all", "disease_scores", "assess_domain",
                   "resolve_query", "render_report", "build_network_svg"]:
            assert hasattr(app, fn), f"app.{fn} is missing"
        return "public entry points present"
    check("entry points", _module)

    print("3. model artefacts")
    models = {}

    def _models():
        nonlocal models
        models = app.load_models()
        assert models, "no models loaded"
        return f"{len(models)} model objects loaded"
    check("model files load", _models)

    def _artefacts():
        missing = [p.name for p in [ROOT / "models_rf" / "endpoint_context.json",
                                    ROOT / "models_rf" / "ad_reference.pkl",
                                    ROOT / "models_rf" / "binder_modes.json",
                                    ROOT / "models_rf" / "readacross_index.pkl"] if not p.exists()]
        assert not missing, f"absent: {', '.join(missing)}"
        return "context, domain reference, binder modes and read-across index all present"
    check("supporting artefacts", _artefacts)

    print("4. knowledge graph")

    def _graph():
        kg, order = app.KNOWLEDGE_GRAPH, set(app.DISEASE_ORDER)
        bad = {d for e in kg.values() for (_, _, d, _) in e} - order
        assert not bad, f"diseases outside DISEASE_ORDER: {sorted(bad)}"
        orphan = [d for d in order if d not in app.DISEASE_CONTRIB]
        assert not orphan, f"diseases with no contributing target: {orphan}"
        bad_w = [(t, d, w) for t, e in kg.items() for (_, _, d, w) in e if not 0 < w <= 1]
        assert not bad_w, f"edge weights outside (0, 1]: {bad_w[:3]}"
        return f"{len(kg)} targets, {len(order)} conditions, every edge well formed"
    check("graph consistency", _graph)

    def _covered():
        # every target named in the graph must be scorable, or the edge is decorative
        unscored = []
        for t in app.KNOWLEDGE_GRAPH:
            if t not in app.TARGET_KIND:
                unscored.append(t)
        assert not unscored, f"in the graph but not scorable: {unscored}"
        return "every graph target has a scoring rule"
    check("graph targets are scorable", _covered)

    print("5. inference on chemically diverse probes")
    reports = {}

    def _predict():
        for name, smi in PROBES.items():
            r = app.predict_all(smi, models)
            assert r is not None, f"{name} failed to featurise"
            bbb, neuro, dz = app.disease_scores(r)
            assert np.isfinite(bbb), f"{name} produced a non-finite BBB probability"
            reports[name] = {"bbb": bbb, "neuro": neuro,
                             "top": [(d["disease"], round(d["gated"], 3)) for d in dz[:3]],
                             "top_driver": (dz[0]["driver"][0] if dz and dz[0].get("driver")
                                            else None),
                             "top_disease": dz[0]["disease"] if dz else None,
                             "vec": tuple(round(d["gated"], 6) for d in
                                          sorted(dz, key=lambda x: x["disease"]))}
        return f"{len(reports)} compounds scored"
    check("all probes score", _predict)

    def _distinct():
        vecs = {n: r["vec"] for n, r in reports.items()}
        dupes = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
                 if vecs[a] == vecs[b] and any(v > 0 for v in vecs[a])]
        assert not dupes, f"identical non-zero disease profiles: {dupes}"
        nz = sum(1 for v in vecs.values() if any(x > 0 for x in v))
        assert nz >= len(vecs) // 2, f"only {nz}/{len(vecs)} probes produced any signal"
        return f"{len(set(vecs.values()))} distinct profiles across {len(vecs)} compounds"
    check("distinct compounds give distinct profiles", _distinct)

    def _sanity():
        # directional expectations that follow from established pharmacology, not from our numbers
        problems = []
        if reports["donepezil"]["bbb"] < 0.5:
            problems.append("donepezil is predicted not to reach the brain")
        if reports["metformin"]["bbb"] > reports["fluoxetine"]["bbb"]:
            problems.append("metformin is predicted to penetrate better than fluoxetine")
        # Each of these names the disease AND the target that must drive it, so the check tests the
        # mechanism rather than the ordering. A compound can reach the right disease for the wrong
        # reason, and that is what the previous version of this check failed to notice.
        expected = {
            "donepezil": ("Alzheimer's disease", "AChE"),
            "haloperidol": ("Psychosis / schizophrenia", "D2"),
            "morphine": ("Chronic pain", "OPRM1"),
            "fluoxetine": ("Depression / anxiety", "SERT"),
        }
        for name, (disease, driver) in expected.items():
            got_d, got_t = reports[name]["top_disease"], reports[name]["top_driver"]
            if got_d != disease:
                problems.append(f"{name} top disease is {got_d}, expected {disease}")
            elif got_t != driver:
                problems.append(f"{name} reaches {disease} but is driven by {got_t}, not {driver}")

        # Levodopa is deliberately not asserted here. It is a dopamine precursor, not a ligand, and
        # it engages none of the panel's Parkinson targets, so the panel has no Parkinson-specific
        # evidence for it. Measured: its Parkinson score is 0.1412, identical to its Alzheimer score
        # of 0.1412, because both are driven by the same generic NEURO axis rather than by any
        # Parkinson mechanism. The old assertion that Parkinson appear in its top 3 was therefore
        # satisfied by the tie-break between two equal scores, and it broke when an unrelated
        # endpoint rose above them. Asserting it again would only re-fix the tie, not the pharmacology.
        # Recorded as a limitation in audit/REGENERATION.md instead of tested here.
        assert not problems, "; ".join(problems)
        return ("BBB ordering holds, and four probes reach the right disease through the right "
                "target: donepezil/AChE, haloperidol/D2, morphine/OPRM1, fluoxetine/SERT")
    check("directional pharmacology", _sanity)

    def _peripheral():
        # peripheral drugs should not be presented as actionable CNS findings
        loud = [n for n in ["losartan", "atorvastatin", "metformin"]
                if reports[n]["top"] and reports[n]["top"][0][1] >= app.MIN_ACTIONABLE_SCORE]
        if loud:
            warns.append(f"peripheral controls above the reporting threshold: {loud}")
            return f"WARNING: {loud} exceed MIN_ACTIONABLE_SCORE"
        return "peripheral controls stay below the reporting threshold"
    check("peripheral negative controls", _peripheral)

    print("6. input handling")

    def _resolve():
        smi, nm, err = app.resolve_query("CC(=O)Nc1ccc(O)cc1")
        assert err is None and smi, f"a valid SMILES was rejected: {err}"
        _, _, err2 = app.resolve_query("not-a-molecule-@@@")
        assert err2, "nonsense input was accepted without an error"
        return "valid SMILES accepted, malformed input rejected"
    check("query resolution", _resolve)

    def _network():
        svg = app.build_network_svg(app.predict_all(PROBES["donepezil"], models), "donepezil")
        assert svg and "<svg" in svg, "network figure did not render"
        assert "NaN" not in svg and "None" not in svg, "network figure contains undefined coordinates"
        assert svg.count("<text") >= 4, "network figure rendered without labels"
        return f"network SVG renders, {len(svg):,} characters, {svg.count('<text')} labels"
    check("network figure", _network)

    print("7. export and batch")

    def _exports():
        smi = PROBES["donepezil"]
        r = app.predict_all(smi, models)
        df = app.result_frame(smi, "Donepezil", r)
        assert len(df) > 60, f"result table has only {len(df)} rows"
        assert df.value.notna().all(), "result table contains missing values"
        n_ep = len(app.TARGET_CLASSIFIERS) + len(app.RECEPTOR_REGRESSORS) + len(app.ADME)
        assert len(df) >= n_ep, "result table does not cover every declared endpoint"
        j = app.result_json(smi, "Donepezil", r)
        import json as _j
        _j.dumps(j)  # must be serialisable, so a stray numpy type fails here not in the browser
        assert j["caveats"], "export carries no caveats"
        html = app.build_html_report(smi, "Donepezil", r)
        assert html.lstrip().startswith("<!doctype html"), "report is not a complete document"
        assert "http://" not in html.replace("http://www.w3.org", ""), \
            "report references an external resource and will not render offline"
        assert "<svg" in html, "report contains no figures"
        return (f"table {len(df)} rows, json {len(_j.dumps(j)):,} bytes, "
                f"report {len(html):,} bytes and self-contained")
    check("export formats", _exports)

    def _batch():
        rows = [app.batch_row(PROBES[n], n, models) for n in ("donepezil", "metformin")]
        assert all(x["status"] == "ok" for x in rows), "batch row failed on a valid compound"
        assert rows[0]["top_diseases"] != rows[1]["top_diseases"], \
            "batch returns the same answer for different compounds"
        bad = app.batch_row("not-a-smiles", "junk", models)
        assert bad["status"] != "ok", "batch accepted an unparseable structure"
        return "batch rows differ per compound and reject bad input"
    check("batch screening", _batch)

    def _profile():
        svg = app.build_profile_svg(app.predict_all(PROBES["metformin"], models))
        assert svg and "<svg" in svg, "profile chart did not render for a silent compound"
        assert "NaN" not in svg, "profile chart contains undefined coordinates"
        return f"profile renders even when nothing is engaged, {svg.count('<rect')} elements"
    check("target profile chart", _profile)

    def _independence():
        """Homologous targets must be grouped, or a raw count reads as corroboration it is not."""
        r = app.predict_all(PROBES["haloperidol"], models)
        neuro = app.neuro_signal(r)
        eng = [t for t in app.TARGET_KIND if t != "NEURO" and app.target_signal(r, neuro, t) > 0]
        ind = app.independent_mechanisms(eng)
        _by, multi = app.grouped_engagement(eng)
        assert ind <= len(eng), "independent count exceeds the raw target count"
        assert multi, "haloperidol engages D2 and D3 but no correlated family was detected"
        assert ind < len(eng), "correlated families detected but the count was not reduced"
        # every family member must be a real, scorable target, or the grouping is decorative
        unknown = [t for ts in app.TARGET_FAMILIES.values() for t in ts
                   if t not in app.TARGET_KIND]
        assert not unknown, f"family members that are not scorable targets: {unknown}"
        # family_cofire() reads its numbers from H8_family_correlation.csv rather than a constant
        # declared alongside TARGET_FAMILIES, precisely so they cannot drift from that artefact the
        # way a hand-written badge once did; this still requires every family to appear in its
        # output, with or without a supported pair, or a badge would silently render for nothing.
        assert set(app.family_cofire()) == set(app.TARGET_FAMILIES), \
            "every family must carry a measured co-firing statement"
        j = app.result_json(PROBES["haloperidol"], "Haloperidol", r)
        assert "mechanism_independence" in j, "export omits the independence correction"
        return (f"haloperidol {len(eng)} targets grouped to {ind} independent mechanisms across "
                f"{len(app.TARGET_FAMILIES)} declared families")
    check("homologous targets are grouped", _independence)

    def _coverage():
        """The coverage panel is read as an audit, so it must not be able to describe a different
        panel from the one deployed. This is the check that would have caught the drift in which it
        advertised a withdrawn endpoint, denied two deployed families, and quoted sensitivities of
        0.54 and 0.58 for endpoints measuring 0.774 and 0.660."""
        modes = app.load_binder_modes()
        live = {t for t in app.TARGET_KIND if t != "NEURO"
                and modes.get(t, {}).get("deployed", True)}
        withdrawn = {t for t, v in modes.items() if not v.get("deployed", True)}

        listed = " ".join(t for t, _ in app.coverage_modelled())
        missing = [t for t in live if app.MECH_LABEL.get(t, t) not in listed]
        assert not missing, f"deployed but absent from the coverage panel: {sorted(missing)}"
        advertised = [t for t in withdrawn if app.MECH_LABEL.get(t, t) in listed]
        assert not advertised, f"withdrawn but still advertised as modelled: {sorted(advertised)}"

        # nothing declared unmodelled may in fact be deployed, tested against the structured claim
        # rather than the prose, since prose that explains a substitution necessarily names both
        wrong = sorted(app.COVERAGE_NO_MECHANISMS & live)
        assert not wrong, f"declared not modelled but deployed: {wrong}"
        assert app.COVERAGE_NO and app.COVERAGE_NO_MECHANISMS, "coverage claims are empty"

        # every quoted figure must equal the deployed value, and must actually miss the floor it
        # names. coverage_low() explains a failure by whichever gate the endpoint missed, sensitivity
        # or AUROC or both, not by sensitivity alone, so the check follows the same branching rather
        # than assuming the one failure mode LOW_SENSITIVITY_CUT once assumed.
        for label, why in app.coverage_low():
            ep = next((e for e in modes if app.MECH_LABEL.get(e, e) == label), None)
            assert ep, f"low-sensitivity entry {label} matches no endpoint"
            s = modes[ep].get("sensitivity_at_threshold")
            a = modes[ep].get("auroc_vs_measured_inactives")
            if "sensitivity" in why:
                assert s is not None and f"{s:.2f}" in why, \
                    f"{label} quotes a sensitivity that is not the deployed value {s}"
                assert s < app._panel.MIN_SENSITIVITY, \
                    f"{label} is listed as low on sensitivity but measures {s}"
            if "AUROC" in why:
                assert a is not None and f"{a:.2f}" in why, \
                    f"{label} quotes an AUROC that is not the deployed value {a}"
                assert a < app._panel.MIN_AUROC, \
                    f"{label} is listed as low on AUROC but measures {a}"
            assert "sensitivity" in why or "AUROC" in why, \
                f"{label} names neither gate it could have missed: {why!r}"
        return (f"{len(live)} deployed endpoints all listed, {len(withdrawn)} withdrawn none "
                f"advertised, {len(app.coverage_low())} low-sensitivity entries all matching "
                f"deployed values")
    check("coverage panel matches the deployed panel", _coverage)

    def _adme_units():
        """A declared unit must be consistent with the range of the labels the model was fitted on.

        This exists because plasma-protein binding was labelled "fraction bound" while its training
        labels run from 10.09 to 99.95, so every prediction was displayed as a fraction of 33. A
        user cannot detect that from the number alone; the model was right and only the unit was
        wrong, which is exactly the kind of error that survives review."""
        import pandas as pd
        problems = []
        for ep, (_label, unit, _tf) in app.ADME.items():
            f = ROOT / "data" / "adme" / f"{ep}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            col = "y" if "y" in d else ("label" if "label" in d else None)
            if col is None:
                continue
            y = pd.to_numeric(d[col], errors="coerce").dropna()
            if y.empty:
                continue
            if unit in ("fraction bound", "fraction") and y.max() > 1.0:
                problems.append(f"{ep} declares '{unit}' but labels reach {y.max():.2f}")
            if unit == "probability" and not set(y.unique()) <= {0, 1}:
                problems.append(f"{ep} declares 'probability' but labels are not binary")
        assert not problems, "; ".join(problems)
        return f"all {len(app.ADME)} ADME units consistent with their training label ranges"
    check("ADME units match their data", _adme_units)

    def _nored():
        """No red anywhere: the palette must survive red-green colour-vision deficiency."""
        # Judged on hue, not on the red channel. The brand gold (#F0A500) and the caution amber
        # (#B45309) are oranges near 40 and 26 degrees and are meant to be here; what must not
        # appear is a true red or crimson, which sits above 335 or below 18 degrees.
        import colorsys
        import re as _re
        bad = []
        for path in ((ROOT / "app.py"), (ROOT / ".streamlit" / "config.toml")):
            if not path.exists():
                continue
            for hexv in set(_re.findall(r"#([0-9A-Fa-f]{6})\b", path.read_text(encoding="utf-8"))):
                r_, g_, b_ = (int(hexv[i:i + 2], 16) / 255 for i in (0, 2, 4))
                h, s, v = colorsys.rgb_to_hsv(r_, g_, b_)
                deg = h * 360
                if s > 0.30 and v > 0.25 and (deg >= 335 or deg <= 18):
                    bad.append(f"#{hexv} ({deg:.0f} deg, in {path.name})")
        assert not bad, f"red tones present: {sorted(bad)}"
        return "no red or crimson hue in the palette; gold and amber are oranges and are intended"
    check("no red in the interface", _nored)

    print()
    for n, r in reports.items():
        top = ", ".join(f"{d} {v:.2f}" for d, v in r["top"]) or "silent"
        print(f"  {n:14} BBB {r['bbb']:.2f}   {top}")

    print()
    if fails:
        print(f"FAILED: {len(fails)} check(s)")
        for f in fails:
            print(f"  - {f}")
    if warns:
        print(f"warnings: {len(warns)}")
        for w in warns:
            print(f"  - {w}")
    if not fails:
        print("app.py is healthy: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
