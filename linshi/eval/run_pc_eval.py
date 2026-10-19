#!/usr/bin/env python3
"""
RFantibody PC-side evaluation runner (SDAA port).

Runs the *official* RFantibody test scripts (test/<module>/scripts/*.sh), which
themselves invoke the RFantibody CLI (rfdiffusion / proteinmpnn / rf2 / qv*), and
computes quantitative accuracy metrics for each module by comparing the produced
structures/scores against the repository's reference outputs.

The official harness itself only reports pass/fail (pytest). This runner adds:
    * per-atom coordinate deviation (max / mean / RMSD, Angstrom)
    * residue (sequence) identity percentage
    * element match percentage
    * REMARK (CDR label) consistency
    * RF2 SCORE-metric deltas (pae, pLDDT, RMSD, ...)
    * quiver tag/content preservation

Outputs:
    linshi/pc_report/pc_metrics.json          raw metrics
    linshi/pc_report/pc_metrics_report.md    human-readable report
    linshi/logs/eval_run.log                 full run log

Modules that require model weights are reported as SKIPPED when weights are
missing; the download URLs are recorded in the report.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile

ROOT = Path(__file__).resolve().parents[2]
LINSHI = ROOT / "linshi"
EVAL_OUT = LINSHI / "eval_outputs"
REPORT_DIR = LINSHI / "pc_report"
LOG_DIR = LINSHI / "logs"

H100 = "H100_references"  # SDAA is compared against the H100 references

# ---------------------------------------------------------------------------
# Module definitions: which scripts to run and what to compare
# ---------------------------------------------------------------------------
MODULES = {
    "util": {
        "requires_weights": False,
        "scripts": ["antibody_hlt_conversion.sh", "nanobody_hlt_conversion.sh"],
        "pairs": [
            ("antibody_HLT.pdb", f"test/util/reference_outputs/hu-4D5-8_Fv.pdb"),
            ("nanobody_HLT.pdb", f"test/util/reference_outputs/h-NbBCII10.pdb"),
        ],
        "qv_pairs": [],
    },
    "quiver": {
        "requires_weights": False,
        "scripts": [],
        "pairs": [],
        "qv_pairs": [],
        "pytest": "test/quiver/test_quiver.py",
    },
    "rfdiffusion": {
        "requires_weights": True,
        "scripts": [
            "antibody_pdbdesign.sh",
            "antibody_qvdesign.sh",
            "nanobody_pdbdesign.sh",
            "nanobody_qvdesign.sh",
        ],
        "pairs": [
            ("ab_des_0.pdb", f"test/rfdiffusion/reference_outputs/{H100}/ab_des_0.pdb"),
            ("ab_des_1.pdb", f"test/rfdiffusion/reference_outputs/{H100}/ab_des_1.pdb"),
            ("nb_des_0.pdb", f"test/rfdiffusion/reference_outputs/{H100}/nb_des_0.pdb"),
            ("nb_des_1.pdb", f"test/rfdiffusion/reference_outputs/{H100}/nb_des_1.pdb"),
        ],
        "qv_pairs": [
            ("ab_designs.qv", f"test/rfdiffusion/reference_outputs/{H100}/ab_designs.qv"),
            ("nb_designs.qv", f"test/rfdiffusion/reference_outputs/{H100}/nb_designs.qv"),
        ],
    },
    "proteinmpnn": {
        "requires_weights": True,
        "scripts": ["ab_seq_design.sh"],
        "pairs": [
            ("ab_des_0_dldesign_0.pdb", f"test/proteinmpnn/reference_outputs/{H100}/ab_des_0_dldesign_0.pdb"),
            ("ab_des_0_dldesign_1.pdb", f"test/proteinmpnn/reference_outputs/{H100}/ab_des_0_dldesign_1.pdb"),
        ],
        "qv_pairs": [],
    },
    "rf2": {
        "requires_weights": True,
        "scripts": ["ab_prediction.sh"],
        "pairs": [
            ("ab_proteinmpnn_output_best.pdb", f"test/rf2/reference_outputs/{H100}/ab_proteinmpnn_output_best.pdb"),
        ],
        "qv_pairs": [],
    },
}

WEIGHT_URLS = {
    "RFdiffusion_Ab.pt": "https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt",
    "ProteinMPNN_v48_noise_0.2.pt": "https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt",
    "RF2_ab.pt": "https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt",
    "RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt": (
        "https://zenodo.org/records/17488258/files/"
        "RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _remarks(path) -> list:
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("REMARK"):
                out.append(line.strip())
    return out


def _score_lines(path) -> dict:
    scores = {}
    with open(path) as f:
        for line in f:
            if line.startswith("SCORE "):
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    try:
                        scores[parts[0].replace("SCORE ", "").strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
    return scores


def _within(delta, rel, abs_) -> bool:
    return delta <= abs_ or delta <= rel


def pdb_metrics(ref_path: Path, out_path: Path, coord_threshold=0.1) -> dict:
    m = {"metric_type": "pdb", "coord_threshold_A": coord_threshold}
    try:
        ref = PDBFile.read(str(ref_path)).get_structure(model=1)
        out = PDBFile.read(str(out_path)).get_structure(model=1)
    except Exception as e:  # pragma: no cover
        m["status"] = f"parse_error: {e}"
        return m
    m["ref_atoms"] = int(len(ref))
    m["out_atoms"] = int(len(out))

    ref_remarks, out_remarks = _remarks(ref_path), _remarks(out_path)
    m["remark_lines_match"] = ref_remarks == out_remarks
    m["ref_remark_lines"] = len(ref_remarks)
    m["out_remark_lines"] = len(out_remarks)

    if len(ref) != len(out):
        m["status"] = "atom_count_mismatch"
        return m

    d = np.linalg.norm(ref.coord - out.coord, axis=1)
    m["coord_max_dev_A"] = round(float(d.max()), 4)
    m["coord_mean_dev_A"] = round(float(d.mean()), 6)
    ca = ref.atom_name == "CA"
    if ca.any():
        m["ca_rmsd_A"] = round(float(np.sqrt((d[ca] ** 2).mean())), 4)
    m["within_{}A_pct".format(coord_threshold)] = round(100.0 * float((d <= coord_threshold).mean()), 3)
    m["residue_identity_pct"] = round(100.0 * float((ref.res_name == out.res_name).mean()), 3)
    m["element_match_pct"] = round(100.0 * float((ref.element == out.element).mean()), 3)
    m["status"] = "ok"

    ref_sc, out_sc = _score_lines(ref_path), _score_lines(out_path)
    if ref_sc or out_sc:
        sm = {}
        for k in sorted(set(ref_sc) | set(out_sc)):
            rv, ov = ref_sc.get(k), out_sc.get(k)
            entry = {"ref": rv, "out": ov}
            if rv is not None and ov is not None:
                entry["abs_delta"] = round(abs(rv - ov), 4)
                entry["rel_delta"] = round(abs(rv - ov) / abs(rv), 6) if rv else None
                entry["pass"] = _within(entry["abs_delta"], 0.01, 0.1)
            sm[k] = entry
        m["score_metrics"] = sm
    return m


def qv_metrics(ref_qv: Path, out_qv: Path) -> dict:
    """Compare two quiver files tag-by-tag and per-atom on the contained PDBs."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from rfantibody.util.quiver import Quiver
    except Exception as e:
        return {"metric_type": "qv", "status": f"quiver_import_error: {e}"}

    m = {"metric_type": "quiver"}
    try:
        rq, oq = Quiver(str(ref_qv), mode="r"), Quiver(str(out_qv), mode="r")
    except Exception as e:
        return {"metric_type": "quiver", "status": f"open_error: {e}"}

    r_tags, o_tags = list(rq.get_tags()), list(oq.get_tags())
    m["ref_tags"] = len(r_tags)
    m["out_tags"] = len(o_tags)
    m["tags_match"] = r_tags == o_tags
    m["tags_preserved_pct"] = round(100.0 * len(set(r_tags) & set(o_tags)) / max(1, len(r_tags)), 2)

    per_tag = {}
    with tempfile.TemporaryDirectory() as td:
        for tag in [t for t in r_tags if t in set(o_tags)][:5]:
            try:
                rp = Path(td) / "r.pdb"
                op = Path(td) / "o.pdb"
                rp.write_text("\n".join(rq.get_pdblines(tag)))
                op.write_text("\n".join(oq.get_pdblines(tag)))
                per_tag[tag] = pdb_metrics(rp, op, coord_threshold=0.2)
            except Exception as e:
                per_tag[tag] = {"status": f"error: {e}"}
    m["per_tag"] = per_tag
    m["status"] = "ok"
    return m


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def build_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{LINSHI / 'eval' / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "src"), str(ROOT / "include" / "SE3Transformer"),
        env.get("PYTHONPATH", ""),
    ])
    env.setdefault("RFANTIBODY_METRICS_FILE", str(LOG_DIR / "inference_metrics.jsonl"))
    return env


def run_module(name: str, cfg: dict, log) -> dict:
    result = {"module": name, "requires_weights": cfg["requires_weights"], "scripts": {}}
    outdir = EVAL_OUT / name
    outdir.mkdir(parents=True, exist_ok=True)

    env = build_env()

    if cfg.get("pytest"):
        cmd = [sys.executable, "-m", "pytest", "-q", cfg["pytest"]]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        log(f"[{name}] $ {' '.join(cmd)}  (rc={p.returncode}, {time.time()-t0:.1f}s)")
        log(p.stdout[-4000:])
        if p.stderr:
            log(p.stderr[-2000:])
        # parse pytest summary
        summary = [ln for ln in p.stdout.splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
        result["pytest_summary"] = summary[-1] if summary else "no summary"
        result["pytest_returncode"] = p.returncode
        return result

    for script in cfg["scripts"]:
        script_path = ROOT / "test" / name / "scripts" / script
        cmd = ["bash", str(script_path), str(outdir)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                               text=True, timeout=int(os.environ.get("EVAL_SCRIPT_TIMEOUT", "900")))
        except subprocess.TimeoutExpired as e:
            dt = time.time() - t0
            log(f"[{name}] TIMEOUT after {dt:.1f}s")
            result["scripts"][script] = {"returncode": -1, "elapsed_sec": round(dt, 2), "status": "TIMEOUT"}
            continue
        dt = time.time() - t0
        log(f"[{name}] $ {' '.join(cmd)}  (rc={p.returncode}, {dt:.1f}s)")
        log(p.stdout[-6000:])
        if p.stderr:
            log("STDERR: " + p.stderr[-3000:])
        entry = {"returncode": p.returncode, "elapsed_sec": round(dt, 2)}
        if p.returncode != 0:
            text = (p.stdout or "") + (p.stderr or "")
            if "weights" in text.lower() and ("not found" in text.lower() or "ERROR" in text):
                entry["status"] = "SKIPPED_MISSING_WEIGHTS"
            else:
                entry["status"] = "FAILED"
        else:
            entry["status"] = "ok"
        result["scripts"][script] = entry

    no_weights = cfg["requires_weights"] and not any(weights_available().values())
    if no_weights:
        for script, entry in result["scripts"].items():
            entry["status"] = "SKIPPED_MISSING_WEIGHTS"
        result["skip_reason"] = "model weights not present in ./weights"

    # metrics for declared pairs
    result["file_metrics"] = {}
    for out_name, ref_rel in cfg["pairs"]:
        out_path = outdir / out_name
        ref_path = ROOT / ref_rel
        if not out_path.exists():
            result["file_metrics"][out_name] = {
                "metric_type": "pdb",
                "status": "SKIPPED_MISSING_WEIGHTS" if no_weights else "output_missing",
            }
            continue
        if not ref_path.exists():
            result["file_metrics"][out_name] = {"metric_type": "pdb", "status": "reference_missing"}
            continue
        result["file_metrics"][out_name] = pdb_metrics(ref_path, out_path)

    result["qv_metrics"] = {}
    for out_name, ref_rel in cfg["qv_pairs"]:
        out_path = outdir / out_name
        ref_path = ROOT / ref_rel
        if not out_path.exists():
            result["qv_metrics"][out_name] = {
                "metric_type": "quiver",
                "status": "SKIPPED_MISSING_WEIGHTS" if no_weights else "output_missing",
            }
            continue
        if not ref_path.exists():
            result["qv_metrics"][out_name] = {"metric_type": "quiver", "status": "reference_missing"}
            continue
        result["qv_metrics"][out_name] = qv_metrics(ref_path, out_path)

    return result


def weights_available() -> dict:
    wdir = ROOT / "weights"
    status = {}
    for fn in WEIGHT_URLS:
        status[fn] = (wdir / fn).exists()
    return status


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "eval_run.log"
    fh = open(log_path, "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        fh.write(msg + "\n")
        fh.flush()

    log(f"# RFantibody PC evaluation  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"project root : {ROOT}")
    log(f"device       : {os.environ.get('RFANTIBODY_DEVICE', 'auto')}")

    wstat = weights_available()
    log(f"weights present: {wstat}")

    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "root": str(ROOT), "weights": wstat, "modules": {}}

    for name, cfg in MODULES.items():
        log(f"\n===== module: {name} =====")
        if cfg["requires_weights"] and not any(wstat.values()):
            log(f"[{name}] no weights available -> will attempt run and report")
        results["modules"][name] = run_module(name, cfg, log)

    (REPORT_DIR / "pc_metrics.json").write_text(json.dumps(results, indent=2))
    log(f"\nwrote {REPORT_DIR / 'pc_metrics.json'}")

    # collect instrumentation metrics
    mfile = LOG_DIR / "inference_metrics.jsonl"
    if mfile.exists():
        recs = [json.loads(l) for l in mfile.read_text().splitlines() if l.strip()]
        results["instrumentation_records"] = recs
        (REPORT_DIR / "pc_metrics.json").write_text(json.dumps(results, indent=2))
        log(f"collected {len(recs)} instrumentation records")

    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())