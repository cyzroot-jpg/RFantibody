"""
Lightweight, non-invasive inference instrumentation.

Records wall-clock inference time, accelerator peak/current memory usage and
throughput for each inference unit (one RFdiffusion design, one RF2 pose, one
ProteinMPNN structure). Records are appended as JSON lines so they can be
aggregated later without touching the models' normal output.

The instrumentation is *additive*: it never changes model inputs/outputs and
fails silently if the accelerator exposes no memory API.

Control via environment variables:
    RFANTIBODY_PROFILE=0            disable instrumentation
    RFANTIBODY_METRICS_FILE=<path>  override output file
                                    (default: <project_root>/linshi/logs/inference_metrics.jsonl)
"""

import json
import os
import time

from rfantibody.util import device_utils


def _default_metrics_file() -> str:
    root = os.environ.get("RFANTIBODY_ROOT")
    if not root:
        # src/rfantibody/util/inference_metrics.py -> project root is 4 levels up
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(root, "linshi", "logs", "inference_metrics.jsonl")


def metrics_file() -> str:
    return os.path.abspath(os.environ.get("RFANTIBODY_METRICS_FILE") or _default_metrics_file())


def profiling_enabled() -> bool:
    return os.environ.get("RFANTIBODY_PROFILE", "1") != "0"


def start() -> float:
    """Reset peak memory and return a high-resolution start timestamp."""
    if not profiling_enabled():
        return time.perf_counter()
    try:
        device_utils.synchronize()
        device_utils.reset_peak_memory_stats()
    except Exception:
        pass
    return time.perf_counter()


def finish(script: str, tag: str, t0: float, extra: dict = None) -> dict:
    """Finalise an inference record and append it to the metrics file."""
    try:
        device_utils.synchronize()
    except Exception:
        pass
    elapsed = time.perf_counter() - t0

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": script,
        "tag": tag,
        "backend": device_utils.get_backend(),
        "device": device_utils.device_str(),
        "device_name": device_utils.get_device_name(),
        "elapsed_sec": round(elapsed, 6),
        "peak_mem_bytes": device_utils.max_memory_allocated(),
        "used_mem_bytes": device_utils.memory_allocated(),
    }
    mem = device_utils.device_memory_info()
    record["total_mem_bytes"] = mem[1] if mem else 0
    if extra:
        record.update(extra)

    if profiling_enabled():
        _append(record)
    return record


def _append(record: dict) -> None:
    path = metrics_file()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass