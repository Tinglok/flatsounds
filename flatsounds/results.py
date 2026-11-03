# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Results JSON shared by both evaluations: {meta, summary, metric_definitions, results}."""

import json
import math
import subprocess
import tomllib
from pathlib import Path
from typing import Dict, List

from flatsounds import REPO_DIR

DEFAULT_RESULTS_DIR = Path("evaluation_results")  # results go to <this>/<generated_audio_dir name>/


def code_version() -> str:
    """`git describe --always --dirty` of this checkout; the package version outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=REPO_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        with open(REPO_DIR / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]


def nan_to_none(obj):
    """Replace NaN with None (JSON null) in nested dicts and lists."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def write_results(path: Path, meta: Dict, summary: Dict, metric_definitions: Dict, records: List[Dict]) -> Dict:
    """Write the results as strict JSON (NaN -> null) and return them."""
    output = nan_to_none(
        {"meta": meta, "summary": summary, "metric_definitions": metric_definitions, "results": records}
    )
    with open(path, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"Results saved to: {path}")
    return output
