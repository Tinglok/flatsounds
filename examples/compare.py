# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Check example results against examples/expected: every number must agree to 1e-6 (relative, or absolute near 0).

usage: python examples/compare.py examples/expected evaluation_results/examples
"""

import json
import math
import sys
from pathlib import Path

TOLERANCE = 1e-6


def differences(expected, actual, path: str) -> list:
    """Paths whose values differ; numbers are compared within TOLERANCE, everything else exactly."""
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        return [d for k in expected for d in differences(expected[k], actual[k], f"{path}.{k}")]
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [d for i, (e, a) in enumerate(zip(expected, actual)) for d in differences(e, a, f"{path}[{i}]")]
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (expected, actual)):
        if math.isclose(expected, actual, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            return []
    elif expected == actual:
        return []
    return [f"{path}: expected {expected}, got {actual}"]


def main(expected_dir: str, actual_dir: str):
    n_failed = 0
    for name in ("physics_results.json", "single_results.json"):
        expected, actual = (json.loads((Path(d) / name).read_text()) for d in (expected_dir, actual_dir))
        del actual["meta"]["code_version"]  # git commit of the checkout that ran the evaluation; not stored in expected
        diffs = differences(expected, actual, name.removesuffix(".json"))
        print(f"{name}: {'OK' if not diffs else f'{len(diffs)} differences'}")
        for d in diffs[:20]:
            print(f"  {d}")
        n_failed += bool(diffs)
    sys.exit(n_failed)


if __name__ == "__main__":
    main(*sys.argv[1:])
