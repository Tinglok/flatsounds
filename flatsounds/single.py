# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Evaluate the temporal alignment of a video-to-audio model on FlatSounds-Single.

For each generated seed, hit onsets are detected in the audio (audio after the clip's duration is ignored) and
matched one-to-one to the annotated hit times of the video within +-0.1 s. Reported per model:
- hit F1 (headline), hit precision and hit coverage (recall): per seed, mean over seeds, then over clips;
- timing error: mean |detected - annotated| time of the matched hits (ms), mean over seeds;
- timing offset and drift (diagnostics): median and Theil-Sen slope against hit time of the per-hit median
  (over seeds) offset of the nearest detected onset within +-0.25 s;
each with a 95% confidence interval from a bootstrap over caption groups. Writes one results JSON (OUTPUT_FILE).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import mir_eval
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import theilslopes
from tqdm import tqdm

from flatsounds import BENCHMARK_DIR
from flatsounds.audio import find_generated_audio_files, load_audio
from flatsounds.metrics import AudioFeatureExtractor
from flatsounds.results import DEFAULT_RESULTS_DIR, code_version, write_results

logger = logging.getLogger(__name__)

METADATA_PATH = BENCHMARK_DIR / "single.json"
OUTPUT_FILE = "single_results.json"
MATCH_TOLERANCE_S = 0.1  # one-to-one hit matching window (hit F1, precision, coverage, timing error)
# Onset peak-picking threshold, stricter than librosa's default 0.07 because hit F1 also counts false alarms
ONSET_DELTA = 0.3
OFFSET_WINDOW_S = 0.25  # nearest-onset window of the offset and drift diagnostics
BOOTSTRAP_SAMPLES = 2000  # caption-group bootstrap resamples of the 95% CIs
BOOTSTRAP_SEED = 0

# Summary rows: key, printed label, number format
SUMMARY_ROWS = [
    ("hit_f1", "Hit F1", ".3f"),
    ("hit_precision", "Hit precision", ".3f"),
    ("hit_coverage", "Hit coverage", ".3f"),
    ("timing_error_ms", "Timing error (ms)", ".1f"),
    ("timing_offset_ms", "Timing offset (ms)", "+.1f"),
    ("timing_drift_ms_per_s", "Timing drift (ms/s)", "+.2f"),
]
METRIC_DEFINITIONS = {
    "onsets": (
        "Hit onsets detected with librosa (onset strength with reflect padding, ~3.3 ms hop, peak-picking threshold "
        "0.3) in the audio up to the clip's duration."
    ),
    "hit_f1": (
        "Headline: F-measure of the detected onsets against the annotated hits, one-to-one matching within +-0.1 s "
        "(mir_eval.onset.f_measure), per seed; mean over seeds (a missing seed scores 0), then over clips."
    ),
    "hit_precision": "Fraction of the detected onsets matched to an annotated hit; same matching and means.",
    "hit_coverage": "Recall: fraction of the annotated hits matched by a detected onset; same matching and means.",
    "timing_error_ms": (
        "Mean |detected - annotated| time (ms) of a seed's matched hits (each within +-0.1 s), mean over the seeds "
        "with at least one match; read it with hit F1, since unmatched hits do not count."
    ),
    "timing_offset_ms": (
        "Diagnostic: per annotated hit, the median over seeds of (nearest detected onset - hit time) when within "
        "+-0.25 s; median over hits (positive = late audio)."
    ),
    "timing_drift_ms_per_s": (
        "Diagnostic: Theil-Sen slope (scipy.stats.theilslopes) of those per-hit medians (ms) against hit time (s); "
        "nonzero = time-scale mismatch. Read offset and drift only when hit F1 is well above chance."
    ),
    "ci95": (
        "95% percentile bootstrap interval: 2000 resamples (seed 0) of whole caption groups (clips sharing a caption); "
        "every model gets the same resamples."
    ),
}


def nearest_onset_offsets(detected: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Signed offset (s) of the detected onset nearest to each hit; NaN when none lies within +-OFFSET_WINDOW_S."""
    if not detected.size:
        return np.full(len(hits), np.nan)
    gaps = detected[None, :] - hits[:, None]
    nearest = gaps[np.arange(len(hits)), np.abs(gaps).argmin(axis=1)]
    return np.where(np.abs(nearest) <= OFFSET_WINDOW_S, nearest, np.nan)


def evaluate_file(path: str, hit_times: List[float], duration: float, sr: int) -> Dict:
    """
    Onset counts, hit F1 / precision / coverage, timing error and per-hit nearest-onset offsets (ms) of one file, and
    whether it could be read.
    """
    extractor = AudioFeatureExtractor(sr=sr)
    y = load_audio(path, sr, duration)
    detected = extractor.detect_hits(y, delta=ONSET_DELTA) if y.size else np.array([])
    detected = detected[detected <= duration]  # resampling can put the last onset frame just after the crop
    hits = np.asarray(hit_times, dtype=float)
    f1, precision, coverage = mir_eval.onset.f_measure(hits, detected, window=MATCH_TOLERANCE_S)
    matched, offsets = extractor.match_hits(detected, hit_times, MATCH_TOLERANCE_S)
    return {
        "n_detected": len(detected),
        "n_matched": len(matched),
        "hit_f1": f1,
        "hit_precision": precision,
        "hit_coverage": coverage,
        "timing_error_ms": 1000.0 * float(np.mean(np.abs(offsets))) if offsets else float("nan"),
        "hit_offsets_ms": (1000.0 * nearest_onset_offsets(detected, hits)).tolist(),
        "readable": bool(y.size),
    }


def median_defined(values) -> float:
    """Median over the finite values; NaN when there is none."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def caption_group_ci(statistic: Callable[[np.ndarray], float], groups: List[str]) -> List[float]:
    """
    95% percentile bootstrap interval of statistic(clip indices), resampling whole caption groups;
    resamples where the statistic is undefined (NaN) are left out, NaN when none defines it.
    """
    labels = np.asarray(groups)
    members = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = (
        np.concatenate([members[k] for k in rng.integers(0, len(members), size=len(members))])
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    samples = np.array([statistic(idx) for idx in resamples])
    samples = samples[np.isfinite(samples)]
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])] if samples.size else [float("nan")] * 2


def summarize(records: List[Dict], clips: Dict) -> Dict:
    """Value and caption-group bootstrap 95% CI of every summary statistic (SUMMARY_ROWS)."""
    names = list(clips)
    by_clip = {c: [] for c in names}
    for r in records:
        by_clip[r["clip"]].append(r)
    f1, precision, coverage = (
        np.array([np.mean([r[k] for r in by_clip[c]]) for c in names])
        for k in ("hit_f1", "hit_precision", "hit_coverage")
    )
    errors = [np.array([r["timing_error_ms"] for r in by_clip[c] if np.isfinite(r["timing_error_ms"])]) for c in names]
    hit_times = [np.asarray(clips[c]["hit_times"], dtype=float) for c in names]
    # Per annotated hit: median over seeds of its nearest-onset offset (NaN when no seed has one)
    hit_offsets = [
        np.array([median_defined(seeds) for seeds in zip(*(r["hit_offsets_ms"] for r in by_clip[c]))]) for c in names
    ]

    def pooled(arrays, idx):
        return np.concatenate([arrays[i] for i in idx])

    def timing_error(idx):
        values = pooled(errors, idx)
        return float(values.mean()) if values.size else float("nan")

    def timing_drift(idx):
        times, offsets = pooled(hit_times, idx), pooled(hit_offsets, idx)
        keep = np.isfinite(offsets)
        if np.unique(times[keep]).size < 2:  # a slope needs two distinct hit times
            return float("nan")
        return float(theilslopes(offsets[keep], times[keep]).slope)

    statistics = {
        "hit_f1": lambda idx: float(f1[idx].mean()),
        "hit_precision": lambda idx: float(precision[idx].mean()),
        "hit_coverage": lambda idx: float(coverage[idx].mean()),
        "timing_error_ms": timing_error,
        "timing_offset_ms": lambda idx: median_defined(pooled(hit_offsets, idx)),
        "timing_drift_ms_per_s": timing_drift,
    }
    groups = [clips[c]["caption"] for c in names]
    all_clips = np.arange(len(names))
    return {key: {"value": stat(all_clips), "ci95": caption_group_ci(stat, groups)} for key, stat in statistics.items()}


def print_summary(meta: Dict, summary: Dict):
    """Print the summary statistics as one compact table."""

    def fmt(x, spec):
        return f"{x:{spec}}" if np.isfinite(x) else "-"

    print(f"\n{meta['benchmark']}: {meta['n_clips']} clips, {meta['num_runs']} seeds, code {meta['code_version']}")
    print(f"{'':22s}{'value':>9s}   95% CI")
    for key, label, spec in SUMMARY_ROWS:
        s = summary[key]
        lo, hi = s["ci95"]
        print(f"{label:22s}{fmt(s['value'], spec):>9s}   [{fmt(lo, spec)}, {fmt(hi, spec)}]")
    if meta["n_missing_files"]:
        print(f"{meta['n_missing_files']}/{meta['n_records']} generated files missing (scored 0)")


def evaluate(
    generated_audio_dir: str,
    output_dir: Optional[str] = None,
    metadata_path: str = METADATA_PATH,
    num_runs: int = 10,
    sample_rate: int = 16000,
    n_jobs: int = -1,
) -> Dict:
    """
    Evaluate all clips and write OUTPUT_FILE to the output directory.

    Args:
        generated_audio_dir: Directory with `{clip}_seed{N}.wav` files (or another audio.EXTENSIONS format).
        output_dir: Results directory (default: evaluation_results/<generated_audio_dir name>).
        metadata_path: FlatSounds-Single metadata JSON.
        num_runs: Number of generated seeds per video.
        sample_rate: Analysis sample rate in Hz.
        n_jobs: Number of worker processes (-1 = all CPUs).

    Returns:
        The written results: meta, summary, metric_definitions and one record per (clip, seed).
    """
    generated_audio_dir = Path(generated_audio_dir)
    if not generated_audio_dir.is_dir():
        raise FileNotFoundError(f"Generated audio directory not found: {generated_audio_dir}")
    output_dir = Path(output_dir or DEFAULT_RESULTS_DIR / generated_audio_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "rb") as f:
        metadata = f.read()
    clips = json.loads(metadata)

    records = []
    for clip, info in clips.items():
        found = find_generated_audio_files(generated_audio_dir, info["video"], num_runs)
        for seed in range(num_runs):
            records.append({"clip": clip, "seed": seed, "file": found.get(seed), "n_hits": len(info["hit_times"])})
    missing = [f"{Path(clips[r['clip']]['video']).stem}_seed{r['seed']}" for r in records if r["file"] is None]
    if len(missing) == len(records):
        raise FileNotFoundError(
            f"No generated audio found in {generated_audio_dir} (expected files like clip_001_seed0.wav)"
        )
    if missing:
        logger.warning(
            f"{len(missing)}/{len(records)} generated files are missing and score 0: "
            f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
        )

    present = [r for r in records if r["file"] is not None]
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_file)(r["file"], clips[r["clip"]]["hit_times"], clips[r["clip"]]["duration"], sample_rate)
        for r in tqdm(present, desc="Detecting hits")
    )
    unreadable = [Path(r["file"]).name for r, res in zip(present, results) if not res.pop("readable")]
    if unreadable:
        logger.warning(
            f"{len(unreadable)} generated files are unreadable, have no audio track or contain non-finite samples, "
            f"and score 0: {', '.join(unreadable[:5])}{' ...' if len(unreadable) > 5 else ''}"
        )
    for r, res in zip(present, results):
        r.update(res, file=Path(r["file"]).name)
    for r in records:
        if r["file"] is None:
            r.update(
                {
                    "n_detected": 0,
                    "n_matched": 0,
                    "hit_f1": 0.0,
                    "hit_precision": 0.0,
                    "hit_coverage": 0.0,
                    "timing_error_ms": float("nan"),
                    "hit_offsets_ms": [float("nan")] * r["n_hits"],
                }
            )

    meta = {
        "benchmark": "FlatSounds-Single",
        "metadata": Path(metadata_path).name,
        "metadata_md5": hashlib.md5(metadata).hexdigest(),
        "code_version": code_version(),
        "num_runs": num_runs,
        "sample_rate": sample_rate,
        "n_clips": len(clips),
        "n_caption_groups": len({info["caption"] for info in clips.values()}),
        "n_records": len(records),
        "n_missing_files": len(missing),
        "n_unreadable_files": len(unreadable),
    }
    summary = summarize(records, clips)
    print_summary(meta, summary)
    return write_results(output_dir / OUTPUT_FILE, meta, summary, METRIC_DEFINITIONS, records)
