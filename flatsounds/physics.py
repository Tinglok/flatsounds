# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Evaluate a video-to-audio model on the FlatSounds-Physics benchmark.

Stage 1 scores every generated clip against its caption with CLAP (GPU if available),
Stage 2 extracts hit coverage and acoustic metrics per file (CPU workers),
Stage 3 votes across seeds per test case and writes one results JSON (OUTPUT_FILE).
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from flatsounds import BENCHMARK_DIR
from flatsounds.audio import MIN_AUDIO_SECONDS, find_generated_audio_files, load_audio
from flatsounds.metrics import (
    ALL_METRIC_NAMES,
    AudioFeatureExtractor,
    ExpectedDirection,
    HitCoverageResult,
    PhysicalFidelityEvaluator,
)
from flatsounds.metrics.evaluator import mean_defined
from flatsounds.results import DEFAULT_RESULTS_DIR, code_version, write_results

logger = logging.getLogger(__name__)

SPLITS = {
    "verified": BENCHMARK_DIR / "physics_verified.json",  # reviewed labels; the default for reporting results
    "paper": BENCHMARK_DIR / "physics_paper.json",  # labels as published in the CVPR 2026 paper
}
OUTPUT_FILE = "physics_results.json"

# Metric order of the paper's Table 1
METRIC_ORDER = [
    "attack_time",
    "decay_rate",
    "fundamental_frequency",
    "spectral_centroid",
    "spectral_flux",
    "spectral_rolloff",
    "temporal_modulation",
    "rt60",
    "drr",
]
METRIC_DEFINITIONS = {
    "attack_time": "10-90% rise time (ms) of the smoothed Hilbert envelope at the hit onset (material hardness).",
    "decay_rate": (
        "Amplitude decay rate 1/tau (1/s) from the slope of the hit's noise-compensated energy decay curve over its "
        "first 10 dB (damping)."
    ),
    "fundamental_frequency": (
        'Pitch (Hz) of the played note (SwiftF0) in cases marked "f0": "pitch"; otherwise the lowest resonance '
        "(>= 150 Hz) excited by the hit."
    ),
    "spectral_centroid": (
        "Magnitude-weighted mean frequency (Hz) of the hit (-20 to +180 ms around the onset) after subtracting twice "
        "the clip's background power spectrum (brightness)."
    ),
    "spectral_flux": (
        "Mean log-mel spectral flux across the hit onset, -94 to +180 ms (transient sharpness and contrast)."
    ),
    "spectral_rolloff": (
        "Frequency (Hz) below which 85% of the spectral magnitude of the same background-subtracted hit spectrum lies."
    ),
    "temporal_modulation": "Amplitude-modulation index (0-1) of the whole clip's envelope (loudness fluctuation).",
    "rt60": (
        "Reverberation time (s): 60 dB over the slope of the hit's noise-compensated energy decay curve between -5 and "
        "-15 dB (T10)."
    ),
    "drr": (
        "Clarity C50 (dB, ISO 3382-1): energy in the first 50 ms after the hit peak over the energy after it (room "
        "dryness)."
    ),
    "hit_values": (
        "Per-hit metrics are measured at the detected onset matched to each annotated hit (the annotated time when "
        "unmatched); a pair test compares clip values (mean over the matched hits, median for rt60 and drr), a "
        "single-video test the per-hit sequence over all hits."
    ),
    "seed_weight": "0.5 x hit coverage above chance + 0.5 x CLAP score of the seed (each the worse side of a pair).",
    "confidence": (
        "Per test: summed weights of the seeds that show the expected behaviour / number of seeds; missing or failed "
        "seeds count in the denominator."
    ),
    "increase/decrease": (
        "A seed votes for a direction when counterfactual - factual exceeds tau = max(2% of the pooled mean, 0.25 x "
        "the robust SD of the pooled seed values)."
    ),
    "no_change (pair)": (
        "Passes when, over >= 3 seeds with both values, |median of ln(counterfactual / factual)| (the difference for "
        "drr and temporal_modulation) is within the metric's margin (half the median labelled change on the original "
        "recordings); every such seed then counts."
    ),
    "no_change (single video)": (
        "A seed votes when the coefficient of variation of its per-hit values (SD in dB for drr) is below the metric's "
        "threshold; undefined (null) for temporal_modulation, which has no per-hit sequence."
    ),
    "ascending/descending": (
        "A seed votes for a trend when the Spearman correlation of its per-hit F0 with hit order (the sign of the "
        "change for 2 hits) passes a length-dependent threshold."
    ),
    "opposite_confidence": (
        "Summed weights of the seeds that vote for the opposite direction or trend / number of seeds; null for "
        "no_change tests."
    ),
    "net_confidence": (
        "confidence - opposite_confidence: 0 at chance, negative when seeds mostly move the wrong way; null for "
        "no_change tests."
    ),
    "overall": (
        "Overall confidence = mean over test cases of the mean confidence of their tests (net: of their directional "
        "tests); a metric's confidence = mean over its tests; a test case without generated audio scores 0; null "
        "entries are left out of every mean."
    ),
    "hit_coverage": (
        "Fraction of annotated hits matched by a detected onset (tolerance 0.10-0.25 s, set by the number and spacing "
        "of the hits), mean over seeds (missing or failed seeds 0)."
    ),
    "hit_coverage_above_chance": (
        "Per seed, hit coverage minus the same fraction at times with no annotated hit (at least the tolerance + 0.1 s "
        "from every hit), floored at 0, so onsets everywhere earn little; mean over seeds (missing or failed seeds 0)."
    ),
    "timing_error": "Mean |detected - annotated| onset time (s) of the matched hits, mean over seeds.",
    "perfect_alignment": "Fraction of seeds with every annotated hit matched.",
    "clap": (
        "CLAP audio-caption cosine similarity of the audio up to the case duration, mapped to [0, 1], mean over seeds "
        "(missing or failed seeds 0)."
    ),
    "factual/counterfactual/min/overall": (
        "hit_coverage, hit_coverage_above_chance, perfect_alignment and clap per side: factual and counterfactual = "
        "mean over test cases; min = mean over test cases of the worse side of a pair, or the only side of a "
        "single-video case; overall = mean over every video of every test case (a pair counts both videos)."
    ),
}


@dataclass
class FileTask:
    """One generated file to analyze; workers load the audio themselves."""

    pair_idx: int
    side: str  # 'factual' or 'cf'
    seed_idx: int
    path: Optional[str]  # None = missing seed
    duration: float  # audio after the case duration (s) is ignored
    hit_times: List[float]
    metric_names: tuple  # expected metrics of the test case; only these are computed
    f0_kind: str  # 'pitch' (played notes) or 'mode' (struck objects)
    need_single_video_aux: bool = False  # also return the per-hit metrics over all hits (single-video factual files)


@dataclass
class FileResult:
    """Hit coverage, metrics and (single-video only) per-hit metrics of one file."""

    pair_idx: int
    side: str
    seed_idx: int
    coverage: Optional[HitCoverageResult]  # None for a missing, unreadable or too short file
    metrics: Dict[str, float]  # all NaN for a missing or unreadable file
    hit_metrics_list: Optional[List] = None  # per-hit metrics over all hits, single-video factual files only


def process_single_file(task: FileTask, sr: int) -> FileResult:
    """Load one file once and compute its hit coverage and metrics."""
    nan_metrics = {m: float("nan") for m in ALL_METRIC_NAMES}
    if task.path is None:
        return FileResult(task.pair_idx, task.side, task.seed_idx, coverage=None, metrics=nan_metrics)

    extractor = AudioFeatureExtractor(sr=sr)
    y = load_audio(task.path, sr, task.duration)
    if y.size < int(MIN_AUDIO_SECONDS * sr):
        return FileResult(task.pair_idx, task.side, task.seed_idx, coverage=None, metrics=nan_metrics)

    coverage = extractor.check_hit_coverage(y, task.hit_times)

    hit_metrics_list = None
    if task.need_single_video_aux:
        # Single-video sequences (trend / consistency) are evaluated over all annotated hits;
        # their per-hit means are not used.
        hit_metrics_list = extractor.extract_hit_metrics(
            y, task.hit_times, coverage.matched_onsets, metric_names=task.metric_names, f0_kind=task.f0_kind
        )
        hit_results = hit_metrics_list
    else:
        # Pair comparisons average per-hit metrics over the hits the generated audio reproduces.
        hit_results = extractor.extract_hit_metrics(
            y, task.hit_times, coverage.matched_onsets, coverage.valid_hit_indices, task.metric_names, task.f0_kind
        )

    metrics = extractor.compute_aggregate_metrics_with_hits(y, hit_results, task.metric_names)

    return FileResult(
        task.pair_idx,
        task.side,
        task.seed_idx,
        coverage=coverage,
        metrics=metrics,
        hit_metrics_list=hit_metrics_list,
    )


@dataclass
class TestPair:
    """One test case from the benchmark metadata."""

    name: str
    factual_clip: str
    counterfactual_clip: Optional[str]  # None for single-video cases
    expected_changes: Dict[str, ExpectedDirection]
    category: str
    evaluation_method: str
    duration: float
    hit_times: List[float]
    factual_caption: str
    cf_caption: Optional[str]  # None for single-video cases
    f0_kind: str  # 'pitch' for played-note cases, else 'mode'

    @property
    def is_single_video(self) -> bool:
        return self.evaluation_method == "single_video"


def convert_paper_case(case: Dict) -> Dict:
    """A paper-split case in the verified schema: `variable` -> `category`, the `change` string
    "<factor>: <from> to <to>" -> `change` {from, to}, "single_video: <pattern>" -> `pattern`."""
    values = case["change"].partition(":")[2].strip()
    if case.get("evaluation_method") == "single_video":
        converted = {"category": case["variable"], "evaluation_method": "single_video", "pattern": values}
    else:
        src, _, dst = values.partition(" to ")
        converted = {
            "category": case["variable"],
            "evaluation_method": "pair_comparison",
            "change": {"from": src, "to": dst},
        }
    keys = ("factual", "counterfactual", "duration", "hit_times", "expected_metrics", "f0")
    return converted | {k: case[k] for k in keys if k in case}


def load_test_pairs(metadata: Dict) -> List[TestPair]:
    """Test cases of a benchmark JSON in the verified schema; paper-split cases are converted first."""
    test_pairs = []
    for name, case in metadata.items():
        if "variable" in case:
            case = convert_paper_case(case)
        cf = case.get("counterfactual", {})
        test_pairs.append(
            TestPair(
                name=name,
                factual_clip=case["factual"]["video"],
                counterfactual_clip=cf.get("video"),
                expected_changes={m: ExpectedDirection(d) for m, d in case["expected_metrics"].items()},
                category=case["category"],
                evaluation_method=case["evaluation_method"],
                duration=case["duration"],
                hit_times=case["hit_times"],
                factual_caption=case["factual"]["caption"],
                cf_caption=cf.get("caption"),
                f0_kind=case.get("f0", "mode"),
            )
        )
    return test_pairs


class BenchmarkEvaluator:
    """Runs the three evaluation stages over all test cases and writes the results."""

    def __init__(
        self,
        generated_audio_dir: str,
        output_dir: Optional[str] = None,
        metadata_path: str = SPLITS["verified"],
        sample_rate: int = 16000,
        clap_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            generated_audio_dir: Directory with `{video_stem}_seed{N}.wav` files (or another audio.EXTENSIONS format).
            output_dir: Results directory (default: evaluation_results/<generated_audio_dir name>).
            metadata_path: Benchmark metadata JSON.
            sample_rate: Analysis sample rate in Hz.
            clap_cache_dir: Hugging Face cache directory for the CLAP checkpoint (default: HF cache).
        """
        self.generated_audio_dir = Path(generated_audio_dir)
        if not self.generated_audio_dir.is_dir():
            raise FileNotFoundError(f"Generated audio directory not found: {self.generated_audio_dir}")
        if output_dir is None:
            output_dir = DEFAULT_RESULTS_DIR / self.generated_audio_dir.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.evaluator = PhysicalFidelityEvaluator()
        from flatsounds.clap import SemanticPlausibilityGate  # keeps torch out of the feature-extraction workers

        self.plausibility_gate = SemanticPlausibilityGate(cache_dir=clap_cache_dir)

        with open(metadata_path, "rb") as f:
            metadata = f.read()
        self.metadata_md5 = hashlib.md5(metadata).hexdigest()
        self.split = next(
            (name for name, path in SPLITS.items() if Path(metadata_path).resolve() == path), Path(metadata_path).name
        )
        self.test_pairs = load_test_pairs(json.loads(metadata))

    def _resolve_audio(self, tc: TestPair, num_runs: int):
        """Per-seed generated files of both sides; a missing seed is None and counts as a failed seed."""
        factual_dict = find_generated_audio_files(self.generated_audio_dir, tc.factual_clip, num_runs)
        factual_audio = [factual_dict.get(s) for s in range(num_runs)]
        if tc.is_single_video:
            return factual_audio, [], bool(factual_dict)
        cf_dict = find_generated_audio_files(self.generated_audio_dir, tc.counterfactual_clip, num_runs)
        cf_audio = [cf_dict.get(s) for s in range(num_runs)]
        return factual_audio, cf_audio, bool(factual_dict) and bool(cf_dict)

    def _clap_scores(self, cases) -> List[tuple]:
        """Per case: (factual seed scores, counterfactual seed scores)."""
        paths, captions, durations = [], [], []
        for tc, factual_audio, cf_audio in cases:
            for audio, caption in ((factual_audio, tc.factual_caption), (cf_audio, tc.cf_caption)):
                for p in audio:
                    if p is not None:
                        paths.append(p)
                        captions.append(caption)
                        durations.append(tc.duration)
        logger.info(f"CLAP scoring {len(paths)} files")
        scores = iter(self.plausibility_gate.compute_similarities(paths, captions, durations))
        return [tuple([0.0 if p is None else next(scores) for p in audio] for audio in (f, c)) for _, f, c in cases]

    def _side_statistics(self, results: List[FileResult], clap: List[float]) -> Dict[str, float]:
        """Hit coverage (raw and above chance), timing error, perfect alignment and CLAP of one side's seeds."""
        coverage = [fr.coverage.hit_coverage if fr.coverage is not None else 0.0 for fr in results]
        above_chance = [fr.coverage.coverage_above_chance if fr.coverage is not None else 0.0 for fr in results]
        errors = [
            fr.coverage.mean_timing_error
            for fr in results
            if fr.coverage is not None and np.isfinite(fr.coverage.mean_timing_error)
        ]
        return {
            "hit_coverage": float(np.mean(coverage)),
            "hit_coverage_above_chance": float(np.mean(above_chance)),
            "timing_error": float(np.mean(errors)) if errors else float("nan"),
            "perfect_alignment": sum(c >= 0.9999 for c in coverage) / len(coverage),
            "clap": float(np.mean(clap)),
        }

    def _summarize_case(
        self,
        tc: TestPair,
        cf_audio,
        factual: List[FileResult],
        cf: List[FileResult],
        clap_f: List[float],
        clap_c: List[float],
    ) -> Tuple[Dict, List[Dict]]:
        """
        Vote across seeds for one test case.

        Returns:
            The case's alignment and CLAP statistics (counterfactual NaN for a single-video case) and one result
            record per expected metric.
        """
        eval_result = self.evaluator.evaluate_test_case(
            cf_audio,
            tc.expected_changes,
            factual_coverage_results=[fr.coverage for fr in factual],
            cf_coverage_results=[fr.coverage for fr in cf],
            clap_scores_factual=clap_f,
            clap_scores_cf=clap_c,
            precomputed_factual_metrics=[fr.metrics for fr in factual],
            precomputed_cf_metrics=[fr.metrics for fr in cf],
            precomputed_hit_metrics_per_seed=[fr.hit_metrics_list for fr in factual],
        )

        side_f = self._side_statistics(factual, clap_f)
        if tc.is_single_video:
            side_c = dict.fromkeys(side_f, float("nan"))
        else:
            side_c = self._side_statistics(cf, clap_c)
        case = {f"{k}_factual": v for k, v in side_f.items()} | {f"{k}_counterfactual": v for k, v in side_c.items()}
        for k in ("hit_coverage", "hit_coverage_above_chance", "perfect_alignment", "clap"):
            case[f"{k}_min"] = side_f[k] if tc.is_single_video else min(side_f[k], side_c[k])
        case["test_case_confidence"] = eval_result["avg_confidence"]
        case["test_case_net_confidence"] = eval_result["avg_net_confidence"]

        records = [
            {
                "case": tc.name,
                "category": tc.category,
                "evaluation_method": tc.evaluation_method,
                "metric": metric_name,
                "expected": result.expected_direction.value,
                "confidence": result.confidence,
                "opposite_confidence": result.opposite_confidence,
                "net_confidence": result.net_confidence,
                "n_valid": result.n_valid,
                "n_total": result.n_total,
                "factual_mean": result.factual_value,
                "counterfactual_mean": result.counterfactual_value,
                "delta": result.delta,
                "test_case_confidence": case["test_case_confidence"],
                "test_case_net_confidence": case["test_case_net_confidence"],
                "hit_coverage_factual": case["hit_coverage_factual"],
                "hit_coverage_counterfactual": case["hit_coverage_counterfactual"],
                "timing_error_factual": case["timing_error_factual"],
                "timing_error_counterfactual": case["timing_error_counterfactual"],
                "factual_clip": tc.factual_clip,
                "counterfactual_clip": tc.counterfactual_clip,
            }
            for metric_name, result in eval_result["metrics"].items()
        ]
        return case, records

    def run_full_evaluation(self, num_runs: int = 10, n_jobs: int = -1) -> Dict:
        """
        Evaluate all test cases and write OUTPUT_FILE to the output directory.

        Args:
            num_runs: Number of generated seeds per video.
            n_jobs: Number of feature-extraction worker processes (-1 = all CPUs).

        Returns:
            The written results: meta, summary, metric_definitions and one record per (test case, metric).
        """
        cases = [(tc, *self._resolve_audio(tc, num_runs)) for tc in self.test_pairs]
        missing = [tc.name for tc, _, _, has_audio in cases if not has_audio]
        if all(p is None for _, f, c, _ in cases for p in f + c):
            raise FileNotFoundError(
                f"No generated audio found in {self.generated_audio_dir} "
                f"(expected files like {Path(self.test_pairs[0].factual_clip).stem}_seed0.wav)"
            )
        if missing:
            logger.warning(
                f"{len(missing)}/{len(cases)} test cases are missing all generated audio of at least one video: "
                f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
            )
        missing_seeds = [
            f"{Path(clip).stem}_seed{seed}"
            for tc, f, c, has_audio in cases
            if has_audio
            for clip, paths in ((tc.factual_clip, f), (tc.counterfactual_clip, c))
            for seed, p in enumerate(paths)
            if p is None
        ]
        if missing_seeds:
            logger.warning(
                f"{len(missing_seeds)} generated files (seeds) are missing and count as failed seeds: "
                f"{', '.join(missing_seeds[:5])}{' ...' if len(missing_seeds) > 5 else ''}"
            )

        # Stage 1: CLAP similarity per seed (missing seeds score 0).
        clap = self._clap_scores([(tc, f, c) for tc, f, c, _ in cases])

        # Stage 2: hit coverage and metrics per file (a missing file gives coverage None and NaN metrics).
        tasks = []
        for pair_idx, (tc, factual_audio, cf_audio, _) in enumerate(cases):
            metric_names = tuple(tc.expected_changes)
            for seed_idx, path in enumerate(factual_audio):
                tasks.append(
                    FileTask(
                        pair_idx,
                        "factual",
                        seed_idx,
                        path,
                        tc.duration,
                        tc.hit_times,
                        metric_names,
                        tc.f0_kind,
                        tc.is_single_video,
                    )
                )
            for seed_idx, path in enumerate(cf_audio):
                tasks.append(
                    FileTask(pair_idx, "cf", seed_idx, path, tc.duration, tc.hit_times, metric_names, tc.f0_kind)
                )
        logger.info(f"Extracting metrics from {sum(t.path is not None for t in tasks)} files (n_jobs={n_jobs})")
        file_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_file)(task, self.sample_rate) for task in tqdm(tasks, desc="Processing files")
        )
        result_map = {(fr.pair_idx, fr.side, fr.seed_idx): fr for fr in file_results}
        unreadable = [Path(t.path).name for t, fr in zip(tasks, file_results) if t.path and fr.coverage is None]
        if unreadable:
            logger.warning(
                f"{len(unreadable)} generated files are unreadable, have no audio track, contain non-finite samples or "
                f"are shorter than {MIN_AUDIO_SECONDS} s, and count as failed seeds: "
                f"{', '.join(unreadable[:5])}{' ...' if len(unreadable) > 5 else ''}"
            )

        # Stage 3: vote across seeds per test case; a case without generated audio has only failed seeds.
        case_rows, records = [], []
        for pair_idx, (tc, factual_audio, cf_audio, _) in enumerate(cases):
            factual = [result_map[(pair_idx, "factual", s)] for s in range(len(factual_audio))]
            cf = [result_map[(pair_idx, "cf", s)] for s in range(len(cf_audio))]
            case, case_records = self._summarize_case(tc, cf_audio, factual, cf, *clap[pair_idx])
            case_rows.append(case)
            records.extend(case_records)

        meta = {
            "benchmark": "FlatSounds-Physics",
            "split": self.split,
            "metadata_md5": self.metadata_md5,
            "code_version": code_version(),
            "num_runs": num_runs,
            "sample_rate": self.sample_rate,
            "n_cases": len(cases),
            "n_records": len(records),
            "missing_cases": missing,
            "n_unreadable_files": len(unreadable),
        }
        summary = self._summarize(case_rows, records)
        self._print_summary(meta, summary)
        return write_results(self.output_dir / OUTPUT_FILE, meta, summary, METRIC_DEFINITIONS, records)

    @staticmethod
    def _summarize(case_rows: List[Dict], records: List[Dict]) -> Dict:
        """Table 1 (overall and per-metric confidence and net) and Table 2 (alignment, CLAP) means over test cases."""

        def case_mean(key):
            return mean_defined([row[key] for row in case_rows])

        summary = {
            "overall": {
                "confidence": case_mean("test_case_confidence"),
                "net_confidence": case_mean("test_case_net_confidence"),
            },
            "metrics": {},
        }
        for metric in METRIC_ORDER:
            tests = [r for r in records if r["metric"] == metric]
            summary["metrics"][metric] = {
                "confidence": mean_defined([r["confidence"] for r in tests]),
                "net_confidence": mean_defined([r["net_confidence"] for r in tests]),
                "n_tests": sum(bool(np.isfinite(r["confidence"])) for r in tests),
            }
        for stat in ("hit_coverage", "hit_coverage_above_chance", "perfect_alignment", "clap"):
            sides = [row[f"{stat}_factual"] for row in case_rows] + [row[f"{stat}_counterfactual"] for row in case_rows]
            summary[stat] = {
                "overall": mean_defined(sides),
                "factual": case_mean(f"{stat}_factual"),
                "counterfactual": case_mean(f"{stat}_counterfactual"),
                "min": case_mean(f"{stat}_min"),
            }
        return summary

    @staticmethod
    def _print_summary(meta: Dict, summary: Dict):
        """Print Table 1 and Table 2 as one compact table."""

        def fmt(x, width, sign=""):
            return f"{x:{sign}{width}.3f}" if np.isfinite(x) else "-".rjust(width)

        print(
            f"\n{meta['benchmark']} ({meta['split']}): {meta['n_cases']} test cases, {meta['num_runs']} seeds, "
            f"code {meta['code_version']}"
        )
        print(f"{'':24s}{'tests':>7s}{'confidence':>12s}{'net':>9s}")
        n_scored = sum(s["n_tests"] for s in summary["metrics"].values())
        rows = [("Overall", n_scored, summary["overall"])]
        rows += [(m, s["n_tests"], s) for m, s in summary["metrics"].items()]
        for name, n, s in rows:
            print(f"{name:24s}{n:>7d}{fmt(s['confidence'], 12)}{fmt(s['net_confidence'], 9, '+')}")
        print(f"{'':24s}{'factual':>9s}{'counterf.':>11s}{'min':>9s}{'overall':>9s}")
        for stat, label in (
            ("hit_coverage", "Hit coverage"),
            ("hit_coverage_above_chance", "Coverage above chance"),
            ("perfect_alignment", "Perfect alignment"),
            ("clap", "CLAP"),
        ):
            s = summary[stat]
            print(
                f"{label:24s}{fmt(s['factual'], 9)}{fmt(s['counterfactual'], 11)}{fmt(s['min'], 9)}"
                f"{fmt(s['overall'], 9)}"
            )
        if meta["missing_cases"]:
            print(f"{len(meta['missing_cases'])} test cases are missing all generated audio of at least one video")
