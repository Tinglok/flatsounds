# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Seed voting: the confidence of each test from quality-weighted per-seed votes.

Pair tests compare factual and counterfactual seed values (direction threshold, or a median-change margin for
no_change); single-video tests score each seed's per-hit sequence (consistency, or the F0 trend).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.ndimage
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize

from flatsounds.metrics.constants import (
    CV_THRESHOLDS,
    DEFAULT_CV_THRESHOLD,
    DIFFERENCE_METRICS,
    DRR_SD_THRESHOLD_DB,
    NO_CHANGE_MARGIN,
    NO_CHANGE_MIN_VALID_SEEDS,
    PER_HIT_METRICS,
    RHO_THRESHOLD_LONG_SEQ,
    RHO_THRESHOLD_MEDIUM_SEQ,
    RHO_THRESHOLD_SHORT_SEQ,
)
from flatsounds.metrics.types import ExpectedDirection, MetricResult


def mean_defined(values) -> float:
    """Mean over the finite values (undefined tests are left out); NaN when there is none."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


class PhysicalFidelityEvaluator:
    """Scores the tests of a test case from per-seed metric values and quality weights."""

    def compare_metrics(
        self,
        factual_values: List[float],
        counterfactual_values: List[float],
        expected_direction: ExpectedDirection,
        metric_name: str,
        per_seed_weights: List[float],
    ) -> MetricResult:
        """
        Compare factual vs counterfactual metrics seed by seed (seed i of both sides forms a pair).

        INCREASE / DECREASE: a seed votes for a direction when its delta (counterfactual - factual) exceeds
        tau = max(2% of the pooled mean, 0.25 x the robust SD of the pooled values, 1e-6); confidence and
        opposite_confidence are the weighted votes for the expected and the opposite direction over n_runs.
        NO_CHANGE: with r_i = ln(c_i / f_i) (c_i - f_i for DIFFERENCE_METRICS) over the seeds where it is finite,
        the test passes when there are >= NO_CHANGE_MIN_VALID_SEEDS of them and |median r_i| <= NO_CHANGE_MARGIN;
        confidence = summed weights of those seeds / n_runs if it passes, else 0.

        Args:
            factual_values: Metric value per seed of the factual video (NaN for a failed or missing seed)
            counterfactual_values: Metric value per seed of the counterfactual video, aligned with factual_values
            expected_direction: Expected direction of change (INCREASE, DECREASE or NO_CHANGE)
            metric_name: Name of the metric
            per_seed_weights: Per-seed quality weights (0-1); the denominator stays n_runs, so failed seeds count

        Returns:
            MetricResult of the test
        """
        assert len(factual_values) == len(counterfactual_values), (
            f"factual_values ({len(factual_values)}) and counterfactual_values ({len(counterfactual_values)}) "
            "must have equal length"
        )

        n_runs = len(factual_values)
        f = np.asarray(factual_values, dtype=float)
        c = np.asarray(counterfactual_values, dtype=float)
        weights = np.asarray(per_seed_weights, dtype=float)
        deltas = c - f  # NaN when either side failed: no vote, but the seed still counts in n_runs

        if expected_direction == ExpectedDirection.NO_CHANGE:
            r = deltas if metric_name in DIFFERENCE_METRICS else np.log(c / f)
            valid = np.isfinite(r)
            unchanged = (
                valid.sum() >= NO_CHANGE_MIN_VALID_SEEDS and abs(np.median(r[valid])) <= NO_CHANGE_MARGIN[metric_name]
            )
            confidence = float(weights[valid].sum()) / n_runs if unchanged else 0.0
            opposite = float("nan")
        else:
            valid = np.isfinite(deltas)
            # Direction threshold from the pooled seed values of both sides
            pooled = np.concatenate([f, c])
            finite = pooled[np.isfinite(pooled)]
            mean_term = 0.02 * abs(finite.mean()) if finite.size else 0.0
            robust_std = 1.4826 * np.median(np.abs(finite - np.median(finite))) if finite.size > 1 else 0.0
            tau = max(mean_term, 0.25 * robust_std, 1e-6)
            votes_increase = sum(weights[deltas > tau].tolist())
            votes_decrease = sum(weights[deltas < -tau].tolist())
            if expected_direction == ExpectedDirection.INCREASE:
                confidence, opposite = votes_increase / n_runs, votes_decrease / n_runs
            else:
                confidence, opposite = votes_decrease / n_runs, votes_increase / n_runs

        return MetricResult(
            metric_name=metric_name,
            factual_value=mean_defined(f),
            counterfactual_value=mean_defined(c),
            delta=mean_defined(deltas),
            expected_direction=expected_direction,
            confidence=confidence,
            opposite_confidence=opposite,
            n_total=n_runs,
            n_valid=int(valid.sum()),
        )

    def evaluate_single_video_metric_from_precomputed(
        self,
        per_seed_hit_metrics: List[Optional[List]],
        expected_direction: ExpectedDirection,
        metric_name: str,
        per_seed_weights: List[float],
    ) -> MetricResult:
        """
        Single-video test from the per-hit metrics of each seed.

        NO_CHANGE: a seed votes when the spread of its per-hit values is below the metric threshold.
        ASCENDING / DESCENDING: each seed's F0 sequence shows an ascending trend, a descending trend or none;
        confidence and opposite_confidence are the weighted votes for the expected and the opposite trend.

        Args:
            per_seed_hit_metrics: HitMetrics list per seed (None for a missing seed)
            expected_direction: Expected pattern (NO_CHANGE, ASCENDING or DESCENDING)
            metric_name: Metric to evaluate
            per_seed_weights: Quality weight per seed; the denominator is the number of seeds
        """

        def robust_spread(values):
            """Centre and spread of >= 2 finite values: mean/sample SD (n = 2), median/1.4826 MAD (n = 3-4),
            mean/population SD (n = 5-9), 10% winsorized mean/population SD (n >= 10)."""
            arr = np.asarray(values, float)
            n = len(arr)
            if n >= 5:
                v = np.asarray(winsorize(arr, limits=(0.1, 0.1)))
                return float(np.mean(v)), float(np.std(v))
            if n >= 3:
                med = np.median(arr)
                return float(med), float(1.4826 * (np.median(np.abs(arr - med)) + 1e-12))
            return float(np.mean(arr)), float(np.std(arr, ddof=1))

        n_seeds = len(per_seed_hit_metrics)
        # Consistency needs a within-clip sequence: undefined (NaN, left out of the means) for whole-clip metrics
        if expected_direction == ExpectedDirection.NO_CHANGE and metric_name not in PER_HIT_METRICS:
            return MetricResult(
                metric_name=metric_name,
                factual_value=float("nan"),
                counterfactual_value=float("nan"),
                delta=float("nan"),
                expected_direction=expected_direction,
                confidence=float("nan"),
                n_total=n_seeds,
            )

        votes_correct = 0.0
        votes_opposite = 0.0
        expected_trend = 1 if expected_direction == ExpectedDirection.ASCENDING else -1
        n_valid = 0

        # Every seed counts in the denominator; missing seeds and seeds with < 2 valid hits vote no.
        for seed_idx in range(n_seeds):
            w = per_seed_weights[seed_idx]
            hit_metrics = per_seed_hit_metrics[seed_idx]

            if not hit_metrics:
                continue

            # NO_CHANGE: spread of the seed's per-hit values
            if expected_direction == ExpectedDirection.NO_CHANGE:
                # drr is in dB; for the other metrics a value <= 0 is a failed measurement
                seq = [
                    v
                    for v in (getattr(hm, metric_name) for hm in hit_metrics)
                    if np.isfinite(v) and (v > 0 or metric_name == "drr")
                ]
                if len(seq) < 2:
                    continue
                n_valid += 1
                centre, spread = robust_spread(seq)
                # drr: absolute spread in dB; the other metrics: coefficient of variation
                if metric_name == "drr":
                    consistent = spread < DRR_SD_THRESHOLD_DB
                else:
                    consistent = spread / abs(centre) < CV_THRESHOLDS.get(metric_name, DEFAULT_CV_THRESHOLD)
                if consistent:
                    votes_correct += w
                continue

            # ASCENDING / DESCENDING: trend of the seed's F0 sequence, +1 ascending, -1 descending, 0 none
            f0_seq = [
                hm.fundamental_frequency
                for hm in hit_metrics
                if hm.fundamental_frequency > 0 and np.isfinite(hm.fundamental_frequency)
            ]
            if len(f0_seq) < 2:
                continue
            n_valid += 1
            if len(f0_seq) == 2:
                trend = np.sign(np.log2(f0_seq[1] + 1e-10) - np.log2(f0_seq[0] + 1e-10))
            else:
                logf_raw = np.log2(np.array(f0_seq, float) + 1e-10)
                logf = scipy.ndimage.median_filter(logf_raw, size=3, mode="nearest") if len(logf_raw) >= 5 else logf_raw
                if np.all(logf == logf[0]):
                    rho = 0.0
                else:
                    rho, _ = spearmanr(np.arange(len(logf)), logf)
                rho_min = (
                    RHO_THRESHOLD_SHORT_SEQ
                    if len(logf) <= 4
                    else (RHO_THRESHOLD_MEDIUM_SEQ if len(logf) <= 7 else RHO_THRESHOLD_LONG_SEQ)
                )
                trend = np.sign(rho) if abs(rho) >= rho_min else 0
            if trend == expected_trend:
                votes_correct += w
            elif trend == -expected_trend:
                votes_opposite += w

        trend_test = expected_direction != ExpectedDirection.NO_CHANGE
        return MetricResult(
            metric_name=metric_name,
            factual_value=float("nan"),
            counterfactual_value=float("nan"),
            delta=float("nan"),
            expected_direction=expected_direction,
            confidence=votes_correct / n_seeds,
            opposite_confidence=votes_opposite / n_seeds if trend_test else float("nan"),
            n_total=n_seeds,
            n_valid=n_valid,
        )

    def evaluate_test_case(
        self,
        counterfactual_audio_paths: List[str],
        expected_changes: Dict[str, ExpectedDirection],
        factual_coverage_results: List,
        cf_coverage_results: List,
        clap_scores_factual: List[float],
        clap_scores_cf: List[float],
        precomputed_factual_metrics: List[Dict[str, float]],
        precomputed_cf_metrics: List[Dict[str, float]],
        precomputed_hit_metrics_per_seed: List[Optional[List]],
    ) -> Dict[str, Any]:
        """
        Score every expected metric of one test case with quality-weighted seed votes.

        Seed weight = 0.5 x hit coverage above chance + 0.5 x CLAP score (a pair takes the worse side of each). Pair
        tests use compare_metrics, single-video tests evaluate_single_video_metric_from_precomputed. All per-seed lists
        are aligned by seed; a missing seed has coverage None and NaN metrics.

        Args:
            counterfactual_audio_paths: Counterfactual file per seed (empty for a single-video case)
            expected_changes: Expected direction per metric
            factual_coverage_results, cf_coverage_results: HitCoverageResult per seed of each side
            clap_scores_factual, clap_scores_cf: CLAP score per seed of each side (0 for a missing seed)
            precomputed_factual_metrics, precomputed_cf_metrics: Clip metric values per seed of each side
            precomputed_hit_metrics_per_seed: Factual HitMetrics list per seed (single-video cases)

        Returns:
            {"metrics": MetricResult per metric, "avg_confidence": mean confidence over the defined tests,
            "avg_net_confidence": mean net confidence over the directional tests (NaN if none)}
        """
        is_single_video = not counterfactual_audio_paths

        # Per-seed quality weight in [0, 1]; a missing or unreadable file has coverage 0
        per_seed_weights = []
        for i in range(len(precomputed_factual_metrics)):
            cr_f = factual_coverage_results[i]
            cov_f = cr_f.coverage_above_chance if cr_f is not None else 0.0

            if is_single_video:
                weight_i = 0.5 * cov_f + 0.5 * clap_scores_factual[i]
            else:
                cr_c = cf_coverage_results[i]
                cov_c = cr_c.coverage_above_chance if cr_c is not None else 0.0
                weight_i = 0.5 * min(cov_f, cov_c) + 0.5 * min(clap_scores_factual[i], clap_scores_cf[i])
            per_seed_weights.append(weight_i)

        results = {}

        for metric_name, expected_direction in expected_changes.items():
            if expected_direction in (ExpectedDirection.ASCENDING, ExpectedDirection.DESCENDING) or (
                expected_direction == ExpectedDirection.NO_CHANGE and is_single_video
            ):
                results[metric_name] = self.evaluate_single_video_metric_from_precomputed(
                    precomputed_hit_metrics_per_seed,
                    expected_direction,
                    metric_name,
                    per_seed_weights,
                )
            else:
                results[metric_name] = self.compare_metrics(
                    [m[metric_name] for m in precomputed_factual_metrics],
                    [m[metric_name] for m in precomputed_cf_metrics],
                    expected_direction,
                    metric_name,
                    per_seed_weights,
                )

        # Test-case scores: averages over the expected metrics with a defined test (NaN left out)
        return {
            "metrics": results,
            "avg_confidence": mean_defined([r.confidence for r in results.values()]),
            "avg_net_confidence": mean_defined([r.net_confidence for r in results.values()]),
        }
