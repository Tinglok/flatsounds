# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Data types shared by the feature extractor, the evaluator and the Physics pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ExpectedDirection(Enum):
    """Expected behaviour of a metric: pair direction (increase / decrease / no_change) or single-video pattern."""

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass
class MetricResult:
    """Outcome of one test (one expected metric of a test case)."""

    metric_name: str
    factual_value: float  # Mean over the seeds with a value; NaN for single-video tests
    counterfactual_value: float  # As factual_value
    delta: float  # Mean per-seed counterfactual - factual; NaN for single-video tests
    expected_direction: ExpectedDirection
    confidence: float = 0.0  # Weighted vote share for the expected behaviour; NaN for an undefined test
    opposite_confidence: float = float("nan")  # Weighted vote share for the opposite direction; NaN for NO_CHANGE
    n_total: int = 0  # Number of seeds, including failed ones
    n_valid: int = 0  # Seeds with a measurement (not necessarily a vote)

    @property
    def net_confidence(self) -> float:
        """Expected minus opposite vote share: 0 at chance; NaN for NO_CHANGE (no opposite outcome)."""
        return self.confidence - self.opposite_confidence


@dataclass
class HitMetrics:
    """Metric values of one hit (NaN when not measured)."""

    spectral_centroid: float
    attack_time: float
    decay_rate: float
    fundamental_frequency: float
    spectral_flux: float
    spectral_rolloff: float
    rt60: float
    drr: float  # C50 clarity (dB)


@dataclass
class HitCoverageResult:
    """
    Which expected (annotated) hits have a matching detected onset.

    Attributes:
        hit_coverage: Fraction of the expected hits matched (0-1)
        chance_coverage: Fraction of times with no expected hit (0.2 s grid, at least the tolerance + 0.1 s from every
            hit) that have a detected onset within the tolerance (0-1; 0 when there is no such time)
        mean_timing_error: Mean |detected - expected| time (s) of the matched hits (NaN if none matched)
        valid_hit_indices: Indices of the matched hits into the expected hits (not into the detected onsets)
        matched_onsets: Detected onset time (s) of each matched hit, keyed by its index into the expected hits
    """

    hit_coverage: float
    chance_coverage: float
    mean_timing_error: float
    valid_hit_indices: List[int]
    matched_onsets: Dict[int, float]

    @property
    def coverage_above_chance(self) -> float:
        """Hit coverage minus its chance level, floored at 0 (the timing term of the Physics seed weight)."""
        return max(0.0, self.hit_coverage - self.chance_coverage)
