# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Analysis windows, scoring thresholds and metric names shared by the feature extractor and the evaluator."""

# ============= Per-hit analysis windows (seconds, relative to the hit anchor) =============
# Anchor = detected onset matched to the annotated hit (the annotated time when unmatched)
HIT_PRE_CONTEXT_S = 0.13  # Per-hit segment starts 130 ms before the anchor (attack gate and flux pre-onset frames)
ATTACK_GUARD_S = 0.03  # Attack: onset gate starts 30 ms before the anchor
ATTACK_SEARCH_S = 0.5  # Attack: analysis ends 500 ms after the anchor (a later gate crossing is another event)
ATTACK_SMOOTH_MS = 0.5  # Attack: Gaussian envelope smoothing (sigma, ms); resolves contact times of a few ms
FLUX_PRE_S = 0.094  # Flux: window starts one STFT frame (64 ms) + guard before the anchor
FLUX_POST_S = 0.18  # Flux: window ends 180 ms after the anchor
TIMBRE_PRE_S = 0.02  # Centroid/rolloff: window starts 20 ms before the anchor (covers the onset detector's lag)
TIMBRE_POST_S = 0.18  # Centroid/rolloff: window ends 180 ms after the anchor (contact transient and early ring)
F0_START_S = 0.01  # F0: window starts 10 ms after the anchor (skips the contact click)
F0_WIN_S = 0.29  # F0: window length (to 300 ms after the anchor)

# ============= F0 lowest excited mode (struck objects) =============
F0_MODE_BAND_HZ = (150.0, 5000.0)  # Search band; below 150 Hz sit mains hum and support thumps
F0_MODE_TONAL_DB = 8.0  # Peak level above the median spectrum within +-15% of its frequency
F0_MODE_EXCITED_DB = 6.0  # Peak gain over the same bins in the 150 ms ending 200 ms before the anchor
F0_MODE_RANGE_DB = 20.0  # Lowest peak within this range below the strongest qualifying peak

# ============= Background-subtracted hit spectrum (spectral_centroid, spectral_rolloff) =============
TIMBRE_BG_SUBTRACT = 2.0  # Subtract twice the clip background power spectrum (over-subtraction of stationary noise)
TIMBRE_SNR_GATE_DB = 3.0  # No measurement when the window mean-square is < 3 dB above the clip background power

# ============= Energy decay curve per hit (decay_rate, rt60, drr) =============
DECAY_PEAK_WIN_S = (-0.03, 0.10)  # Hit peak = max |y| within this window around the anchor
DECAY_END_GAP_S = 0.03  # Decay ends 30 ms before the next hit (or the clip end)
DECAY_MIN_S = 0.05  # No measurement with less than 50 ms from the peak to the decay end
DECAY_RATE_MAX_S = 3.0  # decay_rate: EDC over at most 3 s after the peak
EDC_NOISE_MARGIN_DB = 5.0  # EDC integration stops where the smoothed energy falls to 5 dB above the clip noise
C50_PRE_S = 0.002  # drr (C50): early energy starts 2 ms before the peak

# ============= Pair NO_CHANGE margins =============
# Seeds agree on no change when |median_i r_i| <= margin, with r_i = ln(c_i / f_i) (c_i - f_i for DIFFERENCE_METRICS).
NO_CHANGE_MARGIN = {
    "fundamental_frequency": 0.23,
    "spectral_centroid": 0.083,
    "spectral_rolloff": 0.06,
    "spectral_flux": 0.12,
    "attack_time": 0.15,
    "decay_rate": 0.39,
    "rt60": 0.21,
    "drr": 1.2,
    "temporal_modulation": 0.085,
}
DIFFERENCE_METRICS = ("drr", "temporal_modulation")  # dB / bounded index: compared by difference, not ratio
NO_CHANGE_MIN_VALID_SEEDS = 3

# ============= Single-video NO_CHANGE thresholds =============
# Maximum coefficient of variation of the per-hit values
DEFAULT_CV_THRESHOLD = 0.15
CV_THRESHOLDS = {
    "fundamental_frequency": 0.05,
    "spectral_centroid": 0.10,
    "spectral_rolloff": 0.10,
    "attack_time": 0.30,
    "decay_rate": 0.25,
    "spectral_flux": 0.25,
    "rt60": 0.26,
}
# drr (dB): maximum SD of the per-hit values
DRR_SD_THRESHOLD_DB = 2.2

# ============= Single-video F0 trend: minimum |Spearman rho| by number of valid hits =============
RHO_THRESHOLD_SHORT_SEQ = 0.40  # <= 4 valid hits
RHO_THRESHOLD_MEDIUM_SEQ = 0.30  # 5-7 valid hits
RHO_THRESHOLD_LONG_SEQ = 0.25  # >= 8 valid hits

# ============= Metric names =============
ALL_METRIC_NAMES = [
    "spectral_centroid",
    "attack_time",
    "decay_rate",
    "rt60",
    "drr",
    "fundamental_frequency",
    "spectral_rolloff",
    "temporal_modulation",
    "spectral_flux",
]

# Per-hit metrics (clip value = mean over the valid hits; median for ROOM_METRICS)
PER_HIT_METRICS = (
    "attack_time",
    "decay_rate",
    "fundamental_frequency",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flux",
    "rt60",
    "drr",
)
ROOM_METRICS = ("rt60", "drr")
