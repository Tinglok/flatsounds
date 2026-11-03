# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Acoustic metrics of generated audio.

- Hit onsets and hit coverage against the annotated hit times
- Per-hit metrics at each hit's anchor: attack_time, spectral_flux, spectral_centroid, spectral_rolloff,
  fundamental_frequency, decay_rate (material) and rt60, drr (room)
- Whole-clip temporal_modulation
"""

import functools
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
from swift_f0 import SwiftF0

from flatsounds.metrics.constants import (
    ALL_METRIC_NAMES,
    ATTACK_GUARD_S,
    ATTACK_SEARCH_S,
    ATTACK_SMOOTH_MS,
    C50_PRE_S,
    DECAY_END_GAP_S,
    DECAY_MIN_S,
    DECAY_PEAK_WIN_S,
    DECAY_RATE_MAX_S,
    EDC_NOISE_MARGIN_DB,
    F0_MODE_BAND_HZ,
    F0_MODE_EXCITED_DB,
    F0_MODE_RANGE_DB,
    F0_MODE_TONAL_DB,
    F0_START_S,
    F0_WIN_S,
    FLUX_POST_S,
    FLUX_PRE_S,
    HIT_PRE_CONTEXT_S,
    PER_HIT_METRICS,
    ROOM_METRICS,
    TIMBRE_BG_SUBTRACT,
    TIMBRE_POST_S,
    TIMBRE_PRE_S,
    TIMBRE_SNR_GATE_DB,
)
from flatsounds.metrics.types import HitCoverageResult, HitMetrics


@functools.cache
def _pitch_detector() -> SwiftF0:
    """SwiftF0 pitch detector, built once per process (one thread: files are already processed in parallel)."""
    return SwiftF0(threads=1)


class AudioFeatureExtractor:
    """
    Acoustic metrics of mono audio at sample rate sr.

    Per-hit metrics are NaN for a failed measurement; compute_aggregate_metrics_with_hits treats per-hit values that
    are NaN or (except drr, in dB) <= 0 as failed and reports NaN when no hit gives a valid value.
    """

    def __init__(self, sr: int = 16000):
        """
        Args:
            sr: Sample rate (Hz) of the audio passed to the metric methods
        """
        self.sr = sr

        # STFT window and hop of centroid, rolloff and flux: ~64 ms / ~8 ms at any sample rate
        self.SPECTRAL_N_FFT = int(max(256, round(0.064 * self.sr)))
        self.SPECTRAL_HOP = int(max(64, round(0.008 * self.sr)))

    # ============= Hit spectrum: spectral_centroid, spectral_rolloff =============

    def _stft_power(self, x: np.ndarray) -> np.ndarray:
        """Power spectrogram |X|^2 of float64 x with its mean removed (Hann, SPECTRAL_N_FFT / HOP, uncentred)."""
        return (
            np.abs(librosa.stft(x - np.mean(x), n_fft=self.SPECTRAL_N_FFT, hop_length=self.SPECTRAL_HOP, center=False))
            ** 2
        )

    def _background_psd(self, y: np.ndarray) -> np.ndarray:
        """Clip background power spectrum: mean STFT power of the 10% of frames with the lowest total power."""
        P = self._stft_power(y.astype(np.float64))
        frame_power = P.sum(axis=0)
        return P[:, frame_power <= np.percentile(frame_power, 10)].mean(axis=1)

    def _hit_spectrum(
        self, y: np.ndarray, anchor: float, noise: float, background_psd: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Magnitude spectrum of one hit, [anchor - TIMBRE_PRE_S, anchor + TIMBRE_POST_S], after subtracting
        TIMBRE_BG_SUBTRACT x the clip background power spectrum, as a (bins, 1) column. None when the window is
        shorter than one STFT frame, less than TIMBRE_SNR_GATE_DB above the clip background or empty after the
        subtraction.

        Args:
            y: Full audio signal
            anchor: Hit anchor in seconds
            noise: Clip background power (_noise_power)
            background_psd: Clip background power spectrum (_background_psd)
        """
        start = max(0, int(round((anchor - TIMBRE_PRE_S) * self.sr)))
        x = y[start : int(round((anchor + TIMBRE_POST_S) * self.sr))].astype(np.float64)
        if len(x) < self.SPECTRAL_N_FFT or np.mean(x**2) < noise * 10 ** (TIMBRE_SNR_GATE_DB / 10):
            return None
        power = np.maximum(self._stft_power(x) - TIMBRE_BG_SUBTRACT * background_psd[:, None], 0.0)
        spectrum = np.sqrt(power).sum(axis=1, keepdims=True)
        return spectrum if spectrum.any() else None

    def compute_spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Spectral centroid in Hz (brightness): magnitude-weighted mean frequency of a hit spectrum (_hit_spectrum)."""
        return float(librosa.feature.spectral_centroid(S=spectrum, sr=self.sr, n_fft=self.SPECTRAL_N_FFT)[0, 0])

    def compute_spectral_rolloff(self, spectrum: np.ndarray) -> float:
        """Spectral rolloff in Hz: lowest frequency below which 85% of the spectral magnitude of a hit spectrum
        (_hit_spectrum) lies."""
        return float(
            librosa.feature.spectral_rolloff(S=spectrum, sr=self.sr, n_fft=self.SPECTRAL_N_FFT, roll_percent=0.85)[0, 0]
        )

    # ============= Transient: attack_time, spectral_flux =============

    def compute_attack_time(self, y: np.ndarray, onset_time: float) -> float:
        """
        Attack time (ms): 10-90% rise time of the smoothed Hilbert envelope in the 200 ms after the gated onset (the
        first rising sample above the noise floor of the preceding 20 ms, searched from ATTACK_GUARD_S before the
        anchor; the gate start when no sample passes). NaN when there is no rise to measure: the peak is at the end of
        the window or the envelope already starts at >= 90% of it.

        Args:
            y: Audio segment with at least ATTACK_GUARD_S + 20 ms before the anchor
            onset_time: Anchor time in seconds relative to y
        """
        y = y[: int((onset_time + ATTACK_SEARCH_S) * self.sr)]
        env = scipy.ndimage.gaussian_filter1d(np.abs(scipy.signal.hilbert(y)), sigma=ATTACK_SMOOTH_MS * 1e-3 * self.sr)
        der = scipy.ndimage.uniform_filter1d(np.diff(env, prepend=env[0]), size=3)

        # Noise floor and derivative threshold from the 20 ms before the gate start
        start = max(1, int((onset_time - ATTACK_GUARD_S) * self.sr))
        pre_start = max(0, start - int(0.02 * self.sr))
        pre = env[pre_start:start]
        baseline = np.median(pre)
        noise_floor = baseline + 3.0 * (np.median(np.abs(pre - baseline)) + 1e-12)
        der_pre = der[pre_start:start]
        der_thr = 3.0 * (np.median(np.abs(der_pre - np.median(der_pre))) + 1e-12)

        gated = np.flatnonzero((env[start:] > noise_floor) & (der[start:] > der_thr))
        onset = start + (int(gated[0]) if gated.size else 0)
        seg = env[onset : onset + int(0.20 * self.sr)]
        peak_idx = int(np.argmax(seg))
        # No rise to measure: peak at the window end (still rising) or envelope already >= 90% of the peak at the start
        if peak_idx >= len(seg) - 2 or seg[0] >= 0.9 * seg[peak_idx]:
            return float("nan")

        # Rise on the envelope made non-decreasing up to the peak
        mono = np.maximum.accumulate(seg[: peak_idx + 1])
        i10 = int(np.argmax(mono >= 0.1 * seg[peak_idx]))
        i90 = int(np.argmax(mono >= 0.9 * seg[peak_idx]))
        return (i90 - i10) / self.sr * 1000.0

    def compute_spectral_flux(self, y: np.ndarray, onset_time: float) -> float:
        """
        Per-hit spectral flux: mean log-mel spectral flux (librosa onset strength, uncentred STFT) of the
        RMS-normalized window [anchor - FLUX_PRE_S, anchor + FLUX_POST_S]. NaN when the window is shorter than one
        STFT frame plus one hop.

        Args:
            y: Audio segment with at least FLUX_PRE_S before the anchor
            onset_time: Anchor time in seconds relative to y
        """
        o = int(onset_time * self.sr)
        y = y[max(0, o - int(FLUX_PRE_S * self.sr)) : o + int(FLUX_POST_S * self.sr)]
        if y.size < self.SPECTRAL_N_FFT + self.SPECTRAL_HOP:
            return float("nan")
        y = y / (np.sqrt(np.mean(y**2)) + 1e-12)
        S = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_fft=self.SPECTRAL_N_FFT, hop_length=self.SPECTRAL_HOP, center=False
            )
        )
        # [1:] drops the zero frame that onset_strength pads in front for lag 1
        return float(librosa.onset.onset_strength(S=S, sr=self.sr, center=False)[1:].mean())

    # ============= Energy decay: decay_rate (material), rt60 and drr (room) =============

    def _noise_power(self, y: np.ndarray) -> float:
        """Clip background power: 10th percentile of the 20 ms frame mean-square (room tone / recording noise)."""
        n = int(0.02 * self.sr)
        m = len(y) // n
        return float(np.percentile(np.mean(y[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1), 10))

    def _decay_span(self, y: np.ndarray, anchor: float, next_hit: Optional[float]) -> Optional[Tuple[int, int]]:
        """
        Peak and end sample of one hit's decay; None when there is no peak window or less than DECAY_MIN_S of decay.

        The peak is max |y| within DECAY_PEAK_WIN_S of the anchor (not past the end); the end is DECAY_END_GAP_S
        before next_hit (start of the next hit, None for the last hit) or before the clip end.
        """
        dur = len(y) / self.sr
        end_s = min((dur if next_hit is None else next_hit) - DECAY_END_GAP_S, dur)
        a = max(0, int((anchor + DECAY_PEAK_WIN_S[0]) * self.sr))
        b = min(len(y), int(min(anchor + DECAY_PEAK_WIN_S[1], end_s) * self.sr))
        if b <= a + 2:
            return None
        peak = a + int(np.argmax(np.abs(y[a:b])))
        end = int(end_s * self.sr)
        return (peak, end) if end - peak >= int(DECAY_MIN_S * self.sr) else None

    def _edc_db(self, x: np.ndarray, noise: float) -> np.ndarray:
        """
        Noise-compensated Schroeder energy decay curve of x (from the hit peak) in dB re its start; empty when the
        decay reaches the noise floor within 10 ms or carries no energy above it.

        The backward integral of x^2 - noise stops where the 5 ms-smoothed energy first falls to EDC_NOISE_MARGIN_DB
        above the noise after its maximum (or at the end of x); the energy beyond that point is added as the decay
        continuing at the slope of its last 10 dB (Lundeby tail compensation).
        """
        e = x.astype(np.float64) ** 2
        smooth = scipy.ndimage.uniform_filter1d(e, size=int(0.005 * self.sr), mode="nearest")
        top = int(np.argmax(smooth))
        below = np.flatnonzero(smooth[top:] <= noise * 10 ** (EDC_NOISE_MARGIN_DB / 10))
        stop = top + int(below[0]) if below.size else len(e)
        if stop < int(0.01 * self.sr):
            return np.empty(0)
        edc = np.cumsum((e[:stop] - noise)[::-1])[::-1]
        smooth_db = 10 * np.log10(np.maximum(smooth[:stop], 1e-30))
        last10 = int(np.flatnonzero(smooth_db <= smooth_db[stop - 1] + 10.0)[0])
        p_end = max(float(smooth[stop - 1]) - noise, 0.0)  # Signal power where the integration stops
        if p_end > 0 and stop - last10 > 8:
            slope = np.polyfit(np.arange(last10, stop) / self.sr, smooth_db[last10:stop], 1)[0]
            if slope < 0:
                # Energy (in samples) of an exponential decay of power p_end with T60 = 60 / -slope
                edc = edc + p_end * (60.0 / -slope) / (6 * np.log(10)) * self.sr
        if edc[0] <= 0:
            return np.empty(0)
        return 10 * np.log10(np.maximum(edc, 1e-30 * edc[0]) / edc[0])

    def _edc_slope(self, edc: np.ndarray, hi_db: float, lo_db: float) -> float:
        """OLS slope (dB/s) of edc between its first hi_db and first lo_db crossings; NaN when lo_db is not reached,
        the interval has fewer than 8 samples or the curve does not fall."""
        i1, i2 = np.flatnonzero(edc <= hi_db), np.flatnonzero(edc <= lo_db)
        if i2.size == 0 or i2[0] - i1[0] < 8:
            return float("nan")
        slope = np.polyfit(np.arange(i1[0], i2[0]) / self.sr, edc[i1[0] : i2[0]], 1)[0]
        return float(slope) if slope < 0 else float("nan")

    def compute_decay_rate(self, y: np.ndarray, peak: int, end: int, noise: float) -> float:
        """
        Amplitude decay rate 1/tau (1/s) of one hit (material damping): slope of the noise-compensated EDC of
        y[peak:end] (at most DECAY_RATE_MAX_S) over its first 10 dB (EDT interval); NaN when it does not reach -10 dB.

        Args:
            y: Full audio signal
            peak, end: Hit peak and decay end sample (_decay_span)
            noise: Clip background power (_noise_power)
        """
        edc = self._edc_db(y[peak : min(end, peak + int(DECAY_RATE_MAX_S * self.sr))], noise)
        return float(-self._edc_slope(edc, 0.0, -10.0) * np.log(10.0) / 20.0)

    def compute_rt60(self, y: np.ndarray, peak: int, end: int, noise: float) -> float:
        """
        Reverberation time (s) of one hit: 60 dB over the decay slope of the noise-compensated EDC of y[peak:end]
        between -5 and -15 dB (T10); NaN when it does not reach -15 dB. Args as compute_decay_rate.
        """
        return -60.0 / self._edc_slope(self._edc_db(y[peak:end], noise), -5.0, -15.0)

    def compute_drr(self, y: np.ndarray, peak: int, end: int, noise: float) -> float:
        """
        Clarity C50 of one hit in dB (ISO 3382-1), reported as drr: energy from C50_PRE_S before the peak to 50 ms
        after it over the energy after that, from the noise-compensated EDC of y[peak - C50_PRE_S:end]; NaN when the
        EDC ends within 50 ms of the peak or the early part holds no energy above the noise. Args as compute_decay_rate.
        """
        start = max(0, peak - int(C50_PRE_S * self.sr))
        edc = self._edc_db(y[start:end], noise)
        n50 = peak - start + int(0.05 * self.sr)
        if len(edc) <= n50:
            return float("nan")
        late = 10 ** (edc[n50] / 10)  # Fraction of the energy after 50 ms
        if late >= 1:
            return float("nan")
        return float(10 * np.log10((1 - late) / late))

    # ============= fundamental_frequency: played pitch or lowest excited mode =============

    @staticmethod
    def _parabolic_peak_interp(freqs: np.ndarray, spectrum_db: np.ndarray, peak_idx: int) -> float:
        """Sub-bin frequency of a spectral peak from a parabola through the dB spectrum at peak_idx +- 1
        (the bin frequency when the parabola is flat or its vertex lies beyond the neighbouring bins)."""
        y0, y1, y2 = spectrum_db[peak_idx - 1 : peak_idx + 2]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return float(freqs[peak_idx] + (delta if abs(delta) <= 1.0 else 0.0) * (freqs[1] - freqs[0]))

    def _welch_db(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch power spectrum in dB: 128 ms Hann frames, 32 ms hop, 4x zero-padded FFT (x padded to one frame)."""
        n = int(round(0.128 * self.sr))
        x = np.pad(np.asarray(x, dtype=np.float64), (0, max(0, n - len(x))))
        freqs, psd = scipy.signal.welch(
            x, fs=self.sr, window=np.hanning(n), noverlap=n - n // 4, nfft=4 * n, detrend=False, scaling="spectrum"
        )
        return freqs, 10.0 * np.log10(psd + 1e-18)

    @staticmethod
    def _excerpt(y: np.ndarray, start: int, n: int) -> np.ndarray:
        """y[start:start + n], zero-padded where the window extends past either end of y."""
        a, b = max(0, start), min(len(y), start + n)
        return np.pad(y[a:b], (a - start, start + n - b))

    def _lowest_mode(self, y: np.ndarray, hit_time: float, x: np.ndarray) -> float:
        """
        Lowest resonance excited by the hit, in Hz: the lowest spectral peak of x (the F0 window) in F0_MODE_BAND_HZ
        that is tonal, louder than before the hit (when enough of the pre-hit window is in the clip) and within
        F0_MODE_RANGE_DB of the strongest such peak (see the F0_MODE_* constants); NaN when no peak qualifies.
        """
        freqs, db = self._welch_db(x)
        n = int(0.15 * self.sr)
        early_db = self._welch_db(self._excerpt(x, 0, n))[1]
        pre_start = int(round((hit_time - 0.20) * self.sr)) - n
        pre_db = self._welch_db(self._excerpt(y, pre_start, n))[1] if pre_start + n >= n // 2 else None

        lo = np.searchsorted(freqs, F0_MODE_BAND_HZ[0])
        hi = np.searchsorted(freqs, F0_MODE_BAND_HZ[1], side="right")
        modes = []  # (frequency, level) of the tonal, hit-excited peaks
        for k in scipy.signal.find_peaks(db[lo:hi])[0] + lo:
            w = max(3, int(0.15 * freqs[k] / (freqs[1] - freqs[0])))
            if db[k] - np.median(db[max(0, k - w) : k + w + 1]) < F0_MODE_TONAL_DB:
                continue
            bins = slice(k - 2, k + 3)
            if pre_db is not None and early_db[bins].max() - pre_db[bins].max() < F0_MODE_EXCITED_DB:
                continue
            modes.append((self._parabolic_peak_interp(freqs, db, k), db[k]))
        if not modes:
            return float("nan")
        top = max(level for _, level in modes)
        return min(f for f, level in modes if level >= top - F0_MODE_RANGE_DB)

    def compute_fundamental_frequency(self, y: np.ndarray, hit_time: float, kind: str) -> float:
        """
        Per-hit F0 in Hz on [hit_time + 10 ms, hit_time + 300 ms]; NaN when it cannot be measured.

        kind 'pitch' (played notes): SwiftF0 pitch, the median over frames with confidence >= 0.5 (NaN with
        fewer than 3). kind 'mode' (struck objects): the lowest resonance excited by the hit (_lowest_mode).

        Args:
            y: Full audio signal (the mode excitation check reads up to 350 ms before the hit)
            hit_time: Hit anchor in seconds relative to y
            kind: 'pitch' or 'mode' (test-case metadata field "f0"; default 'mode')
        """
        start = int(round((hit_time + F0_START_S) * self.sr))
        x = y[start : start + int(F0_WIN_S * self.sr)]
        # Fewer than 64 ms of audio after the hit (clip end)
        if len(x) < int(0.064 * self.sr):
            return float("nan")
        if kind == "pitch":
            result = _pitch_detector().detect(x, self.sr)
            voiced = result.confidence >= 0.5
            return float(np.median(result.pitch_hz[voiced])) if voiced.sum() >= 3 else float("nan")
        return self._lowest_mode(y, hit_time, x)

    # ============= Whole clip: temporal_modulation =============

    def compute_temporal_modulation(self, y: np.ndarray) -> float:
        """
        Amplitude-modulation index in [0, 1] of the whole clip (loudness fluctuation): a weighted sum of the
        envelope's coefficient of variation (CV), peak factor (PF) and share of modulation power at 4-16 Hz (ME).
        Clips shorter than 1.5 s use the CV only.
        """
        # Shorter than 1.5 s: coefficient of variation of the envelope only
        min_duration_sec = 1.5
        if len(y) < int(min_duration_sec * self.sr):
            min_samples = int(0.01 * self.sr)
            if len(y) < min_samples:
                return 0.0
            envelope = np.abs(scipy.signal.hilbert(y))
            cv = np.std(envelope) / (np.mean(envelope) + 1e-10)
            return float(np.clip(cv / 2.0, 0, 1))

        # Amplitude envelope, smoothed over 5 ms to remove the carrier
        envelope = np.abs(scipy.signal.hilbert(y))
        window_size = int(0.005 * self.sr)
        if window_size > 1:
            envelope = scipy.ndimage.uniform_filter1d(envelope, size=window_size)

        # Anti-alias low-pass (0.45 x the target rate), then decimate to ~200 Hz
        target_sr = 200
        nyquist_ratio = 0.45
        downsample_factor = max(1, int(self.sr / target_sr))
        cutoff_hz = nyquist_ratio * target_sr
        sos_aa = scipy.signal.butter(4, cutoff_hz, fs=self.sr, output="sos")
        envelope = scipy.signal.sosfiltfilt(sos_aa, envelope)
        envelope_ds = envelope[::downsample_factor]
        actual_sr = self.sr / downsample_factor

        # Flat envelope: no modulation
        if np.std(envelope_ds) / (np.mean(envelope_ds) + 1e-10) < 0.005:
            return 0.0

        # Zero-phase high-pass at 1-2 Hz (4 / duration) removes DC and slow level trends
        signal_duration = len(envelope_ds) / actual_sr
        hp_cutoff = max(1.0, min(2.0, 4.0 / signal_duration))
        sos_hp = scipy.signal.butter(2, hp_cutoff, "hp", fs=actual_sr, output="sos")
        envelope_hp = scipy.signal.sosfiltfilt(sos_hp, envelope_ds)

        # CV: envelope fluctuation relative to the mean level
        env_mean = np.mean(np.abs(envelope_ds))
        if env_mean < 1e-10:
            return 0.0
        env_std = np.std(envelope_hp)
        cv = env_std / env_mean

        # PF: peak factor with the 99th percentile as a robust peak (dynamic contrast)
        env_rms = np.sqrt(np.mean(envelope_ds**2))
        if env_rms < 1e-10:
            return 0.0
        env_p99 = np.percentile(envelope_ds, 99)
        peak_factor = env_p99 / env_rms

        # ME: share of the modulation power (DC excluded) at 4-16 Hz (rhythmic modulation)
        mod_fft = np.fft.rfft(envelope_hp)
        mod_freq = np.fft.rfftfreq(len(envelope_hp), d=1 / actual_sr)
        mod_power = np.abs(mod_fft) ** 2
        mod_range = (mod_freq >= 4) & (mod_freq <= 16)
        total_power = np.sum(mod_power[1:])
        if total_power > 1e-10:
            mod_energy = np.sum(mod_power[mod_range]) / total_power
        else:
            mod_energy = 0.0

        # Full scale at CV 1.5 and PF 4
        cv_p95 = 1.5
        pf_p95 = 4.0
        cv_scaled = np.clip(cv / cv_p95, 0, 1)
        pf_normalized = np.clip((peak_factor - 1.0) / (pf_p95 - 1.0), 0, 1)
        modulation_index = 0.4 * cv_scaled + 0.3 * pf_normalized + 0.3 * mod_energy * 2.0
        return float(np.clip(modulation_index * 0.85, 0.0, 1.0))

    # ============= Hit onsets and coverage =============

    def detect_hits(self, y: np.ndarray, delta: float = 0.07) -> np.ndarray:
        """
        Hit onset times (s) in y: librosa onset detection on the onset-strength envelope.

        Args:
            y: Audio signal
            delta: Peak-picking threshold over the local mean, as a fraction of the clip's strongest onset
        """
        # ~3.3 ms hop (300 frames/s) for onset timing resolution: 53 samples at 16 kHz
        onset_hop = max(32, int(self.sr / 300))
        # Reflect padding: zero padding would turn any sound present at t=0 into a false onset
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=onset_hop, pad_mode="reflect")
        return librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=onset_hop, backtrack=False, units="time", delta=delta
        )

    @staticmethod
    def match_hits(detected: np.ndarray, expected_hits: List[float], tolerance: float) -> Tuple[List[int], List[float]]:
        """
        Greedily match expected hits (in time order) to the closest unused detected onset within +-tolerance s.

        Returns:
            Indices (into expected_hits) of the matched hits, and the signed offset (s, detected - expected)
            of each match.
        """
        matched: List[int] = []
        offsets: List[float] = []
        used = np.zeros(len(detected), dtype=bool)
        for idx, exp_t in sorted(enumerate(float(t) for t in expected_hits), key=lambda x: x[1]):
            mask = (~used) & (detected >= exp_t - tolerance) & (detected <= exp_t + tolerance)
            if not np.any(mask):
                continue
            j = np.flatnonzero(mask)[np.argmin(np.abs(detected[mask] - exp_t))]
            used[j] = True
            matched.append(idx)
            offsets.append(float(detected[j] - exp_t))
        return matched, offsets

    def check_hit_coverage(self, y: np.ndarray, expected_hits: List[float]) -> HitCoverageResult:
        """Fraction of the expected hits reproduced in y, with a matching tolerance set by the number and spacing of
        the hits, and the same fraction at times with no hit (chance level)."""
        n = len(expected_hits)
        if n >= 4:
            median_interval = float(np.median(np.diff(sorted(float(t) for t in expected_hits))))
            # 15% of the median hit interval, clamped to [100 ms, 150 ms]
            timing_tolerance = float(min(0.15, max(0.10, 0.15 * median_interval)))
        else:
            timing_tolerance = 0.25 if n <= 2 else 0.20  # 250 ms for 1-2 hits, 200 ms for 3
        detected = self.detect_hits(y)
        matched, offsets = self.match_hits(detected, expected_hits, timing_tolerance)
        # Coverage that onsets at random times would reach: test times with no annotated hit the same way
        grid = np.arange(0.3, len(y) / self.sr - 0.3, 0.2)
        grid = [g for g in grid if min(abs(g - float(t)) for t in expected_hits) > timing_tolerance + 0.1]
        chance = np.mean([np.any(np.abs(detected - g) <= timing_tolerance) for g in grid]) if grid else 0.0
        return HitCoverageResult(
            hit_coverage=len(matched) / n,
            chance_coverage=float(chance),
            mean_timing_error=float(np.mean(np.abs(offsets))) if offsets else float("nan"),  # NaN when no hit matched
            valid_hit_indices=matched,
            matched_onsets={idx: float(expected_hits[idx]) + off for idx, off in zip(matched, offsets)},
        )

    # ============= Per-hit and clip metric values =============

    def extract_hit_metrics(
        self,
        y: np.ndarray,
        hit_times: List[float],
        matched_onsets: Dict[int, float],
        valid_hit_indices: Optional[List[int]] = None,
        metric_names: Iterable[str] = PER_HIT_METRICS,
        f0_kind: str = "mode",
    ) -> List[HitMetrics]:
        """
        Per-hit metrics, each hit analysed at its anchor: the matched detected onset, or the annotated time when no
        onset matched it.

        Args:
            y: Audio signal
            hit_times: Annotated hit times in seconds
            matched_onsets: Detected onset time per matched hit index (HitCoverageResult.matched_onsets)
            valid_hit_indices: Which hits to analyse (None = all hits)
            metric_names: Per-hit metrics to compute; the others are NaN
            f0_kind: F0 definition, 'pitch' (played notes) or 'mode' (struck objects)

        Returns:
            List of HitMetrics for the analysed hits whose anchor lies inside the audio
        """
        hit_metrics = []
        noise = self._noise_power(y)
        background_psd = self._background_psd(y)

        for idx, hit_time in enumerate(hit_times):
            if valid_hit_indices is not None and idx not in valid_hit_indices:
                continue

            anchor = matched_onsets.get(idx, hit_time)
            # An (unmatched) annotated time past the end of the audio has nothing to measure
            if int(anchor * self.sr) >= len(y):
                continue

            # Segment from HIT_PRE_CONTEXT_S before the anchor (attack gate, flux pre-onset frames); each metric
            # bounds its own window after the anchor
            start_sample = int(max(0.0, anchor - HIT_PRE_CONTEXT_S) * self.sr)
            segment = y[start_sample:]
            onset_time_in_segment = anchor - start_sample / self.sr

            # The decay ends where the next hit starts: its annotated time or matched onset, whichever is earlier
            nxt = idx + 1
            next_hit = min(hit_times[nxt], matched_onsets.get(nxt, np.inf)) if nxt < len(hit_times) else None
            span = self._decay_span(y, anchor, next_hit)
            spectrum = self._hit_spectrum(y, anchor, noise, background_psd)

            values = {}
            for m in PER_HIT_METRICS:
                if m not in metric_names:
                    values[m] = float("nan")
                elif m == "fundamental_frequency":
                    # F0 reads the full signal: its excitation check looks up to 350 ms before the anchor
                    values[m] = self.compute_fundamental_frequency(y, anchor, f0_kind)
                elif m in ("decay_rate", *ROOM_METRICS):
                    # Decay-based metrics need a detected event: on background noise the EDC still yields a C50
                    measured = span is not None and idx in matched_onsets
                    values[m] = getattr(self, f"compute_{m}")(y, *span, noise) if measured else float("nan")
                elif m in ("spectral_centroid", "spectral_rolloff"):
                    values[m] = getattr(self, f"compute_{m}")(spectrum) if spectrum is not None else float("nan")
                else:
                    values[m] = getattr(self, f"compute_{m}")(segment, onset_time=onset_time_in_segment)
            hit_metrics.append(HitMetrics(**values))

        return hit_metrics

    def compute_aggregate_metrics_with_hits(
        self,
        y: np.ndarray,
        hit_results: List[HitMetrics],
        metric_names: Iterable[str] = ALL_METRIC_NAMES,
    ) -> Dict[str, float]:
        """
        Clip value of every metric (NaN for the metrics not in metric_names).

        - Per-hit metrics: mean over the analysed hits (median for rt60 and drr), ignoring failed measurements;
          NaN when no hit gives a valid value
        - temporal_modulation: whole clip

        Args:
            y: Audio signal
            hit_results: Per-hit metrics from extract_hit_metrics
            metric_names: Metrics to compute
        """
        results = {m: float("nan") for m in ALL_METRIC_NAMES}

        for metric in PER_HIT_METRICS:
            if metric not in metric_names:
                continue
            # drr is in dB; for the other metrics a value <= 0 is a failed measurement
            valid_values = [
                v for v in (getattr(hm, metric) for hm in hit_results) if np.isfinite(v) and (v > 0 or metric == "drr")
            ]
            if valid_values:
                results[metric] = float(np.median(valid_values) if metric in ROOM_METRICS else np.mean(valid_values))

        if "temporal_modulation" in metric_names:
            results["temporal_modulation"] = self.compute_temporal_modulation(y)

        return results
