# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Audio loading and the naming convention of generated files (`{video_stem}_seed{N}.<extension>`)."""

import logging
from pathlib import Path
from typing import Dict

import av
import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Generated-file formats, in lookup order: audio files read with soundfile, then audio and video files decoded with
# PyAV (FFmpeg; the first audio track of a video)
SOUNDFILE_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")
# Raw .aac and .avi (mp3) are left out: they keep no encoder-delay information, so their audio decodes late
PYAV_EXTENSIONS = (".mp3", ".m4a", ".opus", ".mp4", ".mov", ".mkv", ".webm")
EXTENSIONS = SOUNDFILE_EXTENSIONS + PYAV_EXTENSIONS
MIN_AUDIO_SECONDS = 0.1  # shorter audio counts as a failed seed


def load_audio(audio_path: str, sr: int, max_seconds: float) -> np.ndarray:
    """
    Load an audio or video file as mono float32 at sr Hz: SOUNDFILE_EXTENSIONS with soundfile, everything else with
    PyAV (first audio track), both decoded at the native rate, cropped to max_seconds and resampled with librosa. An
    unreadable file, a video without audio or audio with non-finite samples gives an empty array, which callers score
    as a failed seed.
    """
    if Path(audio_path).suffix.lower() in SOUNDFILE_EXTENSIONS:
        try:
            y, file_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        except Exception as e:
            logger.debug(f"Error reading {audio_path} with soundfile: {e}")
            return np.array([], dtype=np.float32)
        y = y.mean(axis=1)
    else:
        try:
            with av.open(audio_path) as container:
                if not container.streams.audio:
                    return np.array([], dtype=np.float32)
                stream = container.streams.audio[0]
                file_sr = stream.rate
                # Planar float at the native rate and channel layout: one row per channel
                to_planar = av.AudioResampler(format="fltp")
                chunks = [f.to_ndarray() for frame in container.decode(stream) for f in to_planar.resample(frame)]
                chunks += [f.to_ndarray() for f in to_planar.resample(None)]
        except Exception as e:
            logger.debug(f"Error reading {audio_path} with PyAV: {e}")
            return np.array([], dtype=np.float32)
        if not chunks:
            return np.array([], dtype=np.float32)
        y = np.concatenate(chunks, axis=1).mean(axis=0)
    y = y[: round(max_seconds * file_sr)]
    if not np.isfinite(y).all():
        return np.array([], dtype=np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def find_generated_audio_files(generated_audio_dir: Path, video_clip: str, num_runs: int) -> Dict[int, str]:
    """Map seed index -> generated file `{video_stem}_seed{N}.<ext>` (first of EXTENSIONS that exists) per seed."""
    generated_audio_dir = Path(generated_audio_dir)
    stem = Path(video_clip).stem
    audio_files = {}
    for seed in range(num_runs):
        for ext in EXTENSIONS:
            path = generated_audio_dir / f"{stem}_seed{seed}{ext}"
            if path.exists():
                audio_files[seed] = str(path)
                break
    return audio_files
