# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
CLAP semantic plausibility gate.

Before judging whether a generated sound follows a physical change, the benchmark checks
that it is a plausible sound for the event at all: each seed is weighted by the CLAP
similarity between its audio and the caption of its video (mapped to [0, 1]).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from flatsounds.audio import MIN_AUDIO_SECONDS, load_audio

# Tokenizers parallelism would warn (and disable itself) when joblib starts the feature-extraction workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The pinned CLAP revision has only pytorch_model.bin; skip the background download of its unused safetensors conversion
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

logger = logging.getLogger(__name__)

CLAP_MODEL = "laion/larger_clap_general"
CLAP_REVISION = "ada0c23a36c4e8582805bb38fec3905903f18b41"
CLAP_SAMPLE_RATE = 48000
MAX_AUDIO_SECONDS = 10  # CLAP input length; longer audio is cropped to its first 10 s


class SemanticPlausibilityGate:
    """Scores audio-caption agreement with CLAP (laion/larger_clap_general)."""

    def __init__(
        self, device: Optional[str] = None, cache_dir: Optional[str] = None, batch_size: int = 64, num_workers: int = 8
    ):
        """
        Args:
            device: 'cuda' or 'cpu' (default: cuda if available).
            cache_dir: Hugging Face cache directory for the checkpoint (default: HF cache).
            batch_size: Audio clips per CLAP forward pass.
            num_workers: Threads that load and resample audio.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"CLAP gate on {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "")
        )
        self._model = (
            ClapModel.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION, cache_dir=cache_dir).to(self.device).eval()
        )
        self._processor = ClapProcessor.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION, cache_dir=cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._text_embeds: Dict[str, torch.Tensor] = {}

    def _load(self, audio_path: str, duration: float) -> Optional[np.ndarray]:
        """Audio at the CLAP sample rate, cropped to the duration (s) and to MAX_AUDIO_SECONDS; None when shorter than
        MIN_AUDIO_SECONDS."""
        audio = load_audio(audio_path, CLAP_SAMPLE_RATE, min(duration, MAX_AUDIO_SECONDS))
        if len(audio) < int(MIN_AUDIO_SECONDS * CLAP_SAMPLE_RATE):
            return None
        return audio

    @torch.inference_mode()
    def _text_embedding(self, caption: str) -> torch.Tensor:
        """L2-normalized CLAP text embedding of a caption, computed once per caption."""
        if caption not in self._text_embeds:
            inputs = self._processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
            embeds = self._model.get_text_features(**inputs).pooler_output
            self._text_embeds[caption] = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return self._text_embeds[caption]

    @torch.inference_mode()
    def compute_similarities(self, audio_paths: List[str], captions: List[str], durations: List[float]) -> List[float]:
        """
        CLAP cosine similarity of each audio file, cropped to its duration (s), with its caption, mapped from [-1, 1]
        to [0, 1]. Audio shorter than 0.1 s (or unreadable) scores 0.0.
        """
        scores = [0.0] * len(audio_paths)
        chunk = self.batch_size * 4
        with (
            ThreadPoolExecutor(self.num_workers) as pool,
            tqdm(total=len(audio_paths), desc="CLAP scoring", disable=len(audio_paths) < chunk) as progress,
        ):
            for start in range(0, len(audio_paths), chunk):
                audios = list(
                    pool.map(self._load, audio_paths[start : start + chunk], durations[start : start + chunk])
                )
                for b in range(0, len(audios), self.batch_size):
                    idx = [start + b + i for i, a in enumerate(audios[b : b + self.batch_size]) if a is not None]
                    if not idx:
                        continue
                    inputs = self._processor(
                        audio=[audios[i - start] for i in idx], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                    ).to(self.device)
                    audio_embeds = self._model.get_audio_features(**inputs).pooler_output
                    audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = torch.cat([self._text_embedding(captions[i]) for i in idx])
                    similarity = (audio_embeds * text_embeds).sum(dim=-1).cpu().numpy()
                    for i, s in zip(idx, similarity):
                        scores[i] = float(np.clip((s + 1.0) / 2.0, 0.0, 1.0))
                progress.update(len(audios))
        return scores
