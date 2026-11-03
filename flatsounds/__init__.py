# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FlatSounds: physical-fidelity (FlatSounds-Physics) and temporal-alignment (FlatSounds-Single) evaluation of
video-to-audio generation."""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_DIR / "benchmark"
