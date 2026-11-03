# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Evaluate the temporal alignment of a video-to-audio model on FlatSounds-Single (hit F1, timing).

Writes evaluation_results/<generated_audio_dir name>/single_results.json (see --output_dir).
"""

import argparse
import logging

from flatsounds.single import METADATA_PATH, evaluate


def main():
    parser = argparse.ArgumentParser(description="Temporal alignment (hit F1, timing) on FlatSounds-Single")
    parser.add_argument(
        "--generated_audio_dir",
        type=str,
        required=True,
        help="Directory with {clip}_seed{N}.wav files (or another audio or video format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results directory (default: evaluation_results/<generated_audio_dir name>)",
    )
    parser.add_argument("--metadata_path", type=str, default=str(METADATA_PATH), help="FlatSounds-Single metadata JSON")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of generated seeds per video")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Analysis sample rate in Hz (keep 16000: thresholds are calibrated at 16 kHz)",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Worker processes (default -1 = all available CPUs)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(
        generated_audio_dir=args.generated_audio_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        num_runs=args.num_runs,
        sample_rate=args.sample_rate,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
