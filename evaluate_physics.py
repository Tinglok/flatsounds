# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Evaluate a video-to-audio model on FlatSounds-Physics (pair and single-video physics tests).

Writes evaluation_results/<generated_audio_dir name>/physics_results.json (see --output_dir).
"""

import argparse
import logging

from flatsounds.physics import SPLITS, BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a video-to-audio model on FlatSounds-Physics")
    parser.add_argument(
        "--generated_audio_dir",
        type=str,
        required=True,
        help="Directory with {video_stem}_seed{N}.wav files (or another audio or video format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results directory (default: evaluation_results/<generated_audio_dir name>)",
    )
    parser.add_argument(
        "--split", choices=sorted(SPLITS), default="verified", help="Benchmark split: verified (default) or paper"
    )
    parser.add_argument("--metadata_path", type=str, default=None, help="Custom metadata JSON; overrides --split")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Analysis sample rate in Hz (keep 16000: thresholds are calibrated at 16 kHz)",
    )
    parser.add_argument("--num_runs", type=int, default=10, help="Number of generated seeds per video")
    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="Feature-extraction worker processes (default -1 = all available CPUs)"
    )
    parser.add_argument(
        "--clap_cache_dir",
        type=str,
        default=None,
        help="Hugging Face cache directory for the CLAP checkpoint (default: HF cache)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx2", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    evaluator = BenchmarkEvaluator(
        generated_audio_dir=args.generated_audio_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path or SPLITS[args.split],
        sample_rate=args.sample_rate,
        clap_cache_dir=args.clap_cache_dir,
    )
    evaluator.run_full_evaluation(num_runs=args.num_runs, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
