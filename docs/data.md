# Data

The data is released under [CC BY-NC 4.0](../LICENSE-DATA) (non-commercial use, with attribution); the code is MIT.

## Files

| Path | Content |
|---|---|
| `benchmark_warped_clips/` ([Hugging Face](https://huggingface.co/datasets/Tinglok/FlatSounds)) | 542 FlatSounds-Physics videos: the paper split uses 446, the verified split 462 (96 of them new). In a video pair, the counterfactual video is time-warped so that its hits occur at the same times as in the factual video |
| `benchmark_clips_full/` ([Hugging Face](https://huggingface.co/datasets/Tinglok/FlatSounds)) | 185 FlatSounds-Single videos: the original recordings |
| [`benchmark/physics_verified.json`](../benchmark/physics_verified.json) | FlatSounds-Physics, verified split (default): 302 test cases, 724 tests |
| [`benchmark/physics_paper.json`](../benchmark/physics_paper.json) | FlatSounds-Physics, paper split (as published, with caption spelling fixed): 268 test cases, 620 tests |
| [`benchmark/single.json`](../benchmark/single.json) | FlatSounds-Single clips: video, caption, duration and hit times |

The videos keep their recorded sound; give your model only the video frames.

The videos are 2.5-10 s long.

## FlatSounds-Physics Test Cases

A **test case** is one video pair or one single video. A **test** is one expected behaviour of one acoustic metric in a test case; the verified split has 302 test cases and 724 tests.

A **pair** test case consists of a factual and a counterfactual video that differ in one factor, for example a wooden spoon and a metal spoon tapping the same glass. Its tests give the expected change of each affected metric from the factual to the counterfactual video:

```json
"fullness_1h_01_f0007cf0008": {
  "category": "fullness",
  "evaluation_method": "pair_comparison",
  "change": { "from": "full", "to": "empty" },
  "factual": { "video": "fullness_1h_01_f0007cf0008_factual.mp4", "caption": "A metal spoon taps a glass full of coffee from the side" },
  "counterfactual": { "video": "fullness_1h_01_f0007cf0008_counterfactual.mp4", "caption": "A metal spoon taps an empty glass from the side" },
  "duration": 6.098,
  "hit_times": [3.099864],
  "expected_metrics": { "decay_rate": "decrease", "fundamental_frequency": "increase", "spectral_centroid": "increase" }
}
```

- **Expected values:** `increase` or `decrease` (the direction of the change), `no_change` (a metric the factor does not affect, e.g. the room's RT60 when only the striker changes), and `ascending` or `descending` (single videos, below).
- **Categories:** the `category` of a pair test case is the changed factor: `striker`, `material`, `fullness`, `surface`, `surface_texture`, `surface_damping`, `scraper_type`, `interaction_type`, `configuration` or `acoustic_environment`. Pair test cases of `consistency` compare two recordings of the same setup.
- **Single-video test cases** have `"evaluation_method": "single_video"`, no `counterfactual` and a `pattern` instead of `change`: `consistent` (category `consistency`: repeated strikes whose per-hit values should stay the same) or `ascending` / `descending` (category `direction`: a note sequence whose pitch rises or falls).
- **F0:** in the verified split, the 29 piano and guitar note sequences carry `"f0": "pitch"` and measure the pitch of each played note; every other test case measures the lowest resonance excited by the hit.
- **Paper split:** `physics_paper.json` keeps its published format (`variable` and a `change` string) and is converted to this schema on load. It has no `f0` field, so its piano and guitar note sequences are measured with the lowest resonance, not the played pitch.

## FlatSounds-Single Clips

Each entry of `benchmark/single.json` gives the video, its caption, its duration (s) and the annotated hit times (s from the start of the video):

```json
"clip_001": {
  "video": "clip_001.mp4",
  "caption": "A metal spoon repeatedly taps the side of a glass cup full of liquid",
  "duration": 6.707,
  "hit_times": [1.352562, 2.49034, 4.179592, 5.723719]
}
```

## Generated Audio

- **File names:** `{video_stem}_seed{N}.wav`, where `{video_stem}` is the video file name without `.mp4` and N runs from 0 to 9. The leaderboard uses 10 seeds; with fewer, pass `--num_runs` (absent seeds otherwise count as failed).
- **File formats:** audio files `.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`, `.mp3`, `.m4a`, `.opus`, or videos with an audio track `.mp4`, `.mov`, `.mkv`, `.webm` (the first audio track is used). Raw `.aac` and `.avi` are not accepted because their audio decodes late. WAV or FLAC is safest: some AAC files decode late and some `.mp3` files do not decode.
- **Timing:** the audio must start with the video, because hit times are seconds from the video start. Generate the full duration; hits after the end of the audio count as missed. Audio after the duration is ignored.
- **Sample rate:** any sample rate and channel count. Audio is mixed to mono, then resampled to 16 kHz for the acoustic metrics and to 48 kHz for CLAP.
- **Captions:** optional; text-conditioned models can use the caption of each video from the metadata.
- **Pairs:** seed *i* of a factual video is compared with seed *i* of its counterfactual video; generate both with random seed *i*.
- **Missing or unusable files:** a missing or unreadable file, a video without an audio track or audio with NaN/inf samples counts as a failed seed, as does audio shorter than 0.1 s in FlatSounds-Physics. The evaluation warns about them; `n_unreadable_files` counts the unusable files that exist.

## Listing the Videos

Every video to generate, with its caption and duration:

```python
import json

videos = {}  # video file -> (caption, duration in s)
for split in ("verified", "paper"):  # a video in both splits has the same caption
    for case in json.load(open(f"benchmark/physics_{split}.json")).values():
        for side in ("factual", "counterfactual"):
            if case.get(side):
                videos.setdefault(case[side]["video"], (case[side]["caption"], case["duration"]))
for clip in json.load(open("benchmark/single.json")).values():
    videos[clip["video"]] = (clip["caption"], clip["duration"])
```
