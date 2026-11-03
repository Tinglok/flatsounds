# Metrics and Scoring

## Headline Numbers

Higher is better, except for timing error.

- **Confidence** (FlatSounds-Physics, 0-1): how often the generated sound behaves as each test expects, for example a higher pitch when the glass is emptied. Each seed counts by how well it reproduces the annotated hits and matches its caption; the score is averaged over tests and test cases. Passing every test can still score below 1 (0.8 with every hit reproduced, no onsets away from the hits and CLAP 0.6).
- **Net confidence** (FlatSounds-Physics): confidence minus how often the sound changes the opposite way. A model that changes the metrics at random scores about 0.
- **Hit F1** (FlatSounds-Single, 0-1): whether the generated hits happen when the video shows them. A generated hit counts if it is within 0.1 s of an annotated hit. F1 combines coverage (the share of annotated hits that are reproduced) and precision (the share of generated hits that are real), so both missing and extra hits lower it. The original recordings reach 0.92; audio that is not synchronized with the video reaches 0.05-0.10.
- **Timing error** (FlatSounds-Single, ms): how far the matched generated hits are from the annotated hits, on average. Hit F1 counts every hit within 0.1 s as correct; timing error tells how close they are. The original recordings reach 17.5 ms.

## FlatSounds-Physics Metrics

| Metric | Captures | Definition | Unit |
|---|---|---|---|
| Attack time | Hardness of the contact | 10-90% rise time of the smoothed (0.5 ms) Hilbert envelope at the hit onset | ms |
| Decay rate | Damping of the struck object | Amplitude decay rate 1/τ from the slope of the hit's noise-compensated energy decay curve over its first 10 dB | 1/s |
| Fundamental frequency (F0) | Pitch or main resonance | Pitch of the played note (SwiftF0) in note sequences (`"f0": "pitch"`); otherwise the lowest resonance (>= 150 Hz) excited by the hit | Hz |
| Spectral centroid | Brightness | Magnitude-weighted mean frequency of the hit (-20 to +180 ms around the onset) after subtracting twice the clip's background power spectrum | Hz |
| Spectral flux | Sharpness of the onset | Mean log-mel spectral flux from 94 ms before to 180 ms after the onset | dB |
| Spectral rolloff | High-frequency content | Frequency below which 85% of the spectral magnitude of the same background-subtracted hit spectrum lies | Hz |
| Temporal modulation | Loudness fluctuation | Amplitude-modulation index of the whole clip's envelope | 0-1 |
| RT60 | Room reverberation | 60 dB divided by the decay slope of the hit's noise-compensated energy decay curve between -5 and -15 dB (T10) | s |
| DRR | Room dryness | Clarity C50 (ISO 3382-1): energy in the first 50 ms after the hit peak over the energy after it | dB |

How the metrics are measured:
- **Hit detection:** hit onsets are detected with librosa and matched to the annotated hits, with a tolerance of 0.10-0.25 s set by the number and spacing of the hits.
- **Pair tests** compare clip values: the mean over the matched hits, each measured at its detected onset (the median for RT60 and DRR).
- **Single-video tests** use every annotated hit: matched hits at their detected onset, the others at the annotated time. Decay rate, RT60 and DRR are not measured on unmatched hits.
- **Quiet hits:** a hit less than 3 dB above the clip background has no spectral centroid or rolloff.

## FlatSounds-Physics Scoring

Each seed of a test gets a weight and a vote:

```
weight      = 0.5 × hit coverage above chance + 0.5 × CLAP score   (a pair takes the worse side of each term)
confidence  = Σ over seeds (weight × vote) / number of seeds  (missing or failed seeds count, with vote 0)
opposite    = the same sum over the votes for the opposite direction (directional tests only)
net         = confidence − opposite                        (0 = chance)
```

Hit coverage above chance is the seed's hit coverage minus the same fraction at times with no annotated hit (a 0.2 s grid, at least the matching tolerance + 0.1 s from every hit), floored at 0. Audio with onsets everywhere matches the hits by chance and earns little; the original recordings reach 0.93 on average.

The CLAP score is the cosine similarity between the generated audio and its caption under `laion/larger_clap_general`, mapped to [0, 1]. A seed votes 1 when:

| Expected | Vote |
|---|---|
| `increase` / `decrease` | counterfactual − factual exceeds τ in the expected direction, with τ = max(2% of the mean, 0.25 × the robust SD) of the seed values of both videos |
| `ascending` / `descending` | the per-hit F0 rises (falls) with hit order: a Spearman correlation beyond ±0.25-0.40, depending on the number of hits (for 2 hits, the sign of the change). A trend the other way is an opposite vote |
| `no_change`, pair | the median over seeds of ln(counterfactual / factual) (the difference for DRR and temporal modulation) lies within the metric's margin, with at least 3 seeds measured on both sides; then every such seed votes |
| `no_change`, single video | the coefficient of variation of the seed's per-hit values (the SD in dB for DRR) is below the metric's threshold. Temporal modulation has no per-hit values, so this test is not scored |

| | Attack | Decay | F0 | Centroid | Flux | Rolloff | Temp. Mod. | RT60 | DRR |
|---|---|---|---|---|---|---|---|---|---|
| Pair margin | 0.15 | 0.39 | 0.23 | 0.083 | 0.12 | 0.06 | 0.085 | 0.21 | 1.2 dB |
| Single-video threshold | 0.30 | 0.25 | 0.05 | 0.10 | 0.25 | 0.10 | not scored | 0.26 | 2.2 dB |

The pair margins are half the median labelled change on the original recordings; the single-video thresholds are at or above the 75th percentile of the spread between repeated hits of the original recordings.

**Aggregation:** overall confidence is the mean over test cases of each test case's mean test confidence (net confidence: of its directional tests); a metric's confidence is the mean over its tests; a test case without generated audio scores 0.

**Alignment and CLAP** (reported next to confidence; the leaderboard shows the `overall` values):
- **Hit coverage:** the share of annotated hits matched by a detected onset, averaged over seeds (a missing or failed seed counts 0).
- **Hit coverage above chance:** as defined above, averaged the same way.
- **Perfect alignment:** the share of seeds with every annotated hit matched.
- **CLAP:** the mean CLAP score of the seeds defined above.

Each is reported for the factual videos, the counterfactual videos, all videos (`overall`) and the worse video of each test case (`min`; the only video of a single-video test case).

## FlatSounds-Single

FlatSounds-Single uses the FlatSounds-Physics onset detector with a stricter threshold (0.3 of the clip's strongest onset instead of 0.07), because hit F1 also counts extra onsets. Audio after the end of the clip is ignored, and the detected onsets are matched one-to-one to the annotated hits within ±0.1 s (`mir_eval.onset.f_measure`).

| Number | Definition |
|---|---|
| **Hit F1** (headline) | F1 of the matched hits per seed, averaged over seeds (a missing seed scores 0), then over clips |
| **Precision** | Matched hits / detected onsets, averaged the same way |
| **Coverage** | Matched hits / annotated hits (recall), averaged the same way |
| **Timing error** | Mean \|detected − annotated\| time of a seed's matched hits, averaged over the seeds with a match (ms) |
| **Timing offset** (diagnostic) | Per annotated hit, the median over seeds of (nearest onset within ±0.25 s − hit time); then the median over hits (ms, positive = audio late) |
| **Timing drift** (diagnostic) | Theil-Sen slope of those per-hit offsets over hit time (ms/s) |

95% confidence intervals bootstrap the 100 caption groups (clips that share a caption show the same scene).
