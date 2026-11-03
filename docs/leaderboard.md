# Leaderboard

Video-to-audio models on both evaluations, 10 seeds per video. **Caption**: ✓ the model also received the caption of each video,
✗ video only, – the model takes no text. Model names link to their papers. Higher is better, except timing error. [Metrics and Scoring](metrics.md) defines every column.

At a glance: FlatSounds-Physics on the verified split and FlatSounds-Single, sorted by Physics confidence.

| Model | Caption | Physics Confidence ↑ | Physics Net Confidence ↑ | Single Hit F1 ↑ | Single Timing Error (ms) ↓ |
|---|---|---|---|---|---|
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.369 | +0.183 | 0.555 | 42.8 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.334 | +0.130 | 0.349 | 62.6 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.332 | +0.155 | 0.585 | 41.9 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.332 | +0.086 | 0.550 | 49.9 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.322 | +0.084 | 0.742 | 37.4 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.310 | +0.083 | 0.386 | 61.2 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.287 | +0.058 | 0.435 | 47.1 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.286 | +0.038 | 0.541 | 50.3 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.265 | +0.026 | 0.465 | 45.8 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.207 | +0.054 | 0.174 | 62.2 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.187 | +0.025 | 0.182 | 60.2 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.156 | +0.064 | 0.298 | 54.9 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.147 | +0.040 | 0.280 | 54.3 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.129 | +0.055 | 0.111 | 51.2 |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.116 | +0.012 | 0.166 | 48.1 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.113 | +0.030 | 0.085 | 50.6 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.099 | +0.023 | 0.103 | 49.6 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.076 | +0.017 | 0.075 | 50.5 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.068 | +0.010 | 0.070 | 50.2 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.054 | -0.008 | 0.067 | 48.4 |

## FlatSounds-Physics

### Verified Split

The default test set. Overall confidence and net confidence (0 = chance), and the hit coverage (also above chance), perfect alignment and CLAP of all videos:

| Model | Caption | Confidence ↑ | Net Confidence ↑ | Hit Coverage (%) ↑ | Above Chance (%) ↑ | Perfect Align (%) ↑ | CLAP ↑ |
|---|---|---|---|---|---|---|---|
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.369 | +0.183 | 96.4 | 64.9 | 92.6 | 0.625 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.334 | +0.130 | 84.0 | 64.7 | 76.7 | 0.627 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.332 | +0.155 | 98.2 | 57.5 | 96.4 | 0.608 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.332 | +0.086 | 92.3 | 77.2 | 85.8 | 0.606 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.322 | +0.084 | 94.0 | 81.2 | 90.3 | 0.600 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.310 | +0.083 | 84.0 | 67.1 | 76.8 | 0.594 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.287 | +0.058 | 89.3 | 57.2 | 80.3 | 0.630 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.286 | +0.038 | 91.4 | 77.6 | 83.7 | 0.541 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.265 | +0.026 | 91.2 | 60.2 | 83.0 | 0.590 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.207 | +0.054 | 69.9 | 47.4 | 52.0 | 0.641 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.187 | +0.025 | 72.2 | 46.1 | 55.4 | 0.612 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.156 | +0.064 | 62.3 | 38.6 | 47.5 | 0.573 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.147 | +0.040 | 68.4 | 38.0 | 52.7 | 0.547 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.129 | +0.055 | 76.4 | 12.8 | 60.4 | 0.610 |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.116 | +0.012 | 62.7 | 30.5 | 45.8 | 0.536 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.113 | +0.030 | 56.5 | 12.8 | 38.0 | 0.646 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.099 | +0.023 | 83.9 | 8.3 | 71.8 | 0.545 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.076 | +0.017 | 46.5 | 12.0 | 30.0 | 0.612 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.068 | +0.010 | 60.7 | 10.1 | 45.7 | 0.553 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.054 | -0.008 | 60.0 | 10.0 | 44.5 | 0.520 |

Confidence per metric (number of tests in parentheses) and overall (mean over test cases):

| Model | Caption | Attack (53) | Decay (119) | F0 (85) | Centroid (157) | Flux (37) | Rolloff (67) | Temp. Mod. (26) | RT60 (110) | DRR (70) | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.242 | 0.407 | 0.251 | 0.413 | 0.280 | 0.371 | 0.154 | 0.479 | 0.505 | **0.369** |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.261 | 0.365 | 0.325 | 0.291 | 0.227 | 0.224 | 0.255 | 0.423 | 0.438 | **0.334** |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.200 | 0.396 | 0.278 | 0.332 | 0.268 | 0.324 | 0.213 | 0.402 | 0.435 | **0.332** |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.257 | 0.343 | 0.233 | 0.273 | 0.326 | 0.243 | 0.245 | 0.433 | 0.424 | **0.332** |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.244 | 0.317 | 0.230 | 0.332 | 0.316 | 0.279 | 0.277 | 0.387 | 0.444 | **0.322** |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.229 | 0.330 | 0.311 | 0.259 | 0.214 | 0.188 | 0.253 | 0.402 | 0.404 | **0.310** |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.235 | 0.314 | 0.250 | 0.267 | 0.290 | 0.259 | 0.179 | 0.370 | 0.349 | **0.287** |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.274 | 0.307 | 0.182 | 0.267 | 0.332 | 0.234 | 0.235 | 0.349 | 0.361 | **0.286** |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.223 | 0.288 | 0.213 | 0.231 | 0.294 | 0.218 | 0.248 | 0.351 | 0.322 | **0.265** |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.188 | 0.213 | 0.197 | 0.184 | 0.235 | 0.163 | 0.223 | 0.231 | 0.211 | **0.207** |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.161 | 0.174 | 0.175 | 0.181 | 0.199 | 0.161 | 0.194 | 0.203 | 0.185 | **0.187** |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.129 | 0.131 | 0.114 | 0.154 | 0.176 | 0.154 | 0.233 | 0.147 | 0.121 | **0.156** |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.125 | 0.149 | 0.121 | 0.154 | 0.209 | 0.137 | 0.189 | 0.132 | 0.117 | **0.147** |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.102 | 0.102 | 0.081 | 0.140 | 0.173 | 0.151 | 0.233 | 0.145 | 0.103 | **0.129** |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.112 | 0.106 | 0.108 | 0.119 | 0.123 | 0.086 | 0.143 | 0.134 | 0.118 | **0.116** |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.101 | 0.075 | 0.080 | 0.110 | 0.154 | 0.126 | 0.138 | 0.139 | 0.063 | **0.113** |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.094 | 0.079 | 0.068 | 0.114 | 0.148 | 0.109 | 0.128 | 0.095 | 0.075 | **0.099** |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.060 | 0.052 | 0.074 | 0.080 | 0.112 | 0.089 | 0.097 | 0.058 | 0.055 | **0.076** |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.059 | 0.040 | 0.082 | 0.061 | 0.109 | 0.062 | 0.107 | 0.048 | 0.060 | **0.068** |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.055 | 0.030 | 0.066 | 0.044 | 0.105 | 0.033 | 0.063 | 0.042 | 0.052 | **0.054** |

### Paper Split

The test set as published (`--split paper`), for comparison; we recommend the verified split. Here piano and guitar notes are measured by their lowest resonance, not the played pitch ([F0](data.md#flatsounds-physics-test-cases)), and the 3 single-video temporal modulation tests are not scored.

| Model | Caption | Confidence ↑ | Net Confidence ↑ | Hit Coverage (%) ↑ | Above Chance (%) ↑ | Perfect Align (%) ↑ | CLAP ↑ |
|---|---|---|---|---|---|---|---|
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.340 | +0.175 | 96.7 | 60.6 | 93.3 | 0.625 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.302 | +0.083 | 93.9 | 80.4 | 90.4 | 0.599 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.297 | +0.122 | 85.2 | 63.9 | 79.4 | 0.624 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.293 | +0.071 | 92.7 | 76.1 | 87.9 | 0.601 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.291 | +0.127 | 98.6 | 53.7 | 97.4 | 0.608 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.269 | +0.051 | 92.0 | 77.7 | 86.3 | 0.542 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.266 | +0.071 | 85.4 | 66.1 | 79.5 | 0.590 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.256 | +0.050 | 90.0 | 55.7 | 82.9 | 0.633 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.246 | +0.044 | 91.6 | 59.2 | 84.9 | 0.592 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.203 | +0.051 | 73.8 | 49.7 | 59.0 | 0.642 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.187 | +0.021 | 76.0 | 48.7 | 62.2 | 0.614 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.161 | +0.055 | 67.3 | 41.8 | 54.8 | 0.572 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.154 | +0.028 | 75.6 | 41.1 | 62.2 | 0.548 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.118 | +0.040 | 74.8 | 13.7 | 61.5 | 0.612 |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.110 | -0.005 | 64.4 | 31.9 | 50.4 | 0.538 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.094 | +0.019 | 55.2 | 13.4 | 39.8 | 0.647 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.093 | +0.007 | 84.1 | 8.8 | 74.1 | 0.547 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.075 | +0.019 | 46.3 | 12.3 | 32.7 | 0.610 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.065 | +0.004 | 60.9 | 10.6 | 49.3 | 0.555 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.058 | -0.004 | 62.6 | 10.6 | 49.8 | 0.522 |

| Model | Caption | Attack (91) | Decay (94) | F0 (69) | Centroid (171) | Flux (43) | Rolloff (64) | Temp. Mod. (19) | RT60 (35) | DRR (31) | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.187 | 0.385 | 0.276 | 0.399 | 0.247 | 0.357 | 0.174 | 0.544 | 0.491 | **0.340** |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.206 | 0.306 | 0.226 | 0.337 | 0.310 | 0.310 | 0.285 | 0.354 | 0.410 | **0.302** |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.205 | 0.339 | 0.312 | 0.290 | 0.218 | 0.234 | 0.210 | 0.369 | 0.310 | **0.297** |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.239 | 0.308 | 0.230 | 0.271 | 0.279 | 0.221 | 0.227 | 0.477 | 0.421 | **0.293** |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.174 | 0.355 | 0.250 | 0.315 | 0.232 | 0.309 | 0.211 | 0.415 | 0.395 | **0.291** |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.250 | 0.283 | 0.219 | 0.258 | 0.258 | 0.208 | 0.222 | 0.332 | 0.307 | **0.269** |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.198 | 0.295 | 0.294 | 0.255 | 0.203 | 0.179 | 0.205 | 0.329 | 0.310 | **0.266** |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.207 | 0.291 | 0.237 | 0.265 | 0.225 | 0.246 | 0.170 | 0.332 | 0.273 | **0.256** |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.192 | 0.284 | 0.228 | 0.237 | 0.236 | 0.227 | 0.215 | 0.289 | 0.237 | **0.246** |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.175 | 0.224 | 0.221 | 0.186 | 0.196 | 0.146 | 0.196 | 0.222 | 0.189 | **0.203** |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.153 | 0.195 | 0.208 | 0.184 | 0.157 | 0.144 | 0.146 | 0.189 | 0.163 | **0.187** |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.154 | 0.155 | 0.134 | 0.164 | 0.157 | 0.133 | 0.205 | 0.116 | 0.115 | **0.161** |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.144 | 0.171 | 0.143 | 0.157 | 0.177 | 0.116 | 0.133 | 0.119 | 0.106 | **0.154** |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.104 | 0.082 | 0.092 | 0.130 | 0.148 | 0.122 | 0.183 | 0.162 | 0.143 | **0.118** |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.108 | 0.105 | 0.112 | 0.109 | 0.117 | 0.065 | 0.129 | 0.115 | 0.112 | **0.110** |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.077 | 0.055 | 0.087 | 0.100 | 0.130 | 0.097 | 0.129 | 0.147 | 0.082 | **0.094** |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.079 | 0.076 | 0.070 | 0.109 | 0.135 | 0.091 | 0.139 | 0.099 | 0.081 | **0.093** |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.049 | 0.041 | 0.075 | 0.075 | 0.088 | 0.076 | 0.118 | 0.086 | 0.093 | **0.075** |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.051 | 0.032 | 0.073 | 0.051 | 0.108 | 0.049 | 0.107 | 0.065 | 0.090 | **0.065** |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.059 | 0.029 | 0.074 | 0.040 | 0.085 | 0.026 | 0.066 | 0.043 | 0.072 | **0.058** |

## FlatSounds-Single

Hit F1 with its 95% confidence interval, timing error, precision and coverage:

| Model | Caption | Hit F1 ↑ | Timing Error (ms) ↓ | Precision (%) ↑ | Coverage (%) ↑ |
|---|---|---|---|---|---|
| Original recordings | | 0.924 [0.882, 0.959] | 17.5 | 91.7 | 94.9 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✗ | 0.742 [0.695, 0.789] | 37.4 | 73.7 | 78.4 |
| [Cosmos3 Nano](https://arxiv.org/abs/2606.02800) | ✓ | 0.585 [0.526, 0.636] | 41.9 | 52.9 | 78.0 |
| [Cosmos3 Super](https://arxiv.org/abs/2606.02800) | ✓ | 0.555 [0.494, 0.616] | 42.8 | 51.4 | 73.1 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✓ | 0.550 [0.501, 0.598] | 49.9 | 54.4 | 59.2 |
| [PrismAudio](https://arxiv.org/abs/2511.18833) | ✗ | 0.541 [0.494, 0.591] | 50.3 | 53.6 | 57.8 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✗ | 0.465 [0.415, 0.513] | 45.8 | 44.2 | 55.2 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | ✓ | 0.435 [0.388, 0.483] | 47.1 | 39.8 | 55.3 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✗ | 0.386 [0.333, 0.442] | 61.2 | 37.5 | 42.6 |
| [Omni2Sound](https://arxiv.org/abs/2601.02731) | ✓ | 0.349 [0.297, 0.405] | 62.6 | 33.4 | 39.7 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✓ | 0.298 [0.249, 0.351] | 54.9 | 29.1 | 34.6 |
| [ThinkSound](https://arxiv.org/abs/2506.21448) | ✗ | 0.280 [0.233, 0.331] | 54.3 | 27.2 | 33.4 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✗ | 0.182 [0.143, 0.226] | 60.2 | 17.5 | 22.0 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | ✓ | 0.174 [0.133, 0.218] | 62.2 | 16.5 | 21.1 |
| [Frieren](https://arxiv.org/abs/2406.00320) | – | 0.166 [0.145, 0.188] | 48.1 | 15.2 | 22.6 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✓ | 0.111 [0.095, 0.126] | 51.2 | 8.1 | 24.8 |
| [FoleyCrafter](https://arxiv.org/abs/2407.01494) | ✗ | 0.103 [0.090, 0.116] | 49.6 | 7.5 | 26.1 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✓ | 0.085 [0.075, 0.095] | 50.6 | 6.8 | 17.7 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✓ | 0.075 [0.064, 0.086] | 50.5 | 6.6 | 13.8 |
| [Audio-Omni](https://arxiv.org/abs/2604.10708) | ✗ | 0.070 [0.060, 0.082] | 50.2 | 6.2 | 12.5 |
| [AudioX](https://arxiv.org/abs/2503.10522) | ✗ | 0.067 [0.057, 0.078] | 48.4 | 5.4 | 13.7 |

## Submitting Results

To add a model, open an issue in this repository with:
- the model name, a link to its paper or code, and whether it received the captions;
- `physics_results.json` (verified split) and `single_results.json`, written by the evaluation with the default options;
- a download link to the generated audio for all videos and 10 seeds, which we score again before adding the model.
