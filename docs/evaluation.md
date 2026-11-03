# Evaluation Options and Outputs

## Options

| Option | Default | Meaning |
|---|---|---|
| `--generated_audio_dir` | | Directory with the generated audio (required) |
| `--split` | `verified` | FlatSounds-Physics only: `verified` or `paper` |
| `--metadata_path` | Physics: set by `--split`; Single: `benchmark/single.json` | Custom metadata JSON (Physics: overrides `--split`) |
| `--num_runs` | 10 | Seeds per video |
| `--sample_rate` | 16000 | Analysis sample rate in Hz; the margins and thresholds are calibrated at 16000, so keep the default for reported results |
| `--n_jobs` | -1 | Worker processes for the acoustic metrics (-1 = all CPUs) |
| `--output_dir` | `evaluation_results/<generated_audio_dir name>` | Results directory |
| `--clap_cache_dir` | Hugging Face cache | FlatSounds-Physics only: Hugging Face cache directory for the CLAP model |

FlatSounds-Physics also scores every file with the CLAP model `laion/larger_clap_general` (about 0.8 GB, downloaded on the first run), on a GPU if PyTorch can use one (the locked Linux build needs NVIDIA driver 580 or newer), otherwise more slowly on the CPU. Scoring one model with 10 seeds takes a few minutes on a GPU machine with many CPU cores; fewer cores take longer.

## Outputs

Each evaluation writes one JSON file to the results directory and prints a summary table. In the JSON, `null` marks a value that is undefined or could not be measured.

**`physics_results.json`**

| Key | Content |
|---|---|
| `meta` | `benchmark`, `split`, `metadata_md5`, `code_version` (`git describe`), `num_runs`, `sample_rate`, `n_cases`, `n_records`, `missing_cases` (test cases missing all generated audio of at least one video), `n_unreadable_files` (files that exist but could not be used; see [Generated Audio](data.md#generated-audio)) |
| `summary.overall` | `confidence` (the headline number) and `net_confidence` |
| `summary.metrics.<metric>` | `confidence`, `net_confidence` and `n_tests` per acoustic metric |
| `summary.hit_coverage`, `summary.hit_coverage_above_chance`, `summary.perfect_alignment`, `summary.clap` | `overall`, `factual`, `counterfactual` and `min`; see [Alignment and CLAP](metrics.md#flatsounds-physics-scoring) |
| `metric_definitions` | One sentence per metric and scoring rule |
| `results` | One record per test: `case`, `category`, `evaluation_method`, `metric`, `expected`, `confidence`, `opposite_confidence`, `net_confidence`, `n_valid`, `n_total`, `factual_mean`, `counterfactual_mean`, `delta`, the test case's mean confidence and net confidence, hit coverage and timing error (s) of each side, and the two video names |

**`single_results.json`**

| Key | Content |
|---|---|
| `meta` | `benchmark`, `metadata`, `metadata_md5`, `code_version`, `num_runs`, `sample_rate`, `n_clips`, `n_caption_groups`, `n_records`, `n_missing_files`, `n_unreadable_files` (files that exist but could not be used; see [Generated Audio](data.md#generated-audio)) |
| `summary.<number>` | `value` and `ci95` of `hit_f1` (the headline number), `hit_precision`, `hit_coverage`, `timing_error_ms`, `timing_offset_ms` and `timing_drift_ms_per_s` |
| `metric_definitions` | One sentence per number |
| `results` | One record per clip and seed: `clip`, `seed`, `file`, `n_hits`, `n_detected`, `n_matched`, `hit_f1`, `hit_precision`, `hit_coverage`, `timing_error_ms` and `hit_offsets_ms` (one offset per annotated hit) |

To load the records with pandas (`uv pip install pandas`):

```python
import json
import pandas as pd

with open("evaluation_results/your_model_output/physics_results.json") as f:
    data = json.load(f)
print(data["summary"]["overall"])
df = pd.DataFrame(data["results"])
print(df.groupby("metric")[["confidence", "net_confidence"]].mean())
```
