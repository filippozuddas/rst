# Inference

```bash
rst-infer -s /path/to/observations/                                  # auto-download weights, scan a directory
rst-infer -m checkpoints/best.pth -f ON1.h5 OFF1.h5 ON2.h5 OFF2.h5 ON3.h5 OFF3.h5  # single cadence, local weights
rst-infer -i data/training/inference_cadences_mixed.txt              # manifest from rst-build-backgrounds
rst-infer --list-models                                               # show available Hub model versions
```

`rst-infer` (`src/rst_seti/cli/infer.py`) wraps `InferenceEngine` (`src/rst_seti/inference/engine.py`) with cadence loading (via `blimpy`), CSV/plot/attention-map output, and three input modes.

## Model resolution

- **`--model/-m PATH`** — load a local `.pth` checkpoint, paired with `--config` (default: bundled `default.yaml`).
- **default (no `--model`)** — `ModelHub.resolve(model_name=args.model_name or "rst-base384-v1")` downloads (or loads from `~/.cache/rst-seti/models/`) both the checkpoint and its matching config from the `filippozuddas/rst-seti` Hugging Face Hub repo. A `.meta.json` records the architecture config used at training time so cached weights aren't loaded against an incompatible config; if offline, the cache is used directly.
- **`--list-models`** — prints `ModelHub.MODEL_REGISTRY` (currently `rst-base384-v1`) and exits.

`InferenceEngine.__init__` always builds the model with `imagenet_pretrain=False` (weights come entirely from the checkpoint) and strips a `module.` prefix from `DataParallel`-trained state dicts.

## Hardware auto-detection

`HardwareManager.detect(preferred_device, preferred_batch_size)` (`src/rst_seti/hardware.py`) picks:

- **Device:** `cuda` if available, else `cpu` (overridable with `--device cpu|cuda|cuda:1`).
- **dtype:** `bfloat16` on CUDA devices with compute capability ≥ 8.0 (Ampere+), `float16` on older CUDA, `float32` on CPU.
- **Batch size** (if not given via `--batch-size` or `inference.batch_size` in config): from a VRAM lookup table — 48 GB → 256, 24 GB → 128, 10 GB → 64, 6 GB → 32, < 4 GB → 16; CPU defaults to 16.

## Sliding-window extraction

`InferenceEngine._extract_snippets(cadence)` slides a `snippet_width`-wide window (default 1024 channels, from `data.snippet_width`) across the full `(6, 16, n_freq)` cadence with step `sliding_step` (default 512 = 50% overlap, from `inference.sliding_window_step`). Centers run from `half = snippet_width // 2` to `n_freq - half`; an extra final window at `end_center` is appended if the last stepped center didn't reach it, so the high-frequency edge of the band is never skipped. Each window is independently preprocessed via `preprocess_cadence()` (`extract_snippet` → `stack_cadence` → `normalize_robust`, see [docs/data_pipeline.md](data_pipeline.md)) — i.e. each `(96, 1024)` snippet is normalized using its own statistics.

## Batch inference

`_batch_inference()` runs snippets through the model in chunks of `chunk_size = 1024` (outer loop in `run_cadence`) further split into `batch_size` mini-batches:

- **NaN/Inf guard:** any non-finite values in a batch (can occur with real SRT data) are replaced with `0.0` via `torch.nan_to_num`, with a warning printed.
- **CUDA OOM recovery:** on `torch.cuda.OutOfMemoryError`, `batch_size` is halved (`torch.cuda.empty_cache()`), and the batch is retried; raises if `batch_size` would drop below 1.
- Mixed precision (`autocast`) is **not** used at inference time — disabled to avoid NaNs from overflow on real SRT data.

Output: a flat array of `P(ETI)` probabilities, one per snippet.

## Clustering (`cluster_detections`)

Per-snippet results (`center_channel`, `probability`, `classification = 'ETI' if p >= threshold else 'RFI'`, `freq_mhz`) are filtered to `classification == 'ETI'`, sorted by `center_channel`, and grouped via 1D connected-component grouping: a gap between consecutive `center_channel`s `> sliding_step` starts a new cluster (i.e. only adjacent/overlapping sliding-window detections merge). Each cluster row reports:

| Column | Meaning |
| --- | --- |
| `center_channel` | Channel of the **peak**-probability snippet in the cluster |
| `peak_probability` | Max `P(ETI)` in the cluster |
| `mean_probability` | Mean `P(ETI)` across the cluster's snippets |
| `cluster_width` | `(last - first center_channel) + snippet_width` |
| `n_snippets` | Number of snippets in the cluster |
| `freq_mhz` | Frequency at the peak channel (`0` if the cadence had no header frequency info) |

The cluster table is sorted by `peak_probability` descending.

## Input modes & output directory layout

All modes write under `--output` (default `results/`), and finish with global `results_summary.csv` (all per-snippet results, all cadences) and `clusters_summary.csv` (all clusters, all cadences) if any cadence produced results.

### 1. Single cadence (`--files/-f FILE1 ... FILE6`)

Loads exactly 6 HDF5 files in ON/OFF order via `load_cadence_from_files()` (uses `blimpy.Waterfall`, truncates every observation to 16 time bins, raises if any has fewer). The target name is parsed from the first filename using the same `TIC\d+`/`HIP\d+|GJ\d+`/generic regexes as `background_extractor.parse_filename`. Output directory: **`{output}/{target}/`**.

### 2. Directory scan (`--scan/-s DIR [--band 6GHz|18GHz|1.4GHz|all]`)

Uses `DatasetBuilder.scan_directory()` + `group_into_cadences()` (same logic as `rst-build-backgrounds`, including dedup of turboSETI SNR-folder copies) to find complete 6-file cadences, optionally filtered by `--band`. Output directory per cadence: **`{output}/{target}_{date}/`**, where `date` is the GUPPI-timestamp date parsed from the filename. Each row in `results_summary.csv`/`clusters_summary.csv` additionally carries `target`, `date`, `freq_band`.

### 3. Inference-cadences manifest (`--inference-cadences/-i FILE.txt`)

Parses a manifest produced by `rst-build-backgrounds` (`parse_inference_cadences_file()`), which accepts either format:

- `target_name|freq_start|file1,file2,...,file6` (mixed-band manifests)
- `target_name|file1,file2,...,file6` (per-band manifests)

(lines with neither 2 nor 3 `|`-fields, or not exactly 6 files, are skipped with a warning).

For each entry, the output directory is **`{output}/{target}_{freq_tag}_{date_tag}/`**, where:

- `freq_tag` = `{freq_start_mhz:.0f}MHz` from the HDF5 header, or `nofreq` if unavailable.
- `date_tag` = the date from `DatasetBuilder.parse_filename()` on the first file, or `nodate`.
- If that directory already exists (e.g. two cadences for the same target/band/date), a numeric suffix `_2`, `_3`, ... is appended until unique.

Each row in `results_summary.csv`/`clusters_summary.csv` additionally carries `target` and `cadence_id` (the resolved directory name).

## Per-cadence outputs

Written by `process_cadence()` into each cadence's output directory:

- **`cadence_results.csv`** — every snippet: `center_channel`, `probability`, `classification`, `freq_mhz`.
- **`cadence_clusters.csv`** — cluster table (see above), only if non-empty.
- **`plots/cluster_ch{center:05d}_{freq}_p{prob:.2f}_n{n_snippets}.png`** — one plot per cluster (up to `--top-n`, default 200; `0` = all), via `plot_candidate()`. `{freq}` is `{freq_mhz:.2f}MHz` or `nofreq`.
- **`attention_maps/cluster_ch{center:05d}_{freq}_p{prob:.2f}_attn.png`** — only for clusters with `peak_probability >= attn_threshold` (default `0.9`), via `AttentionExtractor` + `plot_attention_map()`.

Both plot types are skipped entirely with `--no-plots`.

## CLI reference

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model/-m` | none (Hub download) | Local checkpoint path |
| `--model-name` | `rst-base384-v1` | Hub model version (mutually exclusive with `--model`) |
| `--files/-f` (×6) / `--scan/-s` / `--inference-cadences/-i` / `--list-models` | — | Mutually exclusive input modes |
| `--config/-c` | bundled `default.yaml` | Architecture/inference config |
| `--output/-o` | `results/` | Output root directory |
| `--batch-size/-b` | auto (VRAM-based) | Inference batch size |
| `--threshold/-t` | `0.9` (config `inference.threshold`) | ETI classification threshold |
| `--attn-threshold` | `0.9` (config `inference.attn_threshold`) | Attention-map generation threshold |
| `--device` | auto | `cpu` / `cuda` / `cuda:1` |
| `--band` | `all` | Frequency-band filter, `--scan` mode only |
| `--no-plots` | off | Disable plot/attention-map generation |
| `--top-n/-n` | `200` | Max plots+attention maps per cadence (`0` = all) |
