# Data Pipeline

RST datasets are built in two steps:

```text
1. rst-build-backgrounds   raw HDF5 cadences  →  backgrounds_*.npz  (real, un-injected)
2. rst-build-dataset       backgrounds_*.npz  →  train.npz / val.npz  (synthetic ETI/RFI injected)
```

Both stages are thin CLI wrappers around `rst_seti.data.background_extractor` and `rst_seti.data.cadence_generator` / `rst_seti.data.signal_generator`.

## Step 1 — Background Extraction (`rst-build-backgrounds`)

```bash
rst-build-backgrounds --scan /path/to/h5/files --output data/training --name backgrounds --band 6GHz
```

### Cadence discovery

`DatasetBuilder.scan_directory()` recursively finds `.h5`/`.hdf5` files. `parse_filename()` extracts `(target, obs_type)` from the filename using telescope-specific regexes, tried in order:

- SRT: `(TIC\d+)_(ON|OFF)`
- GBT: `(HIP\d+|GJ\d+[A-Za-z]?)_(ON|OFF)`
- Generic GUPPI/filterbank: `([A-Za-z0-9_]+?)_(ON|OFF)(?:_|\.)`

`group_into_cadences()` groups files by `target + date + parent directory`, sorts each group by the GUPPI timestamp embedded in the filename, and keeps only groups of exactly 6 files matching the `ON-OFF-ON-OFF-ON-OFF` pattern. Each cadence's frequency band is read from the HDF5 header (`fch1`) and matched against `DEFAULT_BAND_CONFIG`:

| Band key | Name | Range (MHz) |
| --- | --- | --- |
| `6GHz` | C-band | 4000–8000 |
| `18GHz` | K-band | 17000–19000 |
| `1.4GHz` | L-band | 1000–2000 |

**Deduplication:** turboSETI re-organizes each cadence's files into `SNR5`/`SNR10`/`SNR20` output folders, which are byte-identical copies. Cadences whose 6 filenames (ignoring parent directory) match an already-seen set are dropped, so the same observation isn't over-represented in the training plate.

### Extraction modes

- **Per-band (default, `--band 6GHz|18GHz|1.4GHz|all`):** processes each band's cadences independently, writing `{name}_{band}.npz`. If `--training-cadences/-t N` is given, the first `N` cadences are used for training and the rest are written to `inference_cadences_{band}.txt` (format `target_name|file1,...,file6`).
- **Mixed multi-band (`--band mixed`):** groups complete cadences into frequency bins of width `--mix-bins` MHz (default 1000), then for each bin samples `--train-fraction` (default 0.5) of cadences for training; the remainder is held out for inference. `--exclude-targets TIC... ...` removes specific targets from training entirely (e.g. a benchmark target), routing them straight to the inference pool. `--seed` makes the split reproducible. Output: `{name}_mixed.npz` plus `inference_cadences_mixed.txt` (format `target_name|freq_start|file1,...,file6` — note the extra `freq_start` field used to disambiguate bands later).
- **`--list-only`:** prints the cadence summary (counts per band, potential snippet totals) and exits without extracting.

The `inference_cadences_*.txt` manifests are consumed directly by `rst-infer -i` (see [docs/inference.md](inference.md)).

### Memory-frugal extraction

`extract_backgrounds()` peeks each file's HDF5 shape via `h5py` metadata only (no data read), truncates every observation to the canonical 16 time bins (raises if any file has fewer), and computes `n_freq` as the minimum channel count across the 6 files. It then reads only the chosen `snippet_width`-wide windows directly from disk (`h5py.File(..., rdcc_nbytes=256*1024*1024)` for chunk-cache reuse across adjacent windows), avoiding a full multi-GB load of all 6 waterfalls. Peak memory is roughly the output array size (`n_snippets × 6 × 16 × snippet_width` float32), not the source files.

`build_training_dataset()` preallocates the full output array up front (`capacity = min(--max-snippets, sum(per-cadence caps))`, where each cadence contributes up to `--snippets-per-cadence` snippets sampled without replacement) and fills it cadence-by-cadence.

`hdf5plugin` is imported (if installed) to register the bitshuffle/LZ4 filters used by Breakthrough Listen-style compressed `.h5` files; without it, `h5py` cannot open the `data` dataset directly.

### Output

- `{name}_{band}.npz` / `{name}_mixed.npz` — key `backgrounds`, shape `(N, 6, 16, snippet_width)` float32, **raw** (un-normalized) power values.
- `{name}_{band}_metadata.json` / `{name}_mixed_metadata.json` — `n_samples`, `n_cadences`, `shape`, `fchans`, `targets`.
- `inference_cadences_{band}.txt` / `inference_cadences_mixed.txt` — held-out cadence manifests for `rst-infer -i`.

## Step 2 — Synthetic Signal Injection & Dataset Building (`rst-build-dataset`)

```bash
rst-build-dataset --backgrounds data/training/backgrounds_6GHz.npz --output data/processed \
    --n-true 30000 --n-false 30000 --val-split 0.15 --seed 42
```

### Disjoint train/val split (snippet-level)

Before any sample generation, the background plate is permuted and split into `plate_train` / `plate_val` (`--val-split`, default 0.15) — **at the snippet level**. `CadenceGenerator(plate=plate_train, seed=seed)` and `CadenceGenerator(plate=plate_val, seed=seed+1)` then draw exclusively from their own pool, so no real-observation snippet feeds both splits (prevents the model from memorizing background texture and inflating validation metrics).

`n_true + n_false` samples are generated and shuffled into `[0, total)`; the first `n_train = total - n_val` indices are routed to `gen_train`, the rest to `gen_val`.

### Cadence composition (`CadenceGenerator` / `CadenceParams`)

**True samples** (`create_true_sample`, label 1):

- **`eti_only_fraction` (default 0.4):** a single ETI signal is injected across the full stacked cadence; the ON scans (0, 2, 4) get the injected signal, OFF scans (1, 3, 5) keep the original background unchanged.
- **Remaining 0.6 — ETI + RFI:** 1–2 RFI signals (`max_disturbance_rfi=2`) are injected across **all 6** observations first, then the ETI signal is injected on top. ON scans get ETI + RFI; OFF scans get RFI only — simulating an ETI signal coexisting with terrestrial interference.

**False samples** (`create_false_sample`, label 0):

1. **Hard-false trap (`hard_false_fraction`):** an ETI-style signal is injected across **all 6 scans with no ON/OFF mask** and labeled False. This forces the model to rely on ON/OFF contrast rather than mere signal presence. Default is `0.2` in the `CadenceParams` dataclass, but `rst-build-dataset`'s `--hard-false-fraction` CLI default is **0.3** and always overrides it.
2. **Pure background:** if not a hard-false trap and `rng > rfi_fraction` (default 0.6), the cadence is returned unmodified.
3. **RFI injection (the remaining ~`rfi_fraction`):** 1–4 RFI signals are injected (count drawn from `rfi_count_weights = (0.4, 0.3, 0.2, 0.1)` → P(1)=0.4 ... P(4)=0.1), each via `inject_rfi_signal`.

### Signal realism (`SignalGenerator` / `SignalParams`)

Instrument constants: `df = 2.7939677238464355` Hz/channel (≈2.79 Hz, native SRT C-band resolution), `dt = 18.25361108` s/time-bin, `tchans_per_obs = 16`.

**SNR convention:** `snr_min`/`snr_max` (default `[5, 50]`, log-uniform sampling — more low-SNR examples, matching the expected real-signal distribution) is the SNR **visible in a single ON scan** (turboSETI convention), not the SNR over the full 96-row stacked frame. Since setigen's `get_intensity()` calibrates over the full frame, intensity is rescaled by `√(96/16) = √6` so the label SNR matches the per-ON-scan visible SNR — applied identically to ETI and RFI injections.

**Drift rate** (`_sample_drift_rate`):

- With probability `zero_drift_prob = 0.05`, the drift is exactly zero (a fully frequency-compensated "beacon").
- Otherwise, `|drift rate|` is sampled from `drift_distribution` (default `'lognormal'`): `log10|DR| ~ Normal(log10(drift_median), drift_log_sigma)` with `drift_median = 0.3` Hz/s (the Earth+exoplanet rotational scale at C-band) and `drift_log_sigma = 0.5` dex (±1σ ≈ [0.095, 0.95] Hz/s), then a random sign is applied.
- The alternative `'loguniform'` distribution samples flat per decade across `[min_nonzero_drift, max_drift_rate]`.
- Both are clipped to `[min_nonzero_drift = 0.01, max_drift_rate]`. `max_drift_rate` defaults to `compute_max_drift_rate(snippet_width=1024, df, dt, n_scans=4, bins_per_scan=16) ≈ 2.45 Hz/s` — the geometric limit for a signal to stay within a 1024-channel snippet across 4 scans (ON-OFF-ON-OFF).
- For ETI signals (when `start_channel` isn't given explicitly), the drift rate is sampled **first**, then `start_channel` is constrained so the signal stays in-bounds through at least time bin 47 (`_MIN_VISIBLE_BIN`) — i.e. visible in at least 2 of the 3 ON windows.

**Signal width:**

- ETI: `width = |drift_rate| × dt + U(1, 10)` Hz — narrowband, plus drift-smearing compensation.
- RFI: `width = |drift_rate| × dt + U(1, 55)` Hz — broader, overlapping the ETI range.

**Frequency profiles (ETI only — `freq_profiles`, weights):**

| Profile | Weight | Notes |
| --- | --- | --- |
| `gaussian` | 0.55 | Standard narrowband shape |
| `sinc2` | 0.10 | Spectral-leakage shape |
| `lorentzian` | 0.20 | Models exo-IPM/ISM scattering wings |
| `voigt` | 0.15 | Gaussian core + Lorentzian (scattering) wings |

For `lorentzian`/`voigt`, an additional scattering FWHM is drawn from `U(scatter_width_min=3, scatter_width_max=40)` Hz and added to the profile width — RFI never uses these profiles since it is local/unscattered.

**Time profiles (`time_profiles`, weights):**

| Profile | Weight | Notes |
| --- | --- | --- |
| `constant` | 0.6 | Flat intensity |
| `scintillating` | 0.4 | Stochastic AR(1) red-noise amplitude modulation |

The scintillating profile (`_make_stochastic_t_profile`) builds a log-normal, unit-mean envelope from an AR(1) process: timescale `τ ~ U(60, 600)` s, depth `~ U(0.2, 0.6)`, `ρ = exp(-dt/τ)`. A clean periodic sinusoid was deliberately removed — regular periodicity is now a property of the `pulsed` RFI type only, so the model can't use periodicity itself as an ETI fingerprint.

**RFI taxonomy (`rfi_types`, weights):**

| Type | Weight | Description |
| --- | --- | --- |
| `linear` | 0.28 | Same drift model as ETI, but injected into all observations |
| `stationary` | 0.12 | Fixed frequency with Gaussian jitter (`simple_rfi_path`) |
| `random_walk` | 0.17 | Frequency wanders randomly over time |
| `scintillating` | 0.13 | Linear drift + AR(1) amplitude modulation |
| `broadband` | 0.18 | Wide-band (`U(400, 2000)` Hz), near-stationary terrestrial RFI |
| `pulsed` | 0.12 | Periodic on/off pulses (`periodic_gaussian_t_profile`, period `U(40,120)` s) — radar/beacon-like |

### `rst-build-dataset` CLI reference

| Flag | Default | Meaning |
| --- | --- | --- |
| `--backgrounds/-b` | required | Path to `backgrounds_*.npz` |
| `--output/-o` | `data/processed` | Output directory |
| `--n-true` / `--n-false` | 30000 / 30000 | Number of True / False samples |
| `--val-split` | 0.15 | Validation fraction (snippet-level, disjoint) |
| `--seed` | random | RNG seed (logged if auto-generated) |
| `--fchans` | 1024 | Frequency channels per snippet |
| `--snr-min` / `--snr-max` | 5 / 50 | Log-uniform per-ON-scan SNR range |
| `--eti-only-fraction` | 0.4 | Fraction of True samples that are ETI-only |
| `--rfi-fraction` | 0.6 | Fraction of False samples (after hard-false) with injected RFI |
| `--hard-false-fraction` | 0.3 | Fraction of False samples that are hard-false traps |
| `--drift-distribution` | `lognormal` | `lognormal` or `loguniform` |
| `--drift-median` | 0.3 | Log-normal drift median (Hz/s) |
| `--drift-log-sigma` | 0.5 | Log-normal drift spread (dex) |

### Output

- `train.npz` / `val.npz` — keys `spectrograms` (N, 96, 1024) float32 and `labels` (N,) binary.
- `generation_metadata.json` — seed, sample counts, SNR/drift configuration, RFI/profile taxonomies, and background split sizes (`n_backgrounds_train`/`n_backgrounds_val`).

## Shared preprocessing

Both `rst-build-dataset` (via `stack_cadence`) and inference (via `preprocess_cadence`, in [docs/inference.md](inference.md)) rely on `rst_seti.data.preprocessing`:

- **`extract_snippet(cadence, center_channel, snippet_width)`** — crops `(6, 16, n_freq)` → `(6, 16, snippet_width)`, shifting the window inward at the array edges.
- **`stack_cadence(cadence)`** — reshapes `(6, 16, width)` → `(96, width)`.
- **`normalize_robust(spec)`** — `log10` (after clipping to `≥1e-6`), then per-observation Z-score (mean/std computed independently over each of the 6×16-row blocks, divided by `2σ`), then clipped to `[-5, 5]`.
