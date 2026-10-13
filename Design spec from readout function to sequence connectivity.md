# Design spec: from readout function to sequence connectivity

Oct 13, 2026 · @max

## 1. Purpose and how to use this spec

This spec defines one codebase, a preliminary package (WP-T) that rebuilds the thesis framework, and seven work packages (WP0–WP6). Together they reproduce Braun et al. (2026) from its Methods, then test whether the Wiener-optimal kernel from Tepper (2026, MSc thesis), mapped through the Braun et al. support condition, predicts sequence connectivity and can be learned from a readout signal.

The reader is a coding agent (Claude Code) that will implement everything. The scientific owner is Max Tepper. All source PDFs listed in Section 2 are available to future sessions; this spec cites them by section and equation number so every formula can be checked against its source.

**Rules for the implementing agent**

1. Read Sections 2–6 before writing code. Build Section 6A (WP-T) first; Sections 7–13 are the experiments; Section 14 is the test suite; Section 15 lists ambiguities.
2. Every item marked **\[VERIFY\]** is an ambiguity in a source (garbled PDF extraction, unstated detail, or inconsistency). Implement the stated default, expose the alternative as a config switch, log which one ran, and never resolve it silently.
3. Every function that implements a formula carries a docstring citing its source, e.g. `Braun Suppl. Cor. 3.2` or `Thesis eq. (7.13)`.
4. No backpropagation through time and no autodiff-based training anywhere. Training uses recursive least squares (FORCE/RLS), delta rules, or the thesis's local rules only. This is a design decision, not a performance shortcut: the scientific claim concerns low-rank updates driven by visited activity. Autodiff is allowed only for diagnostics (Jacobians, Floquet multipliers).
5. Implement the analytic tests in Section 14 before any WP that depends on them. A WP is done only when its acceptance criteria are reported as pass or fail with numbers.
6. Each WP writes `results/wpX/REPORT.md` with figures, a table of acceptance criteria, every \[VERIFY\] switch used, and deviations from this spec.

**Out of scope:** spiking models and spiking implementations of any WP.

## 2. Source documents

Three documents are authoritative: the Braun et al. main text with Methods, its analytical supplement, and the thesis. Everything else is supporting. When this spec and a source disagree, the source wins and the disagreement is reported.

| Short name | Full reference | File | Authoritative for |
| --- | --- | --- | --- |
| Braun | Braun, Nordal, Rambø, Clopath, Gonzalo Cogno (2026). A common structure in recurrent networks supports neural sequence generation locally and in downstream neurons. bioRxiv 10.64898/2026.09.02.748960, v1 posted 8 Sep 2026 | `2026_09_02_748960v1_full.pdf` | Model, targets, FORCE variant, all analyses and figure parameters (Methods, lines 764–1299 of the PDF; figure legends lines 406–763) |
| Braun Suppl. | Supplementary Material: Analytical Derivations (§1–§7) | `media-1.pdf` | Support condition (Def. 1.1), Fourier cutoff and aliasing (§2), circulant sufficiency (Prop. 3.1, Cor. 3.2), degeneracy (Prop. 4.1), ridge shrinkage (§5), parameter dependence (§6), scaffolding bound (§7) |
| Thesis | Tepper (2026). Connecting Manifold Rank to Computational Capacity through Structured Kernel Inhibition. MSc thesis, Göttingen, submitted 30 Apr 2026 | `main.pdf` (201 pp.) | Readout model (§2–3), Fourier diagonalization (§4), error decomposition (§6), Wiener-optimal kernel (§7, eq. 7.11–7.13), joint optimum (§8), kernel learning rule (§9, eq. 9.10), SOM chain (eq. 9.14–9.16, App. K.2), E/I (App. F), heterogeneity (App. E, M), worked examples (App. I) |
| MEC data | Gonzalo Cogno et al. (2024). Minute-scale oscillatory sequences in medial entorhinal cortex. Nature 625:338–344 | **\[DATA\]** path to be provided by Max | Session 7 of mouse #60584 (as used in Braun), neuron ordering by the PCA method |
| Bat data | Omer, Las, Ulanovsky (2023). Nat Neurosci 26:285–294 | **\[DATA\]** path to be provided | CA1 time cells of the observer bat, 168 cells, first 6 s after landing |
| FORCE | Sussillo & Abbott (2009) Neuron 63:544; Rajan, Harvey & Tank (2016) Neuron 90:128 | not provided | Background for the RLS variant; Braun Methods eqs. 7–9 are authoritative |

**PDF extraction warnings.** Text extraction of the Braun PDF garbles several Methods equations: the noise term (eq. 2), the motif formulas (eq. 14–15), and the downstream target bin size. Section 15 lists each with a reconstruction. Rasterize the relevant PDF page to check any formula before relying on it.

**Data status.** Braun et al. have not released code. Max holds permission and a copy of the MEC data. The bat data are public. WP0.7, WP5 and WP6 are blocked until data paths are given; all other WPs use synthetic data only.

## 3. Scientific background: the combined chain

The thesis says which bump shape a readout needs; Braun Suppl. says which connectivity a given bump shape forces. Composed, they predict connectivity from function. This section states both results in the form the code needs, then lists the hypotheses the WPs test.

### 3.1 Braun et al.: shape determines connectivity

The network is a rate RNN (Braun Methods eq. 1):

```latex
\tau \dot h_i = -h_i + \sum_j J_{ij}\,\phi(h_j) + \eta_i, \qquad \phi(h) = \frac{1}{1+e^{-h}}
```

Every neuron gets the same target rate time course, delayed by its position: f\_n(t) = a g(t − μ\_n) + b with μ\_n = nT/N and g a wrapped Gaussian of width σ (Methods eq. 4–5; Suppl. eq. 1). The population activity is a bump travelling around a ring of neurons once per period T.

**Support condition** (Suppl. Def. 1.1). W supports the sequence if it delivers the required input at all times:

```latex
W f(t) = x^\star(t) + \tau \dot x^\star(t) \equiv y^\star(t), \qquad x^\star = \phi^{-1}(f)
```

**Circulant sufficiency** (Prop. 3.1). Because relabelling neurons equals shifting time, the circulant average W\_cp = (1/N) Σ\_r S^{−r} W S^r also supports the sequence. W\_cp is the connectivity profile.

**Eigenvalues fixed per mode** (Cor. 3.2). For every active mode |k| ≤ D:

```latex
\lambda_k = \frac{\hat y_k}{\hat f_k} = \frac{(1 + i k \omega \tau)\,\hat x_k}{\hat f_k}, \qquad \omega = 2\pi/T
```

**Degeneracy** (Prop. 4.1). Every supporting W equals W\_D + W̃ with W̃ P\_D = 0, where W\_D has rank 2D+1 and P\_D projects onto the active modes. W̃ is unconstrained by the sequence.

**Ridge shrinkage** (§5). A Frobenius-penalized regression centred at zero gives a circulant minimizer with λ\_k^(α) = |f̂\_k|²/(|f̂\_k|² + α) · λ\_k.

**Scaffolding bound** (§7). A linear readout J from an upstream sequence u to a downstream sequence z needs, for every retained downstream mode m (with T\_u = q T\_d):

```latex
\|J\|_2 \ge q \sqrt{\frac{N_d}{N_u}}\, \frac{a_d \sigma_d}{a_u \sigma_u} \exp\!\Big[\tfrac12 (q m \omega_u)^2 (\sigma_u^2 - \sigma_d^2)\Big]
```

### 3.2 Thesis: function determines shape

A readout ŷ(c) = x(c)ᵀw reads a bump x\_i(c) = k(r\_i − c) at bump position c (Thesis eq. 2.1–2.2). Translation invariance diagonalizes the problem in the Fourier basis (§4). Ridge-regularized readout learning, with synaptic decay λ, passes mode m through the filter g\_m = a\_m/(a\_m + λ), where a\_m = |k̂\_m|² (eq. 4.6). The per-mode reconstruction error is (eq. 6.6):

```latex
E(m) = \frac{\lambda^2 S(m)}{(a_m + \lambda)^2}
```

Minimizing Σ E(m) subject to Σ a\_m = K0 gives the cube-root water-filling law (eq. 7.11–7.13):

```latex
a_m^\star = \max\!\Big( \big(2\lambda^2 S(m)/\nu^\star\big)^{1/3} - \lambda,\ 0 \Big), \qquad \sum_m a_m^\star = K_0
```

The effective rank is r\* = 2M\* + 1, with M\* the largest active mode (eq. 10.1). The joint optimum with sampling rate T uses the budget Σ a\_m + κT = B0 (eq. 8.3) and gives T\* = 2M\* (eq. 8.9).

The online kernel rule descends this error without knowing S(m) (eq. 9.10), with the readout as a faster inner loop (eq. 9.28, η\_k ≪ η\_w):

```latex
\Delta a_m = \eta_k \Big( \frac{2(1 - g_m)}{\lambda}\, \mathbb{E}\big[|\hat\varepsilon_m|^2\big] - \nu_k \Big)
```

The SOM chain (eq. 9.14–9.15, App. K.2) is derived for a linear ring driven by an external input I\_ext that places the bump: γ\_n = 1 − Ĵ\_E(n) + α k̂\_S(n) and a\_n ∝ |Î\_n|²/γ\_n².

### 3.3 Correspondence between the two

| Object | Thesis | Braun Suppl. |
| --- | --- | --- |
| Diagonalization | Circulant design matrix → DFT (§4) | Circulant part W\_cp → DFT (§3) |
| Per-mode quantity | Bump power a\_m | Eigenvalue λ\_k |
| Rank | 2M\* + 1 (eq. 10.1) | 2D + 1 (§4) |
| Mode cutoff | Normative: S(m) > ν\*λ/2 | Descriptive: fidelity cutoff D(ε) |
| Ridge filter | a/(a + λ) on readout weights (eq. 4.6) | \|f̂\|²/(\|f̂\|² + α) on recurrent eigenvalues (§5) |
| Unconstrained part | w⊥, decays under synaptic decay (§3.1, §3.5) | W̃ with W̃P\_D = 0, untouched by FORCE (Prop. 4.1) |
| Finite N | Sampling and network Nyquist (§6) | N > 2D(ε) (Remark 2.1) |
| Narrower readout | Weights grow as e^{(σ\_k² − σ\_y²)m²/2} (App. F.1) | Bound above (§7) |
| Travel | Symmetric kernel assumed (A6) | Exact, via the factor (1 + ikωτ) |
| Generator | Driven by I\_ext (eq. K.6) | Autonomous |

### 3.4 Derived facts the WPs rely on

These follow from the equations above. Each has a test in Section 14.

1. **The combined chain.** Target spectrum S(m), λ and K0 give a\* (Thesis eq. 7.13). A kernel with those powers gives a travelling rate sequence f\*. Its logit gives x\*. Cor. 3.2 then gives λ\_k\* and the circulant profile by inverse DFT. Every step is per mode except the logit.
2. **Odd part equals ωτ times the derivative of the even part.** For a symmetric bump, x̂\_k/f̂\_k is real, so Im λ\_k = kωτ · Re λ\_k. In neuron space the odd part of the profile equals ωτ times the spatial derivative of its even part. Speed and shape are independent knobs in the support condition.
3. **The linear regime carries rank, not shape.** If φ is linear on the bump's range, x̂\_k/f̂\_k is the same constant for every k. Then λ\_k = (1 + ikωτ)/gain on every active mode: a marginal projector plus a skew term. The profile then encodes which modes are active, but not their amplitudes. Amplitude information in the profile comes only from the curvature of φ. Support also says nothing about stability.
4. **Synaptic decay should collapse the degeneracy.** RLS with P(0) = cI from initial J0 minimizes the error plus (1/c)‖J − J0‖². Writing F and Y for the stacked rates and required inputs, the batch solution is W = (YFᵀ + βJ0)(FFᵀ + βI)⁻¹ with β = 1/c. FFᵀ and YFᵀ are circulant for a travelling target, so the non-circulant part of W is the initial random matrix with the active subspace projected out. Ongoing decay (Thesis eq. 3.18, A9) should remove it.
5. **Time and space harmonics are locked.** In a travelling sequence, temporal frequency kω carries exactly spatial harmonic k. So the scalar readout error's component at kω is the per-mode error ε̂\_k that Thesis eq. 9.10 needs.
6. **K0 is a firing-energy budget.** By Parseval, Σ\_m a\_m is proportional to Σ\_i x\_i(c)², with the factor (1 or N) fixed by the normalization settled in WP-T (Section 6A.5). The thesis's spectral budget is then a fixed multiple of the squared norm of the population activity vector.

### 3.5 Hypotheses

| ID | Hypothesis | WP |
| --- | --- | --- |
| HT | The thesis framework, rebuilt from the thesis text alone, reproduces its closed forms and Figs. 2–9, and every discrepancy found is documented | WP-T |
| H0 | Braun's synthetic results reproduce from the Methods, and learned profiles match Cor. 3.2 or §5 quantitatively under one of the two training conventions | WP0 |
| H1 | Adding synaptic decay to FORCE makes independently trained matrices converge to one circulant matrix without degrading the sequence | WP1 |
| H2 | The thesis optimum composed with the Braun map predicts profiles whose active-mode count equals 2M\* + 1, and implies a target spectrum for any measured bump | WP2 |
| H3 | A ring trained only on a readout target, with the thesis kernel rule, converges to the water-filling allocation, and a neuron-level implementation reproduces it in its circulant part | WP3 |
| H4 | Learned profile asymmetry obeys fact 2 quantitatively; profile alone does not determine perturbation responses | WP4 |
| H5 | MEC-like banding arises from heterogeneity and weak initialization, not from the training target; Dale-compliant members of the degenerate family exist | WP5 |
| H6 | Autonomous and input-driven models of the MEC data make distinguishable perturbation predictions | WP6 |

## 4. Notation and conventions

The two sources use different DFT normalizations, index orders and names for the same objects. The codebase fixes one convention set, implements conversions in one module (`seqnet/conventions.py`), and tests every conversion. No other module may call `np.fft` directly.

### 4.1 Fixed conventions

| Item | Convention in code | Source note |
| --- | --- | --- |
| Weight matrix | `W[i, j]` = weight from presynaptic j to postsynaptic i | Braun Methods line 782 |
| Neuron positions | θ\_n = 2πn/N, n = 0..N−1 | Thesis r\_i; Braun Suppl. §2 |
| Direction of travel | Neuron n peaks at μ\_n = nT/N, so the bump moves toward increasing n | Braun Methods eq. 4 |
| Profile index | d = (m − n) mod N = presynaptic minus postsynaptic; stored centred on d ∈ \[−⌊N/2⌋, ⌊N/2⌋\] for plots | Braun Suppl. §3; Braun Methods eq. 11 and Fig. 1C |
| Profile from matrix | cp\_d = (1/N) Σ\_n W\[n, (n + d) mod N\] | Equals the circulant projection W\_cp |
| Spatial DFT (eigenvalues) | λ\_k = Σ\_d cp\_d e^{−2πikd/N}, unnormalized | Braun Suppl. §3; then W\_cp v\_k = λ\_k v\_k with (v\_k)\_n = e^{−2πikn/N} |
| Spatial DFT (powers) | Unitary: k̂\_m = N^{−1/2} Σ\_i k(r\_i) e^{−2πimi/N}, and the thesis's operational a\_m may carry an extra factor N (Section 6A.5) | Thesis Conventions §1.13 and §4.1 |
| Temporal Fourier | f̂\_k = (1/T) ∫\_0^T f0(t) e^{−ikωt} dt, ω = 2π/T, computed by DFT over one period of bins | Braun Suppl. eq. 2 |
| Population expansion | f(θ, t) = Σ\_k f̂\_k e^{ikωt} e^{−ikθ} | Braun Suppl. §2 |
| Mode range | k ∈ {−⌊N/2⌋, …, ⌊N/2⌋}; store both signs; real signals satisfy f̂\_{−k} = conj(f̂\_k) | Aliasing requires D < N/2 (Suppl. Remark 2.1) |
| Time unit | seconds | Braun |
| Nonlinearity | φ = logistic sigmoid; φ⁻¹(f) = log((f + ε)/(1 − f + ε)), ε = 0.01 | Braun Methods eq. 6 |
| Precision | float64 throughout analysis and analytic tests; float32 allowed only in an accelerated backend validated against float64 | — |

### 4.2 Translating the thesis into Braun's setting

The thesis indexes activity by bump position c; Braun indexes by time t. For a bump moving at constant speed, c = ωt mod 2π, so the thesis target y(c) is a periodic readout target s(t) with period T.

The thesis kernel k(u) is the spatial bump profile at fixed time. Matching x\_i(c) = k(r\_i − c) to f\_i(t) = f0(t − μ\_i) gives k(u) = f0(−u/ω); for a symmetric f0 this is k(u) = f0(u/ω). The code converts with `kernel_from_temporal_profile` and `temporal_profile_from_kernel`, and tests that the round trip is exact on the bin grid.

Thesis mode m and Braun mode k are the same spatial harmonic. The thesis power a\_m uses the unitary DFT; Braun's f̂\_k is a temporal coefficient. For a travelling bump sampled at one instant, |k̂\_m|² = N · |f̂\_m|² up to the bin discretization. **\[VERIFY\]** this factor numerically in a test rather than trusting it.

### 4.3 Symbol table

| Symbol | Meaning | Code name | Thesis | Braun |
| --- | --- | --- | --- | --- |
| N | Number of ring neurons | `N` | N | N |
| T | Sequence period (s) | `T_seq` | — | T |
| T\_obs | Observations per epoch | `T_obs` | T (§2.3, §8) | — |
| τ | Neuron time constant (s) | `tau` | τ | τ = 0.3224 |
| dt | Integration step (s) | `dt` | — | 0.0081 |
| a, σ, b | Target amplitude, width (s), offset | `amp`, `width`, `offset` | — | a, σ, b |
| f | Target rates, shape (N, n\_t) | `f` | x(c) | f |
| x\* | Target activation φ⁻¹(f) | `h_target` | — | x\*, h^target |
| y\* | Required input x\* + τẋ\* | `y_target` | — | y\* |
| J or W | Recurrent weights | `J` | — | J (Methods), W (Suppl.) |
| cp | Connectivity profile | `profile` | — | cp |
| λ\_k | Eigenvalue of W\_cp at mode k | `eig` | — | λ\_k |
| a\_m | Kernel power at mode m | `alloc` | a\_m = \|k̂\_m\|² | — |
| S(m) | Target power spectral density | `psd` | S(m) | — |
| λ (thesis) | Synaptic decay = readout ridge | `decay` | λ | — |
| α (Braun) | RLS regularizer | `rls_alpha` | — | α = 0.08 |
| K0, B0, κ | Spectral budget; joint budget; sampling cost | `K0`, `B0`, `kappa` | K0, B0, κ | — |
| ν\* | Budget multiplier | `nu` | ν\* | — |
| g | FORCE initial gain | `g_init` | — | g = 1.5 synthetic, 0.05 data |

The thesis uses λ for synaptic decay and Braun uses λ\_k for eigenvalues. The code never uses a bare `lam`; it uses `decay` or `eig`.

## 5. Codebase requirements

One Python package, `seqnet`, with a NumPy reference implementation, config-driven experiment scripts, and an analytic test suite that runs in under five minutes. An accelerated backend is optional and must reproduce the reference within tolerance.

### 5.1 Stack

- Python ≥ 3.11. Core: NumPy, SciPy, matplotlib, pytest. Config: dataclasses plus YAML (Hydra is acceptable but not required).
- Optional backend: JAX for batched simulation across realizations. It sits behind the same interface as the NumPy reference and is enabled by config. Each accelerated function has a test comparing it with the reference on a small case (relative error < 1e-6 in float64).
- No deep-learning training frameworks. No autodiff in any learning rule (Section 1, rule 4).
- Storage: `.npz` for arrays, JSON for metadata. Optional zarr for large sweeps.

### 5.2 Layout

```
seqnet/
  thesis/          # standalone thesis framework, built in WP-T (Section 6A)
  bridge.py        # conversions between thesis and Braun conventions (6A.2)
  conventions.py   # DFTs, profiles, circulant ops, shift operator, unit conversions (Section 4)
  targets.py       # wrapped Gaussian, logit, Fourier-specified targets, readout targets with given PSD
  models/
    rate_rnn.py    # Braun model: Euler, noise, perturbation hooks
    driven_ring.py # wrapper over seqnet.thesis field simulation (eq. K.6)
    gain_ring.py   # Amplitude-controlled ring: projector + skew + per-harmonic divisive gain (Section 6.5)
    ei_ring.py     # WP3: fixed E ring + I populations + readout
  training/
    rls.py         # Braun FORCE variant, masks, decay, both error conventions
    readout.py     # Braun downstream rule; delta rule, RLS readout, ridge from seqnet.thesis
    kernel_rule.py # wrapper over seqnet.thesis kernel rules (eq. 9.10, Section 10.1)
  spectral/
    braun_map.py   # Cor. 3.2, §5 shrinkage, §7 bound, both conventions
    thesis_opt.py  # wrapper over seqnet.thesis optimum and errors
    kernels.py     # wrapper over seqnet.thesis kernels
  analysis/
    profile_features.py  # smoothing, peaks, range, asymmetry (Braun Methods eq. 11-13)
    motifs.py            # Braun Methods eq. 14-15 (reconstructed)
    perturb.py           # structured and unstructured perturbation protocols
    stability.py         # Floquet multipliers of the periodic orbit
    metrics.py           # correlations, residual norms, per-mode diagnostics
  data/
    mec.py, bat.py       # loaders and preprocessing (Braun Methods lines 842-905)
  run.py                 # CLI entry: python -m seqnet.run <wp> --config <file>
configs/  wp0/ ... wp6/
tests/    test_conventions.py, test_braun_map.py, test_thesis_opt.py, ...
results/  wp0/ ... wp6/   # REPORT.md, figures/, data/
```

### 5.3 Reproducibility

- Every run is defined by one config file. The run directory stores the resolved config, the git commit hash, package versions, the random seed and wall time in `meta.json`.
- One `numpy.random.Generator` per realization, seeded from (base seed, realization index), so realizations can be rerun individually.
- Every \[VERIFY\] switch appears in the config with its default and is copied into `meta.json`.
- Sweeps are lists of configs generated by a sweep file; never hand-edited loops in scripts.

### 5.4 What to store

- Always: final weights' circulant profile, eigenvalues for |k| ≤ 20, mean matrix across realizations, all scalar metrics, training curves (correlation per iteration, mean absolute weight change per iteration as in Braun Methods eq. 10).
- Full matrices: for the first 5 realizations of every condition, and for all realizations in WP0.1 and WP1 (a 501 × 501 float64 matrix is 2 MB).
- Activity: one full test-iteration trajectory per condition at the target bin resolution, not every integration step.

### 5.5 Compute budget

The RLS update costs O(N²) per update. Braun's synthetic setting has 2086 integration steps per iteration with an update every 16 steps, so about 130 updates per iteration, 300 training and 10 test iterations, and 50 realizations per condition. Batch realizations along a leading axis.

The mouse setting is heavier: 39,655 integration steps per iteration with updates every 160 steps. Default development settings use 10 realizations and 100 iterations; final runs use Braun's numbers. Each REPORT.md states which was used.

### 5.6 Code standards

- Type hints and docstrings with source citations on every public function.
- Pure functions for all formulas; simulation state in small dataclasses.
- A figure function never computes; it loads stored results. Figures regenerate from `results/` alone.
- Logging through the standard `logging` module; no prints in library code.

## 6. Core module specifications

These modules are shared by all WPs. Modules 6.4, 6.8 (delta rule and ridge), 6.9, 6.11 and 6.12 are thin wrappers over seqnet.thesis, which WP-T builds (Section 6A). Each subsection states inputs, outputs, the exact algorithm with its source, and the switches it must expose.

### 6.1 `conventions.py`

- `profile_from_matrix(W) -> cp` with cp\_d = (1/N) Σ\_n W\[n, (n+d) mod N\] (Section 4.1).
- `circulant_from_profile(cp) -> W` and `eigs_from_profile(cp) -> eig` (unnormalized DFT, Section 4.1); `profile_from_eigs(eig)` its inverse.
- `centred_profile(cp)` returns d ∈ \[−⌊N/2⌋, ⌊N/2⌋\] for plotting and feature extraction.
- `unitary_dft(x)`, `alloc_from_rates(r_snapshot) -> a` with a\_m = |k̂\_m|².
- `shift(x, r)` = (S^r x)*n = x*{(n+r) mod N} (Braun Suppl. §3).
- `kernel_from_temporal_profile`, `temporal_profile_from_kernel` (Section 4.2).
- `even_odd_split(cp)` returns the even and odd parts about d = 0.

### 6.2 `targets.py`

- `wrapped_gaussian(t, width, T, n_images=6)` = Σ\_{|m|≤n\_images} exp(−(t − mT)²/(2σ²)) (Braun eq. 5). Test the image truncation against n\_images = 20.
- `sequence_rates(N, T, amp, width, offset, t_grid) -> f[N, n_t]` with f\_n(t) = a·g(t − nT/N) + b (Braun eq. 4).
- `logit_target(f, eps=0.01)` (Braun eq. 6) and its analytic time derivative ẋ = ḟ · \[1/(f + ε) + 1/(1 − f + ε)\], used for y\* = x\* + τẋ\*.
- `fourier_coeffs_gaussian(k, amp, width, T)` = a σ √(2π)/T · exp(−σ²(kω)²/2) for k ≠ 0, and a σ√(2π)/T + b for k = 0 (Braun Suppl. eq. 2–3). Used as an analytic check.
- `sequence_from_kernel(kernel_coeffs, T, N, offset_policy)` builds a travelling sequence from an arbitrary spatial kernel (WP2, WP3).
- `readout_target_from_psd(S, T, phases, rng)` builds a periodic scalar target s(t) with power S(m) at harmonic m. Phases are random by default and fixed by seed; the thesis is phase-blind (A4), so results must not depend on them. Test this.

### 6.3 `models/rate_rnn.py` (Braun model)

State h ∈ R^N, rates r = φ(h). One Euler step (Braun eq. 1–3, verified against the rasterized PDF page 21):

```latex
h(t) = h(t - dt) + \frac{dt}{\tau}\big(-h(t - dt) + z(t - dt)\big), \qquad z = J r + \zeta, \qquad \zeta_i = \text{gain}\,\sqrt{\tau/dt}\;\nu_i,\ \nu_i \sim \mathcal N(0,1)
```

Defaults: τ = 0.3224 s, dt = 0.0081 s, gain = 0.5. Initial state h(0) = h\_target(0) (Methods lines 839, 871, 894).

Perturbation hooks, each applied to one ensemble in one time window or to a neuron set:

- Silencing: r\_i(t) = 0 for i ∈ E, t ∈ window; h keeps evolving (Methods eq. 21).
- Current injection: z\_i ← z\_i + I for i ∈ E, t ∈ window (Methods lines 1170–1178), I ∈ {0.5, 2, 3, 5}.
- Noise injection: z\_i ← z\_i + gain\_p · √(τ/dt) · ν\_i for i ∈ Q, t in the first half of the run (Methods eq. 22), with gain\_p ∈ {5, 10, 15, 20} and |Q|/N ∈ {10, 25, 50, 75}%.

Batched over realizations along axis 0.

### 6.4 `models/driven_ring.py` (thesis field equation)

Implements Thesis eq. K.6 on a ring with FFT convolution:

```latex
\tau \partial_t r = -r + J_E * r - \alpha\, k_S * r + I_{ext}(\theta - \omega t)
```

Options: a linear or sigmoid output nonlinearity; a symmetric k\_S plus an optional asymmetric k\_S. The analytic steady state for a travelling input is r̂\_n = Î\_n/(γ\_n + inωτ) with γ\_n = 1 − Ĵ\_E(n) + αk̂\_S(n). Test the simulation against it.

### 6.5 `models/gain_ring.py` (amplitude-controlled ring)

A ring held on a chosen set of harmonics K, with one divisive gain per harmonic and a rigid-rotation term. For each held harmonic k with complex amplitude z\_k (the uniform mode k = 0 included):

```latex
\tau \dot z_k = \big(g_k(|z_k|) - 1\big)\, z_k + i k \Omega \tau\, z_k, \qquad g_k = \frac{A_k^2 + \sigma^2}{\sigma^2 + |z_k|^2}
```

In neuron space, the recurrent input is Σ\_k g\_k (2/N) Σ\_j cos k(θ\_i − θ\_j) r\_j over the held modes, plus the skew kernel Ωτ (2/N) Σ\_k k sin k(θ\_i − θ\_j). Each g\_k is computed from the population's projection onto harmonic k. The fixed point is |z\_k| = A\_k, and every harmonic turns at kΩ, which is rigid travel at angular speed Ω = ω. Fix the sign of the skew kernel by a test so the bump moves toward increasing n (Section 4.1).

Defaults: σ² = 0.25 (configurable); A\_0 large enough that all rates stay positive. Expose the set points A\_k as the parameters the WP3 slow loop updates.

### 6.6 `models/ei_ring.py` (WP3)

A fixed E ring (from 6.3, 6.4 or 6.5) plus an inhibitory population of N\_I neurons on a ring with fixed translation-invariant E→I weights U, trainable I→E weights V, optional I→I weights, and a linear readout z\_out = wᵀr\_E. Specified in WP3.

### 6.7 `training/rls.py` (Braun FORCE variant)

At update times z ∈ Z\_m (every m integration steps; m = 16 synthetic and bat, m = 160 mouse), with R = r(z) (Braun eq. 7–9, verified against PDF page 26):

```latex
P \leftarrow P - \frac{P R R^\top P}{1 + R^\top P R}, \qquad c = \frac{1}{1 + R^\top P R}, \qquad J \leftarrow J - c\, e\, (P R)^\top, \qquad P(0) = p\,I,\ p = 0.08
```

The error e(z) = z(z) − h\_target(z) includes the injected noise (Braun eq. 7). Required switches:

| Switch | Default | Alternative | Why |
| --- | --- | --- | --- |
| `error_target` | `methods`: z is trained toward x\* | `supplement`: z is trained toward y\* = x\* + τẋ\* | Braun eq. 7 and Suppl. Def. 1.1 disagree (Section 15) |
| `c_uses_updated_P` | `true`: c and PR use the updated P, as printed | `false`: standard Sussillo–Abbott update, J ← J − e (P\_new R)ᵀ with no extra c | The printed form scales the step twice \[VERIFY\] |
| `state_reset_each_iteration` | `true`: reset h to h\_target(snippet start) | `false`: continue the state | Not stated in Methods \[VERIFY\] |
| `target_time_alignment` | evaluate the analytic target exactly at update times t = n·m·dt | nearest 0.129 s bin | 16 × 0.0081 = 0.1296 ≠ 0.129 \[VERIFY\] |
| `weight_decay` | 0 | rate λ\_J > 0, applied as J ← (1 − λ\_J) J after each update | WP1 |
| `ridge_centre` | `init` (standard RLS) | `zero`: train ΔJ with J0 fixed but outside the penalty, i.e. ridge on J itself | WP1 |
| `mask` | none | one of three sparsity masks below | Supp. Fig. 2 |

Initialization: J\_ij = (g/√N) ξ\_ij, ξ ∼ N(0,1), g = 1.5 synthetic, 0.05 data (Methods lines 910–913). Each realization draws its own J and its own snippet start, uniform in \[0, 119.2\] s for synthetic targets (Methods lines 918–921).

Sparsity masks (Methods lines 1031–1074): (a) zero all outgoing weights of a random presynaptic set and exclude them from training, with a reduced P over the remaining set; (b) the same for a contiguous block in the sequence ordering; (c) random synapses zeroed, with a separate P\_i per postsynaptic row (Methods eq. 18–19).

Diagnostics per iteration: Pearson correlation between target and response, and mean absolute weight change (Methods eq. 10).

### 6.8 `training/readout.py`

- `delta_rule(w, x, y, eta, decay)`: w ← w + η(e x − decay·w), e = y − xᵀw (Thesis eq. 3.17–3.18).
- `rls_readout(...)`: RLS on a readout, same update as 6.7 with a scalar error.
- `braun_downstream(A, inputs, targets, alpha=0.015, n_iter=1000)`: A initialized to all ones; per iteration A ← A + α · mean\_t(error\_i(t) input\_j(t)) with error = target − output (Braun lines 1266–1275, verified against PDF page 36).
- `ridge_readout_closed_form(X, y, decay)` = (XᵀX + decay·I)⁻¹Xᵀy (Thesis eq. 3.13), the reference for convergence tests.

### 6.9 `training/kernel_rule.py` (Thesis eq. 9.10)

- State: allocation a\_m ≥ 0 for active candidate modes, multiplier ν\_k.
- Per slow step: estimate E|ε̂\_m|² from the readout error over whole sequence cycles. Use the temporal DFT of e(t) over each cycle at frequency mω (Section 3.4, fact 5), averaged over a configurable number of cycles. Then Δa\_m = η\_k(2(1 − g\_m)/λ · E|ε̂\_m|² − ν\_k) and clip at zero.
- Budget enforcement, two options: (i) ν\_k fixed by config; (ii) ν\_k updated as ν\_k ← ν\_k + η\_ν(Σ a\_m − K0). Default (ii).
- Timescale separation (Thesis eq. 9.28): `inner_steps_per_kernel_step` is a config value, and the run logs the readout's distance from its ridge optimum at every slow step.

### 6.10 `spectral/braun_map.py`

- `required_eigs(f, T, tau, convention)` returns λ\_k for |k| ≤ K\_max. Convention `supplement`: λ\_k = (1 + ikωτ)x̂\_k/f̂\_k (Cor. 3.2). Convention `methods`: compute the effective rates F′ = φ(L x\*), where L is the per-mode filter 1/(1 + ikωτ), then λ\_k = x̂\*\_k/F̂′\_k.
- `ridge_shrunk_eigs(eig, f_hat, alpha)` (Suppl. §5).
- `active_cutoff(f, y, eps)` returns D\_f(ε), D\_y\*(ε) and D\_seq (Suppl. §2).
- `scaffolding_bound(...)` (Suppl. eq. 6–7).
- Batch least-squares reference: `batch_ridge_W(F, Y, J0, beta)` = (YFᵀ + βJ0)(FFᵀ + βI)⁻¹ (Section 3.4, fact 4).

### 6.11 `spectral/thesis_opt.py`

- `water_fill(S, decay, K0, T_obs)` returns a\*, ν\*, the active set, M\* and r\* (eq. 7.11–7.13). Solve ν\* by bisection on the budget equation (7.12); use log-space bisection and tolerance 1e-12 on the budget residual.
- `joint_opt(S, decay, B0, kappa, N)` (eq. 8.3–8.9).
- `error_terms(a, S, decay, T_obs, N)` returns bias, aliasing and network-floor terms (eq. 6.7–6.9).
- `marginal_returns(a, S, decay)` = 2λ²S/(a + λ)³ (eq. 7.5); used as the convergence diagnostic.
- `implied_psd(a, decay, nu)` = ν(a + λ)³/(2λ²) on active modes. This inverts eq. 7.10: the target spectrum for which a given bump is Wiener-optimal, defined up to the scale ν (WP2).
- `appendix_I_cases()` returns the six closed-form examples from Thesis App. I.7 (constant, single mode, flat to B, Gaussian PSD, polynomial PSD, narrowband) for tests.

### 6.12 `spectral/kernels.py`

Turns an allocation into rates. The thesis fixes magnitudes only, so two policies are needed:

- `sign_policy`: `positive` (all k̂\_m ≥ 0, maximal central peak, default), `alternating`, or `explicit` (a given sign vector). Readout error is sign-invariant; the Braun map is not, because of the logit. Report both.
- `offset_policy`: rates must lie in (0, 1) for the sigmoid. Default: add offset b = r\_min − min k with r\_min = 0.02; if max rate > 1 − r\_min, rescale the kernel and log the realized K0. Whether the offset counts against K0 is a decision for Max (Section 15); default: the budget excludes the DC mode.

### 6.13 `analysis/profile_features.py` (Braun Methods lines 979–1012)

1. Replace cp at d = 0 by linear interpolation of d = ±1.
2. Smooth with a Gaussian-weighted moving average of window 50 samples. This mimics MATLAB `smoothdata(..., 'gaussian', 50)`. **\[VERIFY\]** MATLAB's kernel width for this call; implement as a parameter `sigma_frac` (std = window × sigma\_frac) and record it.
3. Find local maxima with minimum prominence 0.001 (`scipy.signal.find_peaks`).
4. Peak width: mean width at half prominence across maxima (`peak_widths` with rel\_height = 0.5).
5. Range = max − min (eq. 12).
6. Asymmetry = min over local maxima of |d| at the maximum (eq. 13).
7. Also report the raw self-weight mean (the d = 0 value before step 1), which Braun discards.

### 6.14 `analysis/motifs.py` (Braun eq. 14–15, verified against PDF page 28)

X = W − mean(W); σ² = tr(XXᵀ)/N; τ\_rec = tr(XX)/(σ²N); τ\_div = (Σ\_ij (XXᵀ)\_ij − tr(XXᵀ))/(σ²N(N−1)); τ\_con the same with XᵀX; τ\_chn = 2(Σ\_ij (XX)\_ij − tr(XX))/(σ²N(N−1)).

Note for interpretation: for a centred circulant matrix every row and column sums to zero, so τ\_div, τ\_con and τ\_chn vanish identically. Nonzero values measure departure from translation invariance. Add a test.

### 6.15 `analysis/perturb.py` (Braun Methods lines 1130–1222)

- Run the trained network frozen for two periods (17 s synthetic). Split the first period into 5 windows (1.7 s, 209 steps). Split neurons into ⌊1/f\_E⌋ contiguous ensembles of ⌊N·f\_E⌋ neurons, f\_E ∈ {0.1, 0.2, 0.3}.
- For each (ensemble, window): perturb, then compute the Pearson correlation between the perturbed and unperturbed activity from the end of the window to the end of the run.
- Store the ensemble × window correlation matrix CC. Align each row so the window in which that ensemble is active sits in the middle column (periodic in windows), then average rows (Methods lines 1216–1222).
- Unstructured: noise injection over the first half (8.5 s); correlation over the second half.

### 6.16 `analysis/stability.py`

Support is not stability (Section 3.4, fact 3). For a frozen trained network without noise, find the periodic orbit, then compute the Floquet multipliers of the one-period map. Use a finite-difference Jacobian with Arnoldi iteration for the leading 30 multipliers, or the full matrix for N ≤ 501.

Report: the neutral multiplier (phase, should be 1), the leading transverse multipliers, and their projection onto Fourier modes. Also report whether the noise-free network keeps the sequence at all.

### 6.17 `analysis/metrics.py`

- Pearson correlation of flattened arrays: target vs response (neurons × time), matrix vs matrix, profile vs profile.
- Residual fraction ‖W − W\_cp‖\_F/‖W‖\_F.
- Learned eigenvalues λ̃\_k from the mean profile, compared with predictions per mode: relative error and phase error for |k| ≤ 10.
- Subspace residual ‖W̃P\_D‖\_F (should be about zero for a supporting matrix; Prop. 4.1).

### 6.18 `data/mec.py`, `data/bat.py` (Braun Methods lines 841–905, 1084–1107)

**Mouse (MEC), session 7 of mouse #60584:**

1. Deconvolved activity at 0.129 s bins.
2. Moving mean over 120 bins.
3. Downsample to 1.29 s bins.
4. Divide each neuron by its maximum.
5. Order neurons by the PCA method of Gonzalo Cogno et al. (2024), using PCs 1–2.
6. Training targets: 50 random snippets of 250 bins (322.5 s) each, updates every 160 steps.

**Mouse bump fit** (Supp. Fig. 6), a separate path:

1. Drop the first 4 of 24 sequences.
2. Rescale the remaining 20 to the longest one, 177.8 s (1378 bins).
3. Average across sequences and normalize each neuron by its maximum.
4. Threshold at mean + 1 SD, setting values below to 0.
5. Circularly shift each neuron to peak at bin 689.
6. Rescale to 8.5 s, average across neurons, and fit a Gaussian plus offset. Expected: a = 0.29, σ = 1.68 s, b = 0.15, R² = 0.99.

**Bat (CA1):**

1. 168 time cells, z-scored rates, first 6 s after landing.
2. Order by peak time after a 15-bin moving mean at 0.1 s bins.
3. Downsample the unsmoothed rates to 0.129 s.
4. Set negative values to 0.
5. Apply a 15-bin moving mean.
6. Divide each neuron by its maximum.
7. Expected fit: a = 0.83, σ = 1.72 s, b = 0.05.

## 6A. WP-T (preliminary): rebuild the thesis framework as `seqnet.thesis`

WP-T rebuilds the thesis's computational framework from the thesis PDF alone, as a subpackage of `seqnet` that also runs with no other part of the package installed. It comes first because WP2, WP3, WP4.3 and WP5.3 call it. No earlier implementation is provided; the thesis text and equations are the specification.

### 6A.1 Scope

| Thesis part | Implement | Equations |
| --- | --- | --- |
| §2 Ring and manifold | Ring, kernel-shaped activity x(c), design matrix X, its SVD, observation subspace | eq. 2.1–2.6 |
| App. D Kernel families | Point, box (B⁰), B-spline of order p, Gaussian, Gabor; normalization Σ\_i k(r\_i)Δr = 1 | D.1–D.5 |
| §3 Readout | Forward map, pseudoinverse, ridge closed form, spectral filter, delta rule with decay, per-mode convergence rate | eq. 3.1–3.19 |
| §4 Fourier diagonalization | DFT of the kernel, circulant structure, SVD–DFT identity, effective rank | eq. 4.1–4.8 |
| §5 and App. C Coherence | Pairwise coherence and its spectral form, independence distance d\*(ε), spectral spread, uncertainty bound σ\_x σ\_ξ ≥ 1/2 | eq. 5.1–5.11, App. C |
| §6 and App. G Errors | Bias, aliasing and network-floor terms; observation-noise extension | eq. 6.1–6.9, 6.6′ |
| §7 Wiener-optimal kernel | Cube-root water-filling, budget equation, marginal returns, optimality diagnostic | eq. 7.5, 7.9–7.16 |
| §8 and App. J Joint optimum | Joint budget with sampling cost κ, T\* = 2M\* | eq. 8.1–8.9 |
| §9 and App. K Online learning | Kernel rule, S(m) cancellation, per-mode error estimator, sampling rule, timescale separation, the complete system | eq. 9.1–9.31, K.1 |
| §9.3 and App. K.2 SOM chain | Linear field equation, chain k\_S → γ → a → g → E and its inverse | eq. 9.13–9.16, K.6–K.13 |
| App. E Heterogeneity | Position-dependent kernels and the coupling-error correction | E.1–E.4 |
| App. I Worked examples | The six closed-form target classes | I.1–I.7 |
| App. L Criticality | Mode-resolved branching ratios (low priority) | L.1 |

Excluded: App. M (only log the aggregate energy A\_m, eq. M.1–M.2, for later use) and the narrative parts of App. F (implement only the weight-growth formula F.1 as a diagnostic).

Added beyond the thesis, because WP3 needs it: a second slow-loop rule that descends the full ridge objective (Section 10.1). It sits beside the thesis rule behind the same interface and is never the default.

### 6A.2 Standalone and integration rules

1. The subpackage lives in `seqnet/thesis/` and imports only the standard library, NumPy, SciPy, and matplotlib (plots only). It never imports another `seqnet` module. A test parses every file in `seqnet/thesis/` with `ast` and fails on any `seqnet.` import outside `seqnet.thesis`.
2. It runs alone. A CI job copies `seqnet/thesis/` and `tests/thesis/` into an empty temporary directory and runs the tests there.
3. It has its own entry point `python -m seqnet.thesis <experiment> --config <file>`, its own `configs/thesis/` and `results/thesis/`, and a `README.md` documenting the public API with a minimal example.
4. The public API is the `__all__` list in `seqnet/thesis/__init__.py`. Everything else is private. The rest of `seqnet` imports only public names.
5. Integration runs one way. Other modules import `seqnet.thesis`; conversions between thesis and Braun conventions live in `seqnet/bridge.py`, outside the subpackage. The subpackage uses thesis notation and equation numbers internally and never Braun's.
6. Main-package modules that would duplicate thesis code become thin wrappers over the public API: `spectral/thesis_opt.py`, `spectral/kernels.py`, `training/kernel_rule.py`, the delta rule and ridge closed form in `training/readout.py`, and `models/driven_ring.py`. Sections 6.4, 6.8, 6.9, 6.11 and 6.12 describe those interfaces; WP-T implements them.

### 6A.3 Layout

```
seqnet/thesis/
  __init__.py      # public API (__all__)
  __main__.py      # CLI: python -m seqnet.thesis <experiment> --config <file>
  fourier.py       # eq. 4.1 DFT, circulant eigenvalues, normalization resolver (6A.5)
  ring.py          # Ring (N, positions, dr); Schedule (spacing d, T_obs = N/d, positions c_t)
  kernels.py       # App. D families; kernel from allocation with sign and offset policies
  design.py        # x(c), design matrix X, SVD, observation subspace, effective rank (4.8)
  targets.py       # target classes by PSD S(m); samplers; App. I and Fig. 2 classes
  readout.py       # pinv (3.11), ridge (3.13), per-mode ridge (4.5), filter (4.6), delta rule (3.18), RLS readout
  coherence.py     # §5, App. C
  errors.py        # §6, App. G
  optimum.py       # §7, §8, App. I, App. J; implied PSD (inverse of 7.10)
  learning.py      # §9: kernel rules, per-mode error estimators, sampling rule, LearningSystem
  field.py         # App. K.2 field equation (K.6), SOM chain (9.14-9.16) and its inverse, App. L
  perturbation.py  # App. E position-dependent kernels and coupling error
  figures/         # one script per reproduced thesis figure (Figs. 2-9)
  README.md
tests/thesis/      # all WP-T tests; runnable in isolation
configs/thesis/    # one config per figure and per test experiment
results/thesis/    # REPORT.md, figures/, data/
```

### 6A.4 Public API

| Object | Contents | Source |
| --- | --- | --- |
| `Ring(N)` | Positions r\_i = 2πi/N, spacing Δr | §2.1 |
| `Schedule(N, d)` | T\_obs = N/d observation positions c\_t = t·d·2π/N per epoch | §2.3 |
| `Kernel` | Values k(r\_i), allocation a\_m, signs; constructors `from_family(name, **params)` and `from_alloc(a, sign_policy, offset_policy)`; methods `spectrum()`, `effective_rank(eps)`, `autocorrelation()` | §2, §4, App. D |
| `TargetClass` | PSD S(m); `sample(rng)` returns a target y(c) on the ring with random phases; `fixed(y)` wraps one deterministic target | A4, App. I |
| `design_matrix(kernel, schedule)` | X with X\_{t,i} = k(r\_i − c\_t) | eq. 2.4 |
| `ridge_readout`, `per_mode_ridge`, `spectral_filter` | Closed forms | eq. 3.13, 4.5, 4.6 |
| `DeltaRule`, `RLSReadout` | Online readout learners with a common `step(x, y) -> w` interface | eq. 3.18; Section 6.7 |
| `error_terms(a, S, decay, T_obs, N, noise_var=0)` | Bias, aliasing, network floor | eq. 6.6–6.9, 6.6′ |
| `water_fill(S, decay, K0, T_obs)` | Result with a\*, ν\*, active set, M\*, rank 2M\* + 1 | eq. 7.11–7.13 |
| `joint_optimum(S, decay, B0, kappa, N)` | a\*, T\*, ν\* | eq. 8.3–8.9 |
| `marginal_returns`, `optimality_gap`, `implied_psd` | Diagnostic and inverse map | eq. 7.5, 7.10 |
| `ThesisKernelRule`, `RidgeObjectiveRule` | Slow-loop rules with a common `step(error_stats, a) -> a` interface | eq. 9.10; Section 10.1 |
| `LearningSystem` | Ring, initial kernel, target, readout learner, kernel rule, schedule, η\_w, η\_k, inner epochs per kernel step; `run(n_steps)` returns a History | eq. 9.31 |
| `som_chain`, `inverse_som_chain`, `simulate_field` | Forward chain, its inverse a\* → k̂\_S\*, and the field equation | eq. 9.14–9.16, K.6–K.10 |
| `independence_distance`, `spectral_spread`, `uncertainty_product` | §5 quantities | eq. 5.6–5.11 |
| `coupling_error` | Excess error for position-dependent kernels | App. E |

Every formula function is pure and vectorized over modes. Objects are frozen dataclasses. A History stores, per kernel step: allocation, ν\_k, per-mode error estimates, readout distance from its ridge optimum, total error and its three terms.

### 6A.5 Numerical decisions to settle first

- **DFT normalization \[VERIFY, D10\].** Eq. 4.1 defines a unitary DFT, but eq. 4.2 and 4.7 write X = F\* diag(k̂) F with F unitary, which makes the singular values equal |k̂\_m|. For a circulant X with first row k(r\_i), the eigenvalues are the unnormalized DFT, √N times the eq. 4.1 coefficients. Resolve in `fourier.py`: define the operational quantity a\_m as the squared singular value of X at T\_obs = N, since that is what enters g\_m, E(m) and the optimum. Record a\_m = N|k̂\_m|² under eq. 4.1 and use it everywhere. Then K0 = Σ a\_m = N Σ\_i k(r\_i)². This rescales Section 3.4, fact 6 by N; correct that fact once the test confirms.
- **Sampling below N.** For T\_obs < N, X is T\_obs × N and modes alias. Build X exactly and check the per-mode formulas (eq. 6.7–6.8) numerically, including how a\_m scales with T\_obs/N (the thesis rate table uses |k̂|²/T).
- **Expectation over the target class.** Two modes: `ensemble` draws a new target with random phases every epoch, so expectations run over draws; `fixed` uses one target with S(m) = |ŷ\_m|². Both must reach the same fixed point; report both for Fig. 8.
- **Per-mode error estimator.** E|ε̂\_m|² comes from the DFT over observation positions of one epoch's prediction errors, averaged by an exponential moving average with a configurable window. Also provide the travelling-time version (temporal DFT at frequency mω) for WP3, exposed through the bridge.
- **Symmetry (A6).** Kernels are real and even. Allocations are stored for m ≥ 0 with a\_{−m} = a\_m. Rank counts both signs, giving 2M\* + 1.
- **Budget solve.** Bisection for ν in log space with automatic bracket expansion and tolerance 1e-12 on the budget residual. Handle the case where no mode is active.
- **Nonconvex spectra.** For multi-peak S(m) (thesis §8.2, §9.6), run the kernel rule from 5 random initial allocations and report the spread of fixed points.
- **Timescale separation.** Log the readout's relative distance from its ridge optimum at every kernel step. Warn if it exceeds 1e-2.
- **Parameter values \[VERIFY, D11\].** The thesis text does not list the parameters behind Figs. 2–9. Use defaults N = 128, λ ∈ {1e-3, 1e-2, 1e-1}, and K0 equal to the energy of a Gaussian kernel of width 0.3 rad. Record them, and replace them if Max supplies the originals.

### 6A.6 Reproduction targets

Locate each figure in the thesis PDF by its caption and rasterize the page to compare.

| Thesis figure | Reproduce | Acceptance |
| --- | --- | --- |
| Fig. 2 | Kernel–target matrix: 5 target classes (Gaussian PSD, polynomial, chaotic harmonic, piecewise, narrowband) × 5 kernels (broad Gaussian, narrow Gaussian, point, box, Gabor); error heatmap, best kernel per class, effective ranks | No single kernel best for all classes; best kernel per class matches the figure's stars |
| Fig. 3 | Five kernel families: profile, spectrum, per-mode convergence, reconstruction | Per-mode convergence rates match eq. 3.19 within 5% |
| Fig. 4 | Uncertainty and aliasing against sampling spacing d | Aliasing error matches eq. 6.8 within 5%; σ\_x σ\_ξ ≥ 1/2, with equality for the Gaussian |
| Fig. 5 | Per-mode error split into bias and variance | Matches eq. 6.6 and 6.6′ within 5% |
| Fig. 6 | Kernel families against the Wiener optimum | Optimum error ≤ every family at equal budget |
| Fig. 7 | Joint optimum against budget for a Gaussian target spectrum | T\* = 2M\* at each budget; error decreasing in B0 |
| Fig. 8 | Online learning from a broad Gaussian | Allocation within 5% of eq. 7.13 on active modes; active set exact; readout at ridge equilibrium within each kernel step |
| Fig. 9 | Sweeps over N, K0 and λ | Error falls with N and K0 and rises with λ; T\* values reported |

### 6A.7 Tests (`tests/thesis/`)

| ID | Test | Source |
| --- | --- | --- |
| TT1 | Normalization resolver: singular values of X at T\_obs = N equal √N\|k̂\_m\| under eq. 4.1 (or report otherwise) | eq. 4.1, 4.2, 4.7 |
| TT2 | Right singular vectors of X are Fourier modes | eq. 4.7 |
| TT3 | Per-mode ridge (4.5) equals matrix ridge (3.13) | §3.4, §4.3 |
| TT4 | Delta rule converges to the ridge solution at the per-mode rates of eq. 3.19 (this is T10) | §3.5 |
| TT5 | Effective rank (4.8) for each App. D family matches its analytic spectrum | §4.5, App. D |
| TT6 | Spectral form of the coherence (5.6) equals the pairwise overlap (5.2) | §5 |
| TT7 | Uncertainty bound holds for all families, with equality for the Gaussian | App. C |
| TT8 | Bias + aliasing + floor equals the total error measured by simulation | eq. 6.7–6.9 |
| TT9 | `water_fill` reproduces App. I (this is T11), except the width statement under D1 | App. I.7 |
| TT10 | KKT: marginal returns equal ν\* on active modes and are below ν\* on inactive ones | eq. 7.10–7.11 |
| TT11 | The observable gradient (9.6) equals the analytic gradient (9.5) at ridge equilibrium | eq. 9.5–9.6, K.1 |
| TT12 | `ThesisKernelRule` converges to eq. 7.13 (this is T12) | eq. 9.29 |
| TT13 | `RidgeObjectiveRule` converges to the square-root law (this is T13) | Section 10.1 |
| TT14 | SOM chain forward then inverse returns the input; field simulation matches the K.9–K.10 steady state | App. K.2 |
| TT15 | Joint optimum satisfies T\* = 2M\* and the joint budget | eq. 8.9 |
| TT16 | Coupling error scales as ε² | App. E |
| TT17 | Import boundary (6A.2, rule 1) | — |
| TT18 | Isolated run (6A.2, rule 2) | — |

### 6A.8 Deliverables and acceptance

- The subpackage with its public API, README and CLI.
- `results/thesis/REPORT.md` with the Fig. 2–9 reproductions, all TT tests passing, and a list of every discrepancy found in the thesis, with evidence. Expected entries: the normalization (D10), the √3 direction (D1), T = 2M (D2), κ versus λ/2 (D6). This list feeds the thesis preprint.
- Integration check: WP2.1 runs end to end through the public API and `seqnet/bridge.py` only.

WP-T is accepted when TT1–TT18 pass, Figs. 6, 8 and 9 are reproduced quantitatively, and the other figures qualitatively. The Braun model, FORCE and anything from WP0 onward are out of scope for WP-T.

## 7. WP0: Reproduce Braun et al. and check the supplement against learned networks

WP0 rebuilds every synthetic result of Braun et al. from the Methods and adds one new check: whether learned profiles match Suppl. Cor. 3.2 and §5 quantitatively, and under which training convention. Everything later builds on WP0.1 and WP0.8.

Shared settings unless a sub-WP says otherwise: N = 501, T = 8.5 s, τ = 0.3224 s, dt = 0.0081 s, bin 0.129 s, noise gain 0.5, g = 1.5, p = 0.08, m = 16, snippet 17 s, 300 training plus 10 test iterations, 50 realizations.

### WP0.1 Degeneracy and the common profile (Fig. 1, Supp. Fig. 1)

- Target a = 0.5, σ = 1.7 s, b = 0.
- Outputs: training curves (Supp. 1E), the 1225 pairwise matrix correlations and profile correlations (Fig. 1E), the mean matrix, its profile, and the dynamics it generates (Fig. 1F). Also a scatter of realization-1 versus realization-50 weights before and after training (Supp. 1G).
- Targets: matrix correlation about 0.07; profile correlation about 0.97–0.98; test-iteration target–response correlation about 0.8; mean absolute weight change still decreasing at iteration 300.
- Also record: the mean self-weight (diagonal). Supp. 1G after training shows a separate cluster near (−1.5, −1.5), which is probably the diagonal. Confirm or refute this.

### WP0.2 Profile versus target parameters (Fig. 2, Supp. Figs. 1A–D, 3, 4)

| Sweep | Varied | Fixed |
| --- | --- | --- |
| Offset | b ∈ {0, 0.05, …, 0.45} | a = 0.5, σ = 1 s |
| Width | σ ∈ {0.5, 0.9, 1.0, 1.15, …, 1.9} s | a = 0.5, b = 0 |
| Amplitude | a ∈ {0.2, …, 1.0} step 0.1 | σ = 1 s, b = 0 |
| Amplitude × width | a ∈ {0.4, 0.6, 0.8, 1}, σ ∈ {0.5, 0.9, 1.3, 1.7} | b = 0 |
| Offset × amplitude | b ∈ {0, 0.1, 0.2, 0.3}, a ∈ {0.4, 0.5, 0.6, 0.7} | σ = 1 s |

- Outputs: profiles, and the features from 6.13 (asymmetry, range, mean peak width), plus the training performance box plots of Supp. 1F.
- Expected trends: asymmetry stays near 10–20 neurons up to b = 0.25 and then rises to about 75 at b = 0.45. Range falls with width. Peak width falls with amplitude, and strong oscillations appear at a = 1. Performance rises with amplitude and falls with offset.

### WP0.3 Motifs (Supp. Fig. 5)

Use the mean matrices from WP0.2 and the formulas in 6.14. Expected: τ\_rec between about 0.55 and 0.8, rising with offset and width, and with amplitude up to 0.7 before collapsing near a = 1. τ\_div, τ\_con and τ\_chn should be about 0. Also report all four for the circulant projection alone, and for the residual W − W\_cp alone, to show which part carries τ\_rec.

### WP0.4 Sparsity (Supp. Fig. 2)

Use the three mask types from 6.7 at fractions 0 to 0.95 in steps of 0.05, with 50 realizations each. Expected: (a) and (c) keep performance up to about 0.9; (b) fails beyond about 0.35, and its profile correlation with the dense network drops. Report the profile correlation with the dense network's profile.

### WP0.5 Perturbations (Fig. 4, Supp. Figs. 8–9)

- Target a = 0.6, σ = 1 s, b = 0, plus the offset, width and amplitude rows of Supp. 8.
- Protocols from 6.15: silencing with ensembles of 10, 20 and 30%; current injection with I ∈ {0.5, 2, 5} (3 in Fig. 4C); noise with gains {5, 10, 15, 20} and fractions {10, 25, 50, 75}%.
- Expected: silencing an active ensemble resets the sequence, to a location near the profile's minimum (Supp. 8). Current injection restarts the sequence from the injected ensemble when I ≥ 2. Noise leaves the sequence intact except at high gain and fraction.
- New measurement: predict the reset location from the profile alone, as the argmin of the profile relative to the silenced ensemble. Report prediction error. WP4 reuses this.

### WP0.6 Downstream scaffolding (Fig. 5, Supp. Fig. 10)

- Upstream: a trained network with a = 0.5, σ = 1 s, b = 0.3, run for 8.5 s.
- Downstream: 10 unconnected linear units with the update in 6.8, 1000 iterations, α = 0.015.
- Sweeps: downstream width σ\_d ∈ {0.4, 0.6, …, 2.2} s; downstream amplitude; downstream offset; both perturbations of the upstream sequence.
- Expected: correlation ≥ 0.99 for σ\_d ≥ σ\_u = 1 s, falling to about 0.8 at σ\_d = 0.4 s; insensitive to amplitude and offset.
- New measurement: the readout norm ‖A‖₂ against the Suppl. §7 lower bound across σ\_d.

### WP0.7 Experimental data (Fig. 3, Supp. Figs. 6–7, 9F–K)

- Blocked on \[DATA\]. Use the pipelines in 6.18 with g = 0.05 and N equal to the number of cells.
- Outputs: target versus response correlations (expected 0.90 for bat; 0.95 for mouse), mean matrices showing band structure, profiles, the Gaussian fits, the synthetic matched controls (bat a = 0.83, σ = 1.72 s, b = 0.05; mouse a = 0.29, σ = 1.68 s, b = 0.15), motif comparison (mouse τ\_div about 0.72), and the perturbation responses. The mouse network should fall into an attractor state after structured perturbations (Fig. 4H).

### WP0.8 Analytic check of the supplement (new)

This is the key addition. For every condition in WP0.1 and WP0.2:

1. Compute λ̃\_k from the mean learned profile, for |k| ≤ 10.
2. Compute λ\_k predicted by Cor. 3.2 under both `error_target` conventions, and the §5 shrunk version with α matched to p. Note that RLS from J0 corresponds to a penalty of 1/p, so the effective α is 1/p rescaled by the number of updates; derive the exact factor and record it.
3. Report per-mode relative magnitude error and phase error, and which convention fits better.
4. Repeat with training run under each convention.
5. Repeat with the batch ridge reference `batch_ridge_W` on the recorded rates, to separate RLS-path effects from the idealization.

Acceptance: for modes with |f̂\_k|² well above the shrinkage scale, the better-fitting convention gives relative error below 10%. If neither does, report the discrepancy as a finding rather than tuning parameters.

### WP0.9 The self-connection spike (new)

Identify why the diagonal is strongly negative, if WP0.1 confirms it. Run ablations of noise (gain 0 versus 0.5), p ∈ {0.008, 0.08, 0.8}, `c_uses_updated_P`, `error_target` and `state_reset_each_iteration`. Report the mean diagonal for each, and its effect on Floquet multipliers (6.16).

### WP0 acceptance summary

| Item | Criterion |
| --- | --- |
| WP0.1 | Matrix correlation < 0.15 and profile correlation > 0.95 across realizations |
| WP0.2 | All five qualitative trends reproduced; feature values within a factor of 1.5 of Supp. Fig. 3 |
| WP0.3 | τ\_div, τ\_con, τ\_chn below 0.05 in absolute value; τ\_rec trend reproduced |
| WP0.4–0.6 | Qualitative reproduction of the figures named |
| WP0.8 | Reported per mode, with a stated winning convention or a documented discrepancy |

If a reproduction fails, first sweep the \[VERIFY\] switches, then report. Do not change any other Methods parameter without logging it as a deviation.

## 8. WP1: Does synaptic decay collapse the degeneracy?

WP1 tests H1: with synaptic decay (Thesis A9, eq. 3.18) added to Braun's FORCE, independently trained matrices converge to one circulant matrix while the sequence is preserved. This is the cleanest falsifiable link between the thesis and the preprint, and it is cheap.

**Rationale.** RLS from J0 is ridge regression centred on J0 (Section 3.4, fact 4). The non-circulant part of the learned W is therefore the random initial matrix with the active subspace projected out, which is why realizations differ. Thesis §3.5 shows that ongoing decay drives components outside the observation subspace to zero. The same should hold here.

**Conditions.** Target as WP0.1 (a = 0.5, σ = 1.7 s, b = 0) plus one sharper target (a = 0.6, σ = 1 s, b = 0). For each:

| Condition | Switch settings |
| --- | --- |
| Baseline | `weight_decay = 0`, `ridge_centre = init` (WP0.1) |
| Decay sweep | `weight_decay` λ\_J ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3} per update |
| Zero-centred ridge | `ridge_centre = zero` |
| Decay with small init | λ\_J as above with g = 0.05 |
| Batch reference | `batch_ridge_W` with β swept, on rates recorded from the baseline, with J0 and with J0 = 0 |

Decay continues during training only; the test iterations use frozen weights. Run 300 iterations; if residuals are still falling, extend to 1000 and report both.

**Measurements.**

1. Matrix–matrix correlation across realizations (distribution, as Fig. 1E).
2. Residual fraction ‖W − W\_cp‖\_F/‖W‖\_F against iteration. The prediction is exponential decay at a rate set by λ\_J.
3. Projection of the residual onto the initial matrix: corr(W − W\_cp, J0 − J0,cp) for each realization. The prediction is near 1 at baseline and falling with decay.
4. Active-mode eigenvalues λ̃\_k for |k| ≤ 10. The prediction is that they are unchanged by small decay and shrink by the §5 factor as decay grows.
5. Sequence quality: target–response correlation in test iterations.
6. Stability: Floquet multipliers (6.16) for 5 realizations per condition. This checks whether the random bulk contributes to stability.
7. Perturbation responses (6.15, silencing at 20%): are responses identical across realizations once matrices converge, and do they differ from baseline? This feeds WP4.

**Acceptance.** H1 is supported if, at some λ\_J, the median matrix correlation exceeds 0.9 while the test target–response correlation stays within 0.02 of baseline. H1 is refuted if every λ\_J that raises matrix correlation above 0.5 also degrades the sequence by more than 0.05, or if the sequence collapses. In that case, report whether the random bulk was stabilizing the orbit (from item 6).

**Deliverable.** One figure with matrix and profile correlation distributions against λ\_J, and one with residual fraction against iteration. This is a candidate COSYNE result (Section 16).

## 9. WP2: Normative connectivity profiles in closed form

WP2 composes the thesis optimum with the Braun map to predict connectivity profiles from readout targets, without training. It also runs the map backwards: for any measured bump, including the MEC and bat fits, it returns the readout spectrum for which that bump is Wiener-optimal. WP2 needs no data and no training for its core outputs.

### WP2.1 Forward atlas: target spectrum to profile

Pipeline per condition (Section 3.4, fact 1):

1. Target classes: the six in Thesis App. I.7 (constant, single mode m0, flat to bandwidth B, Gaussian PSD, polynomial PSD, narrowband at carrier ξ0), plus "Gaussian bump equivalent" targets chosen so that a\* matches Braun's Gaussian bumps (σ ∈ {1, 1.7} s).
2. Water-fill with `water_fill` (eq. 7.11–7.13) at N = 501. Set K0 equal to the non-DC energy of Braun's reference bump (a = 0.5, σ = 1.7 s, b = 0) so magnitudes are comparable. Sweep the decay λ log-spaced over {1e-3, 1e-2, 1e-1, 1} × max\_m a\_m of that reference.
3. Build the kernel with each `sign_policy` and the default `offset_policy` (6.12), then the travelling sequence with T = 8.5 s and τ = 0.3224 s.
4. Apply `required_eigs` under both conventions, then invert to the profile.
5. Record: active modes of the profile (|λ\_k| above a tolerance), predicted rank 2M\* + 1, even and odd parts, and the features from 6.13.

Checks inside WP2.1:

- Rank: the number of active profile modes equals 2M\* + 1 for every condition.
- Odd part: equals ωτ times the spatial derivative of the even part, to numerical precision under the `supplement` convention (Section 3.4, fact 2). Report how far the `methods` convention departs from it.
- Linear-regime sensitivity: for each condition, compute ∂λ\_k/∂a\_k by finite differences. Report how much amplitude information the profile carries as the bump's rate range shrinks toward the sigmoid's linear part (Section 3.4, fact 3). Scale the kernel amplitude down by factors {1, 0.5, 0.25, 0.1} at fixed shape.

### WP2.2 Training check

For a subset (Gaussian PSD, flat to B, narrowband; one λ; positive signs), train Braun's FORCE on the normative sequence as target (WP0 settings, 20 realizations). Compare the learned mean profile with the WP2.1 prediction per mode, as in WP0.8. This verifies that normative sequences are learnable and that the map holds for non-Gaussian shapes, including bumps with negative side lobes.

### WP2.3 Inverse map: implied readout spectra

For a bump with powers a\_m, eq. 7.10 inverted gives the target spectrum for which it is optimal, up to the scale ν: S(m) ∝ (a\_m + λ)³ on active modes (6.11, `implied_psd`).

Apply it to:

- Braun's synthetic bumps across the WP0.2 grid.
- The fitted MEC bump (a = 0.29, σ = 1.68 s, b = 0.15) and bat bump (a = 0.83, σ = 1.72 s, b = 0.05) from Braun Supp. Fig. 6. These come from the paper, so this step needs no data access.
- When data arrive, the measured bump spectra themselves, rather than their Gaussian fits.

For each, report the implied S(m) for the λ grid, the implied target autocorrelation width, and the active-set threshold. For λ much smaller than a\_m and a Gaussian bump with power a\_m ∝ exp(−β\_a m²), the implied target spectrum is S ∝ exp(−3β\_a m²).

**\[VERIFY\] against the thesis.** Thesis App. I.4 states that the optimal kernel is broader than the target by a factor √3 (σ\_k = √3 σ\_y, also in the App. I.7 table). The algebra in I.12 gives a\* ∝ exp(−βm²/3), a spectrum broader than S ∝ exp(−βm²). A broader spectrum means a narrower spatial profile, so the kernel's autocorrelation length should be shorter than the target's by √3, not longer. The text appears to treat 1/√β as a length, whereas a Gaussian spectrum exp(−βm²) has an autocorrelation length proportional to √β. Implement the computation numerically, compare spatial autocorrelation widths of target and kernel directly, and report which direction holds. Do not encode either claim as a test until Max confirms.

### WP2 outputs

- An atlas figure: rows are target classes, columns are λ values; each panel shows the kernel, the profile (even and odd parts) and the predicted rank.
- A table of implied target spectra for the MEC and bat bumps.
- A short note on how much of the profile is fixed by rank alone (linear regime) versus by sigmoid curvature.

### WP2 acceptance

| Item | Criterion |
| --- | --- |
| Rank | Profile active-mode count equals 2M\* + 1 in every condition |
| Odd part | Matches ωτ ∂\_θ (even part) with relative error < 1e-6 under `supplement` |
| Training check | Learned profile matches prediction within the WP0.8 tolerance on modes above the shrinkage scale |
| App. I | All six closed forms reproduced by `water_fill`, except the width statement under \[VERIFY\] |

## 10. WP3: Learning the bump from the readout

WP3 trains networks on a readout target only, never on the sequence, and asks whether the bump's harmonic allocation converges to the thesis optimum. It proceeds from the thesis's aggregate system (WP3a) through amplitude-controlled rings (WP3b) to neuron-level inhibitory weights trained by RLS (WP3c), then adds Dale's law (WP3d).

### 10.1 A prediction to test first: which objective sets the exponent

Two slow-loop rules are natural, and they descend different objectives:

- **Thesis rule (eq. 9.10)** descends the reconstruction error alone, Σ λ²S/(a + λ)². Its fixed point is the cube-root law.
- **Error × readout-weight rule.** This is what an RLS update with feedback through the readout weights does (WP3c). By the envelope theorem, −2 ε̂\_k ŵ\_k is the gradient, with respect to the kernel amplitude, of the full ridge objective L\* = Σ λS/(a + λ). Stationarity under the budget gives λS/(a + λ)² = ν:

```latex
a_m^{\text{ridge}} = \max\!\Big( \sqrt{\lambda S(m)/\nu} - \lambda,\ 0 \Big)
```

So the neuron-level rule is predicted to converge to a **square-root** water-filling law, not the cube-root law, unless its update is reweighted. Measure the exponent in every WP3 stage. This sharpens H3.

### 10.2 Shared setup

- Readout target s(t), periodic with period T = 8.5 s, built by `readout_target_from_psd` for the target classes of WP2.1. Use 3 random phase draws per class; results must agree across draws.
- Fast loop: readout learned either by the delta rule with decay (Thesis eq. 3.18) or by RLS. Both must converge to the closed-form ridge solution `ridge_readout_closed_form` (test in Section 14).
- Slow loop: one of the two rules above. Timescale ratio (Thesis eq. 9.28) swept over slow/fast step-size ratios {1e-1, 1e-2, 1e-3}. Log the readout's distance from its ridge optimum at each slow step.
- Budget: either ν updated toward Σ a = K0 (6.9, default) or, in network models, a homeostatic constraint on Σ\_i r\_i² (Section 3.4, fact 6).
- Per-mode error estimate: temporal DFT of e(t) over whole cycles at frequency mω (Section 3.4, fact 5).

### WP3a Aggregate system (reproduce Thesis §9)

Use seqnet.thesis directly (WP-T): the kernel parametrized by a\_m and its LearningSystem (eq. 9.31). N ∈ {64, 128, 501}.

- Start from the WP-T reproductions of Thesis Figs. 8 and 9.
- Run the thesis rule and the error × weight rule side by side and fit the exponent of a\_m against S(m) on active modes (expected 1/3 and 1/2 respectively, after adding λ back).
- Acceptance: thesis rule within 5% of eq. 7.13 on active modes, with the active set exact; error × weight rule within 5% of the square-root law, or a documented reason why not.

### WP3b Amplitude-controlled travelling ring

Use `gain_ring` (6.5) with harmonics up to K\_max = 12 (N = 128 for development, 501 for final runs) and the rigid-rotation term so the bump travels at ω = 2π/8.5 s. The slow loop updates the set points A\_k (converting between a\_k and A\_k with the normalization fixed in 6A.5, under which a cosine of amplitude A\_k has a\_k = N·A\_k²/4 or N²·A\_k²/4; test it), and the network actually runs.

- Measure: realized |z\_k| against set points (tracking error), harmonic mixing from the sigmoid or rectification, and convergence of realized a\_m to the target law.
- Variant: held-mode marginality sweep (projector eigenvalue 1 versus 0.95, 0.9).
- Acceptance: realized allocation within 10% of the WP3a fixed point for the same rule.

### WP3c Neuron-level inhibition trained by RLS

Architecture (`ei_ring`, 6.6):

- E ring: either `gain_ring` with soft amplitudes (held modes marginal, no amplitude control) or a Braun-trained network for a Gaussian bump. The E→E weights stay fixed.
- I ring: N\_I = N\_E neurons with fixed translation-invariant E→I weights U. Default U is a Gaussian kernel of width 0.3 rad; variant: per-harmonic I subrings with read kernel cos(k(θ\_i − θ\_j)).
- Trained I→E weights V, initialized to a translation-invariant baseline V0 (default zero; nonnegative baseline in WP3d).
- Readout z\_out = wᵀr\_E, trained by the fast loop.

The slow update for V is RLS with the scalar readout error broadcast to E neurons through a feedback vector b:

```latex
V \leftarrow V - c\; b\, e(t)\, (P_I\, r_I(t))^\top, \qquad e(t) = z_{\text{out}}(t) - s(t)
```

Feedback variants, each a config value:

| Variant | b | Purpose |
| --- | --- | --- |
| `readout` | b = w (current readout weights) | Gradient of the ridge objective; predicted square-root law |
| `random` | Fixed random b | Feedback-alignment control |
| `quadrature` | Two channels: b\_c with e(t), b\_s with e delayed by T/(4k) per harmonic | Cancels the counter-rotating component |
| `staggered` | M readouts at phase offsets 2πj/M, M ∈ {1, 2, 4, 8} | Makes the loss approximately shift-invariant |
| `thesis_weighted` | b = w, with the per-mode update reweighted to match eq. 9.10 (implement in the Fourier domain) | Predicted cube-root law |

Measurements:

1. Effective recurrent connectivity W\_eff = W\_EE − V U, its profile and eigenvalues; per-mode gain change compared with the WP3a allocation for the same rule.
2. Exponent of realized a\_m against S(m).
3. Residual W\_eff − W\_eff,cp: its norm; the correlation of its row and column norms with |w|; τ\_div and τ\_con (6.14); column-sign consistency (fraction of columns whose nonzero entries share one sign).
4. Sweep of E-ring marginality: learning should be cleanest near-marginal and stall when the E ring holds its amplitudes rigidly. Report where.
5. Neuron heterogeneity of the trained network: spread across neurons of peak rate, width and baseline.

Acceptance: for `readout` or `staggered` feedback, the per-mode gains of the circulant part correlate with the WP3a allocation for the same objective at r > 0.9. Report all variants regardless.

### WP3d Dale's law

Repeat WP3c with V = V0 + ΔV constrained to be nonnegative, with V0 a translation-invariant baseline large enough that V stays nonnegative. A constant kernel is pure mode 0, so the baseline only loads the uniform mode. Hold that mode with a global inhibitory feedback loop on total E rate, not a fixed constant, because learned changes to V load the uniform mode in proportion to the learned gains. Enforce nonnegativity by clipping after each update and report how often clipping is active. Acceptance: same as WP3c within 10%.

## 11. WP4: Travel, asymmetry, stability and the T = 2M claim

WP4 tests three things. Does the derivative relation (Section 3.4, fact 2) explain the profile asymmetry Braun et al. measure? Does the profile alone determine perturbation responses, as Result 4 claims? And does the thesis rule T = 2M (eq. 9.27) survive once speed and shape are separate knobs?

### WP4.1 Asymmetry from the derivative relation

- For every condition in the WP0.2 offset sweep, predict the odd part of the profile from the even part: odd = ωτ ∂\_θ(even), under each training convention.
- Compare with the learned profiles and with Braun's asymmetry feature (eq. 13). Report how much of the measured rise in asymmetry with offset (Supp. Fig. 3A) the relation explains.
- Speed sweep: retrain with T ∈ {4.25, 8.5, 17} s at fixed temporal shape scaled to keep σ/T fixed, and separately at fixed σ. The imaginary parts of λ̃\_k should scale with ω = 2π/T, and the even part should not change when only speed changes.

### WP4.2 Does the profile determine perturbation responses?

Braun et al. attribute reset locations to the profile minimum (Supp. Fig. 8). Construct networks that share an identical profile but differ in their non-circulant part, and compare their responses:

1. Baseline FORCE realizations (WP0), which share a profile and differ in their residual.
2. Decay-trained realizations (WP1), with residual near zero.
3. Pure circulant networks W\_cp built from the WP0 mean profile.
4. W\_cp plus a random residual R drawn so that R P\_D = 0 (Prop. 4.1), scaled to {0.5, 1, 2} × the baseline residual norm.

All four classes support the same sequence to first order. For each, run the silencing and current-injection protocols (6.15) and compute Floquet multipliers (6.16). Report the across-class spread of reset location and correlation matrices CC, against the within-class spread.

H4 (second part) is supported if classes 3 and 4 differ in perturbation response by more than the within-class spread. That would mean the residual matters for responses even though it does not matter for the sequence.

### WP4.3 The T = 2M claim

The thesis derives T = 2M from a matched-projection assumption: traversal speed proportional to 1/bump width (App. K.3, eq. K.11–K.13). In the Braun map, speed enters only through the odd part, independently of shape.

- In `driven_ring` with an asymmetric component set by the derivative relation, vary width and speed independently. Confirm the network supports every combination, which means T = 2M is not forced by the dynamics.
- Separately, test the part of the thesis claim that remains meaningful: at fixed kernel, the reconstruction error as a function of observations per epoch T\_obs should show the aliasing term (eq. 6.8) vanishing once T\_obs ≥ 2M\*. Use the delta-rule readout trained only at T\_obs sample times per cycle.
- Acceptance: report both results. If the first holds, flag eq. 9.27 for revision in the preprint version of the thesis (Section 15).

### WP4.4 Stability atlas

For the WP0.2 grid and the WP2.2 normative networks, report the leading transverse Floquet multipliers and their Fourier content. This is the stability information the support condition omits, and it decides whether the normative bump is an attractor, not just a solution.

## 12. WP5: Heterogeneity, Dale's law and the MEC comparison

WP5 explains the band structure Braun et al. found only in networks trained on real data (Fig. 3D, 3F; Supp. Fig. 7), tests whether Dale-compliant members of the degenerate family exist, and compares the thesis's rank and spectrum predictions with the MEC sequences. WP5.1 and WP5.2 are synthetic; WP5.3 needs \[DATA\].

### WP5.1 Where does banding come from?

H5: banding comes from heterogeneity across neurons and weak initialization, not from training on sequences rather than readouts. Note that Braun's experimental and synthetic networks were both trained on per-neuron sequence targets, so the training target cannot be the cause there.

Design a factorial grid (the MEC-data level takes no added heterogeneity), trained with the Braun FORCE variant (WP0 settings, 20 realizations):

| Factor | Levels |
| --- | --- |
| Initial gain g | 1.5, 0.05 |
| Target source | Synthetic Gaussian matched to the mouse fit (a = 0.29, σ = 1.68 s, b = 0.15); MEC data (after \[DATA\]) |
| Heterogeneity added to the synthetic target | none; per-neuron baseline; per-neuron gain; per-neuron participation (some neurons weakly or not tuned); all three plus observation noise |

Heterogeneity distributions should be estimated from the MEC data when available. Until then use: baseline ∼ Uniform(0, 0.3); gain ∼ LogNormal(0, 0.5); 20% of neurons with participation reduced to 0–0.3; observation noise matched to target SNR of 3.

Measurements: τ\_div, τ\_con (6.14; the mouse data network has τ\_div ≈ 0.72 per Supp. Fig. 7), column-sign consistency, the column-norm distribution, and the profile. Also correlate each neuron's column norm with its baseline, gain and participation.

Acceptance: H5 is supported if heterogeneity at g = 0.05 reproduces τ\_div above 0.5 and visible column stripes, while the homogeneous control at g = 0.05 does not.

### WP5.2 Dale-compliant members of the degenerate family

Prop. 4.1 says every W\_D + W̃ with W̃P\_D = 0 supports the sequence. Ask whether some W̃ makes every column single-signed, given an E/I label per presynaptic neuron.

- Formulate as a linear feasibility problem per column: for neuron j labelled E, all entries of column j of W\_D + W̃ are ≥ 0; labelled I, ≤ 0. The constraint W̃P\_D = 0 couples columns, so solve jointly with `scipy.optimize.linprog` (or a sparse LP solver) for N ∈ {64, 128, 256, 501}. Minimize ‖W̃‖₁ to get the sparsest compliant matrix.
- Label schemes: random 80/20 E/I; E/I alternating in space; I neurons clustered.
- Report feasibility against the E fraction and N, the norm of the minimal W̃, and the stability of feasible solutions (6.16).
- Also report what the resulting compliant matrices look like: band structure, motif statistics, and whether the I columns cluster in sequence position.

### WP5.3 Breaking translation invariance (Thesis App. E, M)

Thesis App. E predicts that position-dependent kernels add an error of order ε² (the first-order term cancels under stationarity). Test it:

- In the WP3a aggregate system, make the kernel position-dependent, k(r; c) = k(r) (1 + ε h(c)) with smooth h, for ε ∈ {0, 0.01, …, 0.3}.
- Measure the excess reconstruction error and fit its exponent in ε (expected 2).
- In WP3c networks, impose position-dependent U (clustered I projections) and measure the same quantity.

The bilevel conjecture (App. M.3) is out of scope for implementation here, but log the aggregate observation energy A\_m (eq. M.1–M.2) in every WP5.3 run so later work can test it.

### WP5.4 MEC comparison (needs \[DATA\])

1. Compute the empirical MEC bump spectrum from the centred mean sequence (Supp. Fig. 6B) with the unitary DFT, the effective rank at tolerance ε ∈ {1e-2, 1e-3}, and the Suppl. §2 cutoffs D\_f(ε) and D\_y\*(ε).
2. Apply the WP2.3 inverse map to the measured spectrum (not just the Gaussian fit), and report the implied readout spectrum and its uncertainty under resampling of sequences.
3. Compare the learned MEC network's profile (WP0.7) with the profile predicted from the measured bump by the Braun map, per mode.
4. Report per-neuron heterogeneity statistics for WP5.1.

## 13. WP6: Is the MEC sequence generated locally or inherited?

WP6 fits three model classes to the same MEC data and finds the perturbation experiments that would tell them apart. Braun et al. assume local autonomous generation, and the thesis formalism is input-driven (eq. K.6), so the program needs this explicitly. Needs \[DATA\].

### 13.1 Model classes

| Class | Recurrent J | Input | Training (no BPTT) |
| --- | --- | --- | --- |
| A: autonomous | Trained | Noise only | Braun FORCE variant (WP0.7) |
| B: driven recurrent | Trained | Low-dimensional oscillatory input u(t) = B \[cos ω\_u t, sin ω\_u t, …\] with fixed ω\_u matched to the sequence period; harmonics up to K\_u | RLS on J and B jointly, treating u(t) as extra regressors |
| C: feedforward | J = 0 | As in B, with K\_u large enough to fit | RLS on B only (the downstream readout setting of Braun Result 5) |

For class B, sweep K\_u ∈ {1, 2, 4, 8} and report how much recurrent structure the fit still needs: the profile norm and ‖W − W\_cp‖ against K\_u. For class C, the fit quality against K\_u gives the input dimensionality needed to produce the sequence without recurrence.

### 13.2 Comparisons

1. Fit quality on held-out cycles: train on the first 15 of 20 sequences, test on the last 5.
2. Cycle-to-cycle period variability: does each class reproduce the spread of sequence durations in the data? (Braun Methods rescaled sequences to 177.8 s, so the raw durations vary.)
3. Perturbation predictions with the 6.15 protocols. Local silencing in class A should reset or restart the sequence (Result 4). In class C, only the perturbed cells should change, and the sequence continues on schedule elsewhere. Report the discriminating protocol, ensemble size and expected effect size for each pair of classes.
4. Near-marginality: the MEC period of about 178 s with τ = 0.32 s implies a slow mode. Report the leading Floquet multipliers of class A fits and how close to 1 they must sit.

### 13.3 Output

A one-page table of experimental predictions that distinguish A, B and C: each with the manipulation, the measurement, and the predicted outcome per class. This is a deliverable for discussions with the Gonzalo Cogno lab.

## 14. Validation suite

These tests must pass before the WPs that depend on them run. They are fast (small N, under five minutes in total) and live in `tests/`. Each test names its source.

| ID | Test | Source | Tolerance | Needed by |
| --- | --- | --- | --- | --- |
| T1 | Circulant from profile has eigenvectors v\_k with eigenvalues from `eigs_from_profile`; profile round trip | Braun Suppl. §3 | 1e-12 | all |
| T2 | Parseval for the unitary DFT, and the relation between a\_m and Σ\_i x\_i² fixed in 6A.5 | Thesis §1.13; Section 3.4 fact 6 | 1e-12 | WP2, WP3 |
| T3 | Kernel ↔ temporal profile conversion and \|k̂\_m\|² = N\|f̂\_m\|² | Section 4.2 | 1e-10 | WP2 |
| T4 | Numerical temporal Fourier coefficients of the wrapped Gaussian target match the closed form | Braun Suppl. eq. 3 | 1e-8 relative | WP0, WP2 |
| T5 | Image truncation: n\_images = 6 vs 20 | Braun eq. 5 | 1e-12 | all |
| T6 | Any W = W\_D + W̃ with W̃P\_D = 0 satisfies W f^(D)(t) = y\*^(D)(t) | Braun Suppl. Prop. 3.1, 4.1 | 1e-10 | WP0.8, WP4, WP5.2 |
| T7 | Circulant built by `required_eigs` (supplement convention) and simulated with a linear φ reproduces the D-mode sequence over 3 periods from the exact initial state, without noise | Cor. 3.2 | rel. error < 1e-4 | WP2 |
| T8 | Odd part of the predicted profile equals ωτ ∂\_θ (even part) | Section 3.4 fact 2 | 1e-8 | WP2, WP4 |
| T9 | Batch ridge `batch_ridge_W` equals the final RLS weights on a fixed recorded trajectory (open loop, no noise, P(0) = pI, enough passes) | Section 3.4 fact 4 | 1e-6 relative | WP0.8, WP1 |
| T10 | Delta rule with decay converges to the ridge closed form (Thesis eq. 3.13) and per-mode rate matches eq. 3.19 | Thesis §3.4–3.5 | 1e-3 | WP3 |
| T11 | `water_fill` reproduces App. I closed forms: constant, single mode, flat to B (K0/(2B+1) per active mode), Gaussian PSD exponent β/3, polynomial exponent −2s/3, narrowband width ℓ√3; budget satisfied | Thesis eq. 7.11–7.13, App. I.7 | 1e-8 | WP2, WP3 |
| T12 | Thesis kernel rule in the aggregate system converges to eq. 7.13 | Thesis eq. 9.10, 9.29 | 2% | WP3 |
| T13 | Error-times-weight rule converges to the square-root law (Section 10.1) | This spec | 2% | WP3 |
| T14 | Per-mode error from the temporal DFT of e(t) equals ε̂\_m from the spatial decomposition for a travelling bump | Section 3.4 fact 5 | 1e-8 | WP3 |
| T15 | Motifs of a centred circulant: τ\_div = τ\_con = τ\_chn = 0 | Section 6.14 | 1e-12 | WP0.3, WP5 |
| T16 | Driven ring steady state matches r̂\_n = Î\_n/(γ\_n + inωτ) | Thesis eq. K.9 extended | 1e-6 | WP4.3 |
| T17 | Floquet code returns a multiplier of 1 (the phase direction) for any sequence-generating orbit | Section 6.16 | 1e-4 | WP1, WP4 |
| T18 | Scaffolding: minimal-norm ridge readout norm ≥ the Suppl. §7 bound for σ\_d < σ\_u | Braun Suppl. eq. 6 | holds for all tested | WP0.6 |

T7 uses a linear φ deliberately: a linear support condition reproduces the sequence exactly, but its amplitudes are neutral. Add the sigmoid version as a reported diagnostic, not a pass/fail test, because support does not imply stability.

**Reporting standard for every WP.** REPORT.md contains: the hypothesis, the configs run, a pass/fail table against that WP's acceptance criteria with the measured numbers, figures, every \[VERIFY\] switch value, deviations from this spec, and open problems found.

## 15. Open decisions and known ambiguities

Items in the first table are source ambiguities the agent handles with switches. Items in the second need a decision from Max before the WP named can be finalized.

### 15.1 Source ambiguities (implement default plus switch)

| # | Ambiguity | Where | Default | Alternative |
| --- | --- | --- | --- | --- |
| A1 | Training target: z is trained toward x\* (Methods eq. 7) but the supplement assumes support of y\* = x\* + τẋ\* (Def. 1.1). Under eq. 7 the realized activation lags by the leak's low-pass filter, about 0.31 s at k = 1 for ωτ ≈ 0.24 | Braun Methods vs Suppl. | `methods` | `supplement` |
| A2 | RLS step uses c with the updated P as printed, which scales the step twice relative to standard FORCE | Braun eq. 8–9 | as printed | standard |
| A3 | Whether the network state resets between training iterations | Methods lines 930–970 | reset | continue |
| A4 | Update times every 16 × 0.0081 = 0.1296 s versus 0.129 s target bins | Methods lines 938–945 | analytic target at update times | nearest bin |
| A5 | MATLAB `smoothdata` Gaussian window: kernel width for window 50 | Methods lines 995–998 | std = window/5 | parameter |
| A6 | Bat network size: N = 170 (line 774) versus 168 cells (lines 856, 866) | Methods | 168 | — |
| A7 | Downstream target bin size printed as 0.0001 s (line 1250); upstream input uses 0.0081 s | Methods | 0.0081 s | 0.0001 s |
| A8 | Fig. 5D title says width 2 s; Methods line 1254 and panel title say 0.5 s | Fig. 5 legend | 0.5 s | — |
| A9 | Supp. 10A–C parameters in Methods (σ = 0.4 s) differ from legends (σ = 1 s; 10A shows random weights) | Methods lines 1258–1260 vs legends | follow legends | Methods values |
| A10 | Fig. 5F–G axes span amplitudes 0.85–7.65 and offsets 0–3.4, unlike the Methods ranges | Fig. 5 | Methods ranges plus the figure ranges | — |
| A11 | Origin of the strongly negative self-connections | Supp. 1G, Fig. 1D | investigate (WP0.9) | — |
| A12 | Effective α of RLS in the §5 shrinkage formula, given P(0) = 0.08 I and the number of updates | Suppl. §5 vs Methods | derive and record (WP0.8) | — |

### 15.2 Decisions for Max

| # | Decision | Affects | Proposed default |
| --- | --- | --- | --- |
| D1 | Thesis App. I.4 and I.7 state σ\_k = √3 σ\_y (kernel broader than target). The algebra a\* ∝ exp(−βm²/3) gives a broader spectrum, hence a narrower spatial kernel. Confirm which length definition is intended, and whether the statement should be corrected in the preprint version | WP2.3, T11, thesis preprint | Treat as narrower by √3 pending confirmation; test numerically |
| D2 | Thesis eq. 9.27 (T = 2M) rests on matched projections (App. K.3); the Braun map makes speed independent of shape | WP4.3, thesis preprint | Test and report; do not encode as a model constraint |
| D3 | Which objective the slow loop should descend: reconstruction error (cube root, eq. 9.10) or the full ridge objective (square root, Section 10.1) | WP3, central claim | Run both and report |
| D4 | Whether the DC mode and the offset count against the budget K0 | WP2, WP3 | Exclude DC |
| D5 | Sign pattern of kernel coefficients for normative kernels | WP2 | Positive; report alternating |
| D6 | Joint budget conversion factor: κ in Thesis eq. 8.3 versus λ/2 in the Discussion's summary of eq. 8.9 | WP4.3 | κ as a parameter |
| D7 | Regime per WP: autonomous (Braun) or driven (thesis eq. K.6). The thesis introduction calls MEC periodicity intrinsic while its formalism is driven | WP3, WP6 | Autonomous E ring in WP3; both in WP6 |
| D8 | Data paths and access for MEC session 7 of mouse #60584 (with the PCA ordering) and the bat dataset | WP0.7, WP5, WP6 | Blocked until provided |
| D9 | Compute: development settings (10 realizations, 100 iterations) versus final settings (50, 300) per WP, and hardware available | all | Development first |
| D10 | DFT normalization: eq. 4.1 is unitary, but eq. 4.2 and 4.7 make the singular values of X equal \|k̂\_m\|, which holds only for the unnormalized DFT. The operational a\_m (squared singular value) is then N\|k̂\_m\|² under eq. 4.1, which rescales λ and K0 | WP-T and everything downstream | Use squared singular values as a\_m; confirm in TT1 |
| D11 | Parameter values (N, λ, K0, T\_obs, learning rates, target classes) used for thesis Figs. 2–9, which the text does not list | WP-T reproduction targets | Defaults in 6A.5 until Max supplies the originals |

## 16. Priorities and order of work

Build the shared core and the tests first, then a COSYNE slice that needs no data, then the full program. The COSYNE abstract deadline is Oct 18, 2026.

### 16.1 COSYNE slice (no data, minimal training)

1. WP-T core: the seqnet.thesis kernels, readout, errors, optimum and learning modules with tests TT1–TT13; then core modules 6.1, 6.2, 6.10 and tests T1–T8.
2. WP2.1 atlas and WP2.3 implied spectra for the published MEC and bat fits. Closed form, no training.
3. Rate RNN and RLS (6.3, 6.7), tests T9 and T17, then WP0.1 at development settings (10 realizations, 100 iterations).
4. WP1 at development settings: baseline versus three decay values on the WP0.1 target.
5. WP3a with both slow-loop rules, to report the cube-root versus square-root exponent.

Candidate abstract claims, each conditional on its result: normative profiles predicted from readout spectra (WP2); synaptic decay collapses connectivity degeneracy (WP1); the slow-loop objective sets the water-filling exponent (WP3a).

### 16.2 Full program order and dependencies

| Order | Work | Depends on | Data needed |
| --- | --- | --- | --- |
| 1 | WP-T (Section 6A), then core modules and the full test suite (Section 14) | — | no |
| 2 | WP0.1–WP0.6, WP0.8, WP0.9 at final settings | 1 | no |
| 3 | WP1 full | WP0.1, T9 | no |
| 4 | WP2.1–WP2.3 full, WP2.2 training check | WP0.8 | no |
| 5 | WP3a, WP3b | WP-T, T14 | no |
| 6 | WP4.1–WP4.4 | WP0.2, WP0.5, WP1, WP2 | no |
| 7 | WP3c, WP3d | WP3b, WP4.4 | no |
| 8 | WP5.2, WP5.3 | WP3c | no |
| 9 | WP0.7 | 2 | yes |
| 10 | WP5.1, WP5.4 | WP0.7 | yes |
| 11 | WP6 | WP0.7, WP4.2 | yes |

If time runs short, cut from the bottom of this table, never the tests.

### 16.3 Handover note for the coding agent

Start a session by reading this document, the Braun Methods (PDF pages 21–36), Braun Suppl. §1–§7, and Thesis §3–§4, §7 and §9 with App. I and K. Build WP-T first and keep seqnet.thesis importable on its own. Report progress per WP in its REPORT.md, and raise any new ambiguity by adding a row to Section 15 rather than deciding it silently.
