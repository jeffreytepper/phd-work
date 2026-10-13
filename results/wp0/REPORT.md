# WP0 report: reproducing Braun et al. (2026) (in progress)

Status 2026-10-13. Spec §7. Source: Braun, Nordal, Rambø, Clopath & Gonzalo Cogno (2026), bioRxiv 10.64898/2026.09.02.748960 v1, Methods and supplement.

## Hypothesis

H0: Braun's synthetic results reproduce from the Methods, and learned profiles match Suppl. Cor. 3.2 or §5 quantitatively under one of the two training conventions.

## Code (main package)

| Module | Content | Source |
| --- | --- | --- |
| `seqnet/conventions.py` | profile ↔ matrix ↔ eigenvalues, shift, DFTs (the only `np.fft` user) | Braun eq. 11, Suppl. §3 |
| `seqnet/targets.py` | wrapped Gaussian, sequences, logit target and its derivative, Fourier closed form | Braun eq. 4–6, Suppl. eq. 1–3 |
| `seqnet/models/rate_rnn.py` | Euler rate RNN with noise; silencing, current and noise perturbations | Braun eq. 1–3, 21–22 |
| `seqnet/training/rls.py` | FORCE variant with all spec 6.7 switches; presynaptic masks (a), (b) | Braun eq. 7–10, 16–17 |
| `seqnet/training/readout.py` | downstream rule; thesis wrappers | Braun lines 1264–1275 |
| `seqnet/analysis/` | profile features (MATLAB `smoothdata`/`findpeaks` emulation), motifs, metrics, perturbation grid | Braun eq. 11–15, lines 1130–1222 |
| `seqnet/spectral/braun_map.py` | required eigenvalues (both conventions), §5 shrinkage, cutoffs, scaffolding bound, batch ridge | Suppl. §2–5, §7 |
| `seqnet/experiments/` | cached parallel runs (`results/wp0/runs/<label>-<hash>/meta.json` with config, commit, versions, seeds, wall time); WP0 condition groups and CLI | spec 5.3–5.5 |

Tests in `tests/core/`: T1–T9 and T15 pass. T15 is corrected; see finding 1.

## Notebooks (`notebooks/wp0/`, one per figure, panel by panel)

`braun_fig1`, `braun_supp1`, `braun_fig2`, `braun_supp3` (Supp. 3–5 and Supp. 1F), `braun_fig4`, `braun_supp8`, `braun_supp9`, `braun_fig5` (Fig. 5 and Supp. 10), `wp0_8_supplement_check`.

Data notebooks: `braun_supp6` (bump fits), `braun_fig3` (Fig. 3 and Supp. 7), `braun_fig4_data` (Fig. 4G/H, Supp. 9F–K). Bat panels run now; mouse panels run unchanged once `configs/wp0/data.yaml: mec.path` is set and `seqnet.data.mec.load_session` reads the delivered file format.

**Not done:**
- MEC panels: Fig. 3E/F, Supp. 6A–C, Supp. 7 circles, Fig. 4H and Supp. 9I–K are blocked on [DATA].
- Supp. 2 (sparsity, WP0.4): all three mask types are implemented. Mask (c), one P_i per row, runs at reduced settings for fractions 0.75–0.95 only, because memory is N·L² per network.
- WP0.9 ablations.

## Settings

- **Fig. 1 (WP0.1):** final settings, 50 realisations × (300 + 10) iterations.
- **Everything else:** dev settings (spec D9), 10 realisations × (100 + 10) iterations.
- **Cost:** one realisation is ~0.47 s per iteration alone, about twice that with 11 parallel workers on this 6-core machine (memory-bandwidth bound). Runs use one worker per physical core by default.

## [VERIFY] switches (defaults used everywhere unless stated)

| # | Switch | Default used |
| --- | --- | --- |
| A1 | `error_target` | `methods`: input trained toward x\* (Methods eq. 7) |
| A2 | `c_uses_updated_P` | `true`, as printed. Test T9 shows this is *not* the ridge/RLS update; see finding 2 |
| A3 | `state_reset` | `true`: h reset to h_target(snippet start) each iteration |
| A4 | `target_time` | `analytic`: target at the exact update time t0 + k·dt |
| A5 | `smoothdata` Gaussian | std = window/5, window 50, end points shrink |
| A6 | bat N | 168 (not used yet) |
| A7 | downstream target bin | 0.0081 s |
| A8 | Fig. 5D width | 0.5 s |
| A9 | Supp. 10B parameters | legend (σ = 1 s) |
| A10 | Fig. 5F/G ranges | Methods values plus figure axes |
| new | perturbation noise | perturbed and unperturbed runs share the noise realisation (`common_noise`) |
| new | noise-perturbation gain | legend values (15, 20); the Methods text says 0.1 / 0.0015 |

## Findings so far

1. **T15 correction.** For any matrix with equal row and column sums (e.g. circulant), the centred motifs are exactly τ_div = τ_con = −1/(N−1) and τ_chn = −2τ_rec/(N−1). Spec 6.14 says they vanish identically; they don't, but they are O(1/N).
2. **The printed FORCE step is not RLS (A2).** Eq. 8 multiplies the standard Sussillo–Abbott update by c = 1/(1 + Rᵀ P_new R) a second time. Single-pass RLS without the extra c reproduces batch ridge (YFᵀ + J0/p)(FFᵀ + I/p)⁻¹ to 1e-6; the printed variant differs by more than 1e-3 (T9). Spec fact 4 (RLS = ridge centred on J0) therefore holds only for the standard update.
3. **Effective §5 α (A12).** Derived: α_eff = 1/(p·n_s·N), with n_s = updates per iteration × training iterations. It ignores the noise in the realised rates, which adds a ridge of its own (WP0.8 notebook).
4. **Methods typos noted while implementing:**
   - The current-injection formula writes W·φ(r_j); this must be φ(h_j).
   - The noise-injection gains in the text (0.1, 0.0015) contradict the legends (15, 20).
   - "Each training iteration is … 2086 time bins (= 16.9 s)" counts integration steps, not 0.129-s bins.
   - Supp. 10's Methods parameters differ from its legend (A9).

## Data (WP0.7)

**Bat.** Braun's bat input was 168 observer time cells (mean z-scored rates over all hanging intervals). The public deposit of Omer, Las & Ulanovsky (2023), Zenodo 10.5281/zenodo.7213099 (CC BY 4.0), holds figure-level data only. The closest matrix is `Fig_2.mat → panel_a.correct`: 166 observer time cells, z-scored, 0.1-s bins from landing, apparently correct trials only. It is used as a **proxy**; the 166-vs-168 difference and the trial selection are recorded deviations. The raw files live in `data/bat_omer2023/` (gitignored, not redistributed).

**Mouse MEC.** Blocked on [DATA] (D8). `seqnet.data.mec` implements the whole pipeline behind a single loader stub and is tested on a synthetic stand-in session. Every mouse cell in the notebooks is written and switches on when the path is set.

## Results

| Item | Reproduced | Paper | Verdict |
| --- | --- | --- | --- |
| WP0.1 median pairwise matrix correlation | 0.063 | ≈ 0.07 | pass |
| WP0.1 median pairwise profile correlation | 0.971 | 0.97–0.98 | pass |
| WP0.1 test target–response correlation | 0.88 | ≈ 0.8 | pass (higher) |
| WP0.1 mean matrix regenerates the sequence | corr 0.89 | yes | pass |
| WP0.1 mean \|ΔW\| still decreasing at iteration 300 | 0.0113 → 0.0110 | yes | pass |
| Supp. 1G off-diagonal cluster is the diagonal | mean self-weight −1.31, about 5 SD below the off-diagonal | cluster near −1.5 | confirmed |
| Bat bump fit (proxy data) | a 0.79, σ 1.19 s, b 0.064, R² 0.97 | 0.83, 1.72 s, 0.05, 0.99 | σ narrower (data proxy) |
| Bat target–response correlation | 0.94 | 0.90 | pass |
| Bat matched synthetic (our fit / paper fit) | 0.92 / 0.91 | 0.90 | pass |
| Bat mean matrices, data vs synthetic | corr 0.25 | 0.13 | pass (differ) |
| Bat profiles, data vs synthetic (our fit) | 0.90 (0.92 with d = 0) | 0.83 | pass |
| Bat profiles vs synthetic from the paper's fit | 0.68 | — | narrower proxy bump |
| Bat motifs: τ_chn, τ_con | −0.007, **0.09** | "absent" | τ_con fails (< 0.05 criterion) |
| Bat perturbations: I = 0.5 has no effect; effect grows with I | min corr 0.99 / 0.91 / 0.57 for I = 0.5 / 2 / 5 | yes | pass |

### WP0.2–WP0.4 (dev settings: 10 realisations × (100 + 10) iterations)

| Item | Reproduced | Paper | Verdict |
| --- | --- | --- | --- |
| Asymmetry, b ≤ 0.25 | 10–16 neurons | 10–20 | pass |
| Asymmetry, b = 0.45 | 54 | ≈ 75 | pass (within factor 1.5) |
| Range falls with width | corr −0.87 | yes | pass |
| Peak width falls with amplitude | corr −0.89 | yes | pass |
| Performance rises with amplitude, falls with offset | corr +0.97 / −0.94 | yes | pass |
| τ_rec rises with offset and width | corr 0.80 / 1.00 | yes | pass |
| τ_rec range | 0.01–0.47 | 0.55–0.8 | lower than the paper |
| \|τ_div\|, \|τ_con\|, \|τ_chn\| | max 0.054 | ≈ 0 | marginal fail (< 0.05 criterion) |
| Sparsity (a), random presynaptic | test corr ≥ 0.82 up to 75%; 0.64 at 90%, 0.39 at 95% | kept up to 95% | **fail** beyond 75% (dev settings) |
| Sparsity (b), consecutive block | ≥ 0.78 up to 30%; 0.68 at 35%, 0.31 at 50% | fails beyond ≈ 35% | pass |
| Sparsity (c), random synapses | 0.89–0.90 at 75–95% (quick settings) | kept up to 95% | pass |
| Profile corr with dense (25% presyn, 25% block, 85% synapses) | 0.87 / 0.83 / 0.88 | 0.93 / 0.91 / 0.94 | pass (within 0.1) |

### WP0.8: learned profiles against Suppl. Cor. 3.2 and §5 (new)

- **Fig. 1 condition (final settings).** At k = 1, both conventions match the learned eigenvalue: `methods` within 0.9%, `supplement` within 1.7%. k = 1 is the only mode above the strict strong-mode threshold (|f̂_k|² > 10³ α_eff).
- **All 28 WP0.2 conditions, k = 1.** Median relative error is 1.6% (`methods`) and 2.1% (`supplement`). It is under 10% in 100% and 96% of conditions respectively, and `methods` is the better convention in 71% of them.
- **k = 2.** Median error 2.4% / 3.1%; under 10% in 82% / 86% of conditions. The exceptions are high offsets (b ≥ 0.4, 30–50%).
- **k ≥ 3–4.** Errors are order 1 (median ≈ 100%). These modes lie below the shrinkage scale. The learned eigenvalues do *not* shrink toward 0 as §5 predicts; they sit on a floor at the self-weight (phase ±π). A diagonal of value s adds s to every λ_k, and the learned diagonal is −1.3.
- **Removing the self-weight** (d = 0 replaced by its neighbours) makes k = 1–2 worse (up to 4.6×). The diagonal is therefore part of how the learned matrix supports the active modes, not a separable artefact. This motivates WP0.9.
- **Verdict on H0's second part:** Cor. 3.2 holds quantitatively on the strongly represented modes (k ≤ 2) under either convention, with `methods` marginally better. Beyond them, §5's shrinkage toward zero is not what FORCE does; the learned spectrum is pinned by the strongly negative diagonal. Reported as a finding, no tuning.

**Interpretation of the bat motif result.** The data-trained mean matrix shows vertical stripes: presynaptic neurons project with consistent sign. Under Braun eq. 15 this registers as τ_con. Braun report banding only in data-trained networks (mouse τ_div ≈ 0.72); here it appears weakly in τ_con for the bat. τ_rec is 0.04 in the data network versus 0.63 in the synthetic control, matching Supp. 7's "mismatch between data and synthetic motifs". This feeds WP5.1 (banding from heterogeneity).

**Finding 5 (bump-fit procedure).** Braun take the offset as the minimum of the bump (Methods line 1114). For a bump that hasn't decayed to baseline within the 8.5-s window, this biases the fit: a true σ = 1.72 s returns σ̂ = 1.63 and b̂ = 0.086 (test `test_fit_bump_bias_of_minimum_offset`). The synthetic matched control built from fitted parameters is therefore not exactly the bump that was fitted.

**Finding 7 (reset location, Supp. 8 / WP0.5).** Braun write that a silenced sequence resets "as far away from the silenced ensemble as the position of the minimum in the connectivity profile".
- Across the seven Supp. 8 networks (all ensembles silenced in their active window), the profile's global minimum predicts the reset distance with a median error of 116 neurons, and the nearest local minimum with 78.
- The **zero crossing** of the smoothed profile on the reset side, where excitation turns into inhibition, predicts it with a median error of **13 neurons**.
- In the Fig. 4 network, resets land 70–104 neurons back. The profile minimum is at −230; the zero crossing is at ≈ −110.
- This sharpens Result 4 and is input for WP4.2.

**Finding 8 (downstream, Fig. 5 / WP0.6).**
- With the upstream network run with noise (Methods), the 1000-iteration readout rule plateaus at correlation 0.93 for wide targets (paper ≥ 0.99). Least squares reaches 1.000, and 20 000 iterations reach 0.99, so the target is representable; the rule just converges slowly on noisy input.
- With a noise-free upstream run, wide targets reach 0.994–0.999, matching the paper. The noise-free run is recorded as a [VERIFY] alternative.
- σ_d = 0.4 s gives 0.87 (paper ≈ 0.8). Correlation is flat in offset, and ≥ 0.97 over the figure's amplitude range.
- T18 holds: ‖A‖₂ = 4.6, far above the scaffolding bound (0.07–0.12).

**Finding 9 (noise perturbation scale).** Methods eq. 22 defines η = gain × ζ, where ζ already contains the base gain 0.5. Spec 6.3 drops that factor, so it is twice as strong as the source. The code follows the source.
- Even so, gains 5–20 into 10–75% of neurons lower the post-release correlation more than the legend's "sequences always continue" suggests. Visually the sequences do continue: the low correlations come from phase shifts after release, not from loss of the sequence.

**Finding 6 (engineering).** Adding a field to `ForceConfig` silently changed every cache hash, and a notebook started retraining Fig. 1. The hash now ignores late fields left at their defaults, and legacy hashes are still accepted.
