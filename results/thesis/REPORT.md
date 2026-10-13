# WP-T report: `seqnet.thesis` (in progress)

Status as of 2026-10-13 (core, tests, CLI, README and Figs. 2–9 notebooks done). Spec: `Design spec from readout function to sequence connectivity.md` §6A.

## Hypothesis

HT: rebuilding the thesis framework from the thesis text alone reproduces its closed forms and Figs. 2–9, and every discrepancy found is documented.

## What is built

The `seqnet/thesis/` subpackage imports only NumPy and SciPy and is checked by TT17/TT18. Modules and their sources:

| Module | Contents | Source |
| --- | --- | --- |
| `fourier.py` | DFTs, D10 normalization resolver, half-spectrum bookkeeping | eq. 4.1, 4.2, 4.7 |
| `ring.py` | `Ring`, `Schedule` | §2.1, §2.3 |
| `kernels.py` | five App. D families, `Kernel.from_alloc` with sign and offset policies, effective rank | §4.5, App. D, eq. 4.8 |
| `design.py` | x(c), design matrix, SVD, observation subspace, rank under sampling | eq. 2.1–2.6, 5.4 |
| `targets.py` | `TargetClass` by S(m): the six App. I classes, random-phase sampling, fixed targets | eq. 6.3, App. I |
| `readout.py` | pinv, ridge, per-mode ridge, filter, delta rule, RLS readout | eq. 3.11–3.19, 4.5–4.6 |
| `coherence.py` | R_k, μ, p_m, σ_ξ, σ_x, uncertainty product, d\*(ε) | eq. 5.1–5.13, App. C, D |
| `errors.py` | thesis three-term decomposition; exact expected error; corrected sampled-error formula | eq. 6.6–6.9, 6.6′ |
| `optimum.py` | water-fill (exact active-set solve and log-bisection), square-root law, eq. 7.17, joint optimum (thesis closed form and numeric), diagnostics, implied PSD, App. I cases | §7, §8, App. I |
| `learning.py` | `ThesisKernelRule`, `RidgeObjectiveRule`, per-mode error estimators (spatial and temporal), `LearningSystem` | eq. 9.5–9.10, 9.31; spec §10.1 |
| `field.py` | field equation K.6 (FFT), SOM chain and its inverse, travelling steady state, branching ratios | eq. 9.13–9.16, K.6–K.10 |
| `perturbation.py` | position-dependent design, coupling matrix, A_m, exact expected ridge error, coupling-error exponent | App. E |

Also built: the CLI (`python -m seqnet.thesis …`), `configs/thesis/`, the README, and `figures/fig2.ipynb` … `fig9.ipynb`, one notebook per figure, panel by panel (see Figure reproductions).

App. G is built: `water_fill_noisy` (eq. G.14–G.16), `snr` (G.8) and `noise_floor_estimate` (G.9), with tests in `tests/thesis/test_noise.py`.

**Not yet built:** App. L is only a stub marked [VERIFY]; App. M only logs A_m (`bridge_energy`).

## Tests (`tests/thesis/`, 72 tests, about 60 s, all passing)

| ID | Result | Notes |
| --- | --- | --- |
| TT1 | pass | Singular values = √N\|k̂_m\| (D10 confirmed), not \|k̂_m\| |
| TT2 | pass | F XᵀX F\* diagonal to 1e-10 |
| TT3 | pass | Per-mode ridge = matrix ridge; numerator carries K_m (matters only for asymmetric kernels) |
| TT4/T10 | pass | Expected dynamics: fixed point at 1e-6, per-mode rates within 5% of eq. 3.19. Stochastic: within 2e-2 (deviation: spec asks 1e-3; measured 0.2–0.8%, seed-dependent) |
| TT5 | pass | Tolerance max(2, 3%) because of the box's sinc sidelobes |
| TT6 | pass | Spectral = pairwise coherence, on and off grid |
| TT7 | pass, with caveats | Gaussian product = 1/2 to 1e-3. See discrepancies 5–6 |
| TT8 | pass | Exact for a band-limited target at T_obs = N. Thesis formula incomplete otherwise; see 3 |
| TT9/T11 | pass | Constant, single mode, flat (with ν from eq. I.9), Gaussian exponent β/3, polynomial −2s/3, narrowband spectral width ℓ√3. Width statements under D1 are not encoded |
| TT10 | pass | KKT to 1e-9 |
| TT11 | pass | Observable gradient (9.6) = analytic (9.5) to 1e-8 |
| TT12/T12 | pass, preconditioned | Within 2%, exact active set, for project/fixed/dual budgets. The plain rule fails; see 4 |
| TT13/T13 | pass, preconditioned | Square-root law within 2% |
| TT14 | pass | Chain round trip; static steady state; travelling steady state within 5e-3 (Euler) |
| TT15 | pass | Thesis closed form: T\* = 2M\*, budget within one step λ |
| TT16 | pass | Coupling-error exponent ≈ 2 |
| TT17, TT18 | pass | Import boundary; isolated copy runs its tests |

## Discrepancies found in the thesis (feeds the preprint)

1. **D10, normalization (confirmed).** Eq. 4.2/4.7 make the singular values \|k̂_m\|. With the unitary eq. 4.1 they are √N\|k̂_m\|. The code uses a_m := N\|k̂_m\|² (the squared singular value at T_obs = N), and K0 = N Σ_i k_i². Spec Section 3.4 fact 6 therefore carries the factor N.
2. **Delta rule vs ridge (new).** Eq. 3.18 with decay λ converges to ridge with regularizer **T·λ**, not eq. 3.13's λ. The thesis's own rate η(σ²/T + λ) (eq. 3.19) implies this (test `test_delta_rule_literal_decay_equals_ridge_T_lambda`). `DeltaRule` takes the ridge λ and uses λ/T internally.
3. **Error decomposition (new).** Eq. 6.7–6.9 are exact only for T_obs = N and a target band-limited below N/2. In general, for a kernel band-limited to \|m\| < T/2, the exact expected error per in-band mode is (1 − g̃)²S(m) + g̃² Σ_{j≠0} S(m + jT), with g̃ = ã/(ã + λ) and ã = (T/N)a_m. The thesis drops the alias-contamination term (it does not average out under stationarity; it is a sum of variances) and uses a_m instead of (T/N)a_m. `sampled_error_exact` matches the exact expectation to 1e-8. The noise term of eq. 6.6′ carries 1/T only when the error is normalized per observation.
4. **Kernel rule stiffness (new).** Eq. 9.10 with a single η is stiff. Its curvature is 6S/λ² at a = 0 and 3ν/(a + λ) at the optimum, a ratio of ((a + λ)/λ)⁴ for an active mode. That is about 10⁷ in the TT12 test case (a/λ ≈ 80) but only about 10³ in the regime of thesis Figs. 8–9 (a/λ ≈ 5, unitary λ = 0.1). There the literal rule converges to machine precision within about 1000 kernel epochs (Fig. 8 notebook). The problem bites at small λ: Fig. 9's λ = 0.003 curve is still falling after 2000 epochs, in the thesis figure and here.
   - Any stable η takes more than 10⁶ steps (test `test_plain_thesis_rule_is_stiff`).
   - A practical η collapses the allocation onto a few modes.
   - Fix (not in the thesis, off by default): precondition each mode's step by the inverse of the observable Hessian, (a_m + λ)/(3G_m), where G_m is the rule's own gradient term. This keeps the rule S-free and its fixed point, and converges to machine precision. The same holds for the square-root rule with factor 2. This affects the claim in §9.2/9.5 that the rule is "convergent in the LMS sense".
5. **σ_x definition (new, minor).** Under eq. 5.10, a Gaussian k ∝ e^{−r²/2σ²} has σ_x = σ/√2. The text and Table 5.1 take σ_x = σ, and σ_ξ² = 1/(2σ)² in §5.4, whereas eq. 5.8 gives 1/(2σ²). The two errors cancel in the product.
6. **Uncertainty bound (new).** The Gabor does **not** saturate σ_xσ_ξ ≥ 1/2: under eq. 5.8, σ_ξ² includes ξ0². This contradicts Table 5.1 and §8 ("Gaussian and Gabor saturate"). On the periodic ring the bound fails for broad kernels (σ = 3 rad gives a product below 1/2), so Theorem 5.1 needs a localization condition.
7. **D1, √3 direction (confirmed numerically: narrower).** For S ∝ e^{−βm²}, the kernel's autocorrelation half-width over the target's is **0.577 = 1/√3** at β = 0.002 and β = 0.01. The kernel is narrower, not broader as App. I.4/I.7 state. Eq. I.19 (σ_k² = 3/ℓ²) is inconsistent with I.18; the spectral width √3ℓ implies an envelope σ_k = 1/(√3ℓ). Not encoded in tests pending Max's confirmation.
8. **App. I.4 cutoff (new).** The active band is \|m\| < √((1/β) ln(2S0/(νλ))). The text prints (1/β) ln(…) without the square root (test `test_TT9_gaussian_psd_active_cutoff_has_square_root`).
9. **D2/D6, joint optimum.** With κ = λ/2, the closed form (8.9) gives T\* = 58, 70, 80 for B0 = 5, 20, 80 (Gaussian PSD, β = 0.01, λ = 1e-2, N = 128). Direct numeric minimization of E\*(T) over even T gives T\* = 54, 66, 78. T\* = 2M\* does hold at the numeric optimum, but the closed form overshoots by 2–4 samples. Eq. 8.9 has no solution in general, because M\*(ν) is a step function; the residual is bounded by λ.
10. **Figure-level findings (new).**
    - Fig. 3's "r_k" labels are cutoff mode indices (largest m above the floor), not the eq. 4.8 count over signed modes.
    - Fig. 4 prints a Gabor uncertainty product of 1.38, which contradicts Table 5.1's claim that the Gabor saturates the bound (see 6).
    - Eq. 6.6′'s 1/T noise factor is wrong by a factor T in the sum-of-squares convention of eq. 6.6. Fig. 5's exact check gives 1e-11 agreement without the 1/T and a 99% error with it.
    - Fig. 5's network-floor panel plots one-sided sums over m > N/2. Eq. 6.9 sums over signed modes, which is twice as large.
11. **App. G.7, noisy optimum (new).** With noise, at fixed ν each active mode gets *less* energy than without noise, because the noise term lowers the marginal return (eq. G.14). The thesis claims the opposite ("the cubic root … is larger"). The budget then forces ν\* **down**, not up, and the active set does not shrink: at σ² = 1 (N = 128, λ = 1, K0 = 500), ν\* falls from 3.2e-4 to 1.9e-4 and M\* rises from 29 to 30. The claim of "fewer active modes, a narrower band and a broader kernel" is reversed.
    - Each mode's noisy error has a finite minimum at a = λS/φ. Once Σ λS/φ < K0, the equality budget makes the problem non-convex (ν < 0). The optimum is then to leave budget unspent; `water_fill_noisy` treats K0 as an upper bound in that regime.
    - G.5's statement that modes with SNR ≲ 1 are excluded is not implied by G.13: the slope at a = 0 is −2S/λ whatever the noise, so the threshold stays S > νλ/2.
    - The G.9 estimator recovers σ² per mode in the sum-over-T convention, not σ²/T (same issue as eq. 6.6′).
12. **Travelling steady state, sign convention.** With a forward DFT using e^{−inθ}, an input I(θ − ωt) gives r̂_n = Î_n e^{−inωt}/(γ_n − inωτ). Spec 6.4's "+inωτ" is the same statement in the opposite n convention. Verified by simulation: the opposite sign is wrong by more than 10×.

## Figure reproductions (`seqnet/thesis/figures/figN.ipynb`)

Parameters [D11] were not in the thesis text. Where a figure prints numbers, I inferred the parameters from them and then checked the inference against other printed numbers. Calibrations are disclosed in each config and notebook. Acceptance tables are in `results/thesis/data/figN/acceptance.json` (`python -m seqnet.thesis summary`).

| Fig. | Spec acceptance | Result | Notes |
| --- | --- | --- | --- |
| 2 | no single best kernel; stars match | **pass** (5/5 stars, 4 distinct winners) | Kernels compared at equal energy K0; with unit-integral kernels the point kernel wins every class. λ = 30 (D10 units) is calibrated by an added λ-sweep panel F, so the star match is a one-parameter calibration. Ranks: Gaussians and point match exactly (5, 17, 128); box 43 vs 37, Gabor 33 vs 39 |
| 3 | per-mode rates within 5% of eq. 3.19 | **pass** (6e-13, mean-field) | Cutoffs 128/7/22/33/23 vs printed 128/7/23/34/24 after choosing widths; box R = 1.2 as in Fig. 2 |
| 4 | aliasing within 5% of eq. 6.8; σ_xσ_ξ ≥ ½, Gaussian = ½ | aliasing **fail** (off by 30–127%); uncertainty **pass** | Aliasing failure is discrepancy 3. Printed products: point 2.45 (here a narrow Gaussian gives 0.50, so the thesis's "point" is another object), box 2.00 (here 4.7 at N = 512; box σ_ξ grows with N), Gabor 1.38 (here 1.67). r_eff matches for d ≥ 3 (10, 6, 4, 2) |
| 5 | per-mode errors within 5% of eq. 6.6/6.6′ | **pass** (1e-11, no 1/T); 6.6′ as printed **fail** | Parameters inferred: S = e^{−m²/50}, N = 128, T = 32, equal-energy kernels, λ = 10⁻³a₀. Confirmed by the printed panel-F sums (0.75–1.30× of 1.46e-5 / 1.66e-4 / 2.79) and panel-C floors (2.30 / 0.555 / 0.0059 vs 2.4 / 0.6 / 0.006; polynomial 0.024 vs 0.025) |
| 6 | optimum ≤ every family at equal budget | **pass** (E 0.003 vs 0.89) | RMSE ordering as printed. eq. 7.17 = E_bias at the optimum to 1e-15 |
| 7 | T\* = 2M\*; error decreasing in B0 | **pass** | λ = 5×10⁻⁴ calibrated; with it, eq. 8.9 reproduces all four printed M\* within ±1 (6/32/63/89 vs 5/32/64/89) |
| 8 | allocation within 5% of eq. 7.13; active set exact; readout at equilibrium | allocation **pass** (3e-14), active set **pass**; readout **fail** in 17/3000 epochs | Literal eq. 9.10 (no preconditioner). Readout violations are all before epoch 21, at active-set changes (20 sweeps per kernel epoch). Target identified: narrowband ξ0 = 10.5, ℓ = 2.5, λ = 0.1, K0 = 10 (unitary) |
| 9 | error falls with N, K0, rises with λ; T\* reported | **pass** (all four, incl. T\* grid 36/36) | Narrowband ξ0 = 24, ℓ = 3 reproduces **all 36 printed T\* values exactly** (T\* = 2M\* at T_obs = N, λ and K0 in unitary units × N). That fixes Fig. 9's D11 parameters and independently confirms the D10 unit reading. Fixed-point errors match the figure's dotted lines (e.g. K0 = 1, 2, 5: 0.48, 0.30, 0.125 vs ≈ 0.47, 0.3, 0.125; λ = 0.03, 0.01: 0.0076, 0.001). Learning curves use the literal rule at η = 1.9·λ²/(3S_max). They reach the fixed point for λ ≥ 0.3 and small K0, but at λ ≤ 0.1 or large K0 (large a/λ) they are still falling after 2000 epochs, more slowly than in the thesis figure. That is discrepancy 4; the thesis's step sizes are unknown (D11) |

**D11 status.** The parameters of Figs. 5, 7, 8 and 9 are now identified from printed values. Figs. 2, 3 and 4 still use documented defaults plus one calibration (Fig. 2's λ).

## [VERIFY] switches and defaults used

- `error_terms(sampling_scaling='thesis')` is the default; `'exact'` and `sampled_error_exact` are the corrected versions.
- `DeltaRule(decay_is_ridge=True)` is the default (λ/T internally); `False` is eq. 3.18 literally.
- Slow-rule `precondition='none'` is the default (thesis); `'newton'` is used in TT12/TT13.
- Slow-rule budget: `dual` is the default; `project` and `fixed` are available.
- `Kernel.effective_rank(reference='dc')` follows eq. 4.8; `'max'` is needed for the Gabor.
- Polynomial PSD: S(0) := S0 (undefined in the thesis).
- The Nyquist mode of even N is excluded from learning tests (its power is ill-defined on the ring).

## Deviations from the spec

- T10 stochastic tolerance is 2e-2 instead of 1e-3; the expected dynamics meet 1e-6.
- TT12/TT13 acceptance is met by the Newton-preconditioned rule, not the literal eq. 9.10 (discrepancy 4).
- Python 3.12 venv via `uv` (spec: ≥ 3.11). The repository is now under git, but figure metadata (`<panel>.json`) does not yet record the commit hash; spec 5.3's `meta.json` belongs to the main-package run directories.
- Figure outputs are panel-per-file (`results/thesis/figures/figN/<panel>.png`), following Max's request for panel-by-panel notebooks. The spec's "one script per figure" became one notebook per figure.
- `seqnet.thesis.figures.load_config` imports PyYAML lazily (optional); spec 6A.2 rule 1 lists only stdlib, NumPy, SciPy and matplotlib.
- Fig. 9 learning uses the exact ridge readout in the inner loop for speed; Fig. 8 covers the delta-rule fast loop.

## Next

Main-package `conventions.py`, `targets.py`, `braun_map.py` with T1–T8 (spec §16.1 step 1), then WP2.1/WP2.3.
