# WP-T report: `seqnet.thesis` (in progress)

Status as of 2026-10-13. Spec: `Design spec from readout function to sequence connectivity.md` §6A.

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

**Not yet built:** the CLI (`__main__.py`), configs, the Fig. 2–9 scripts, the README, and App. G's SNR and noise-floor estimator. TT13 and the RLS readout are present; App. L is only a stub marked [VERIFY].

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
4. **Kernel rule stiffness (new).** Eq. 9.10 with a single η is stiff. Its curvature is 6S/λ² at a = 0 and 3ν/(a + λ) at the optimum; the ratio is about 10⁷ for N = 32, λ = 0.1.
   - Any stable η takes more than 10⁶ steps (test `test_plain_thesis_rule_is_stiff`).
   - A practical η collapses the allocation onto a few modes.
   - Fix (not in the thesis, off by default): precondition each mode's step by the inverse of the observable Hessian, (a_m + λ)/(3G_m), where G_m is the rule's own gradient term. This keeps the rule S-free and its fixed point, and converges to machine precision. The same holds for the square-root rule with factor 2. This affects the claim in §9.2/9.5 that the rule is "convergent in the LMS sense".
5. **σ_x definition (new, minor).** Under eq. 5.10, a Gaussian k ∝ e^{−r²/2σ²} has σ_x = σ/√2. The text and Table 5.1 take σ_x = σ, and σ_ξ² = 1/(2σ)² in §5.4, whereas eq. 5.8 gives 1/(2σ²). The two errors cancel in the product.
6. **Uncertainty bound (new).** The Gabor does **not** saturate σ_xσ_ξ ≥ 1/2: under eq. 5.8, σ_ξ² includes ξ0². This contradicts Table 5.1 and §8 ("Gaussian and Gabor saturate"). On the periodic ring the bound fails for broad kernels (σ = 3 rad gives a product below 1/2), so Theorem 5.1 needs a localization condition.
7. **D1, √3 direction (confirmed numerically: narrower).** For S ∝ e^{−βm²}, the kernel's autocorrelation half-width over the target's is **0.577 = 1/√3** at β = 0.002 and β = 0.01. The kernel is narrower, not broader as App. I.4/I.7 state. Eq. I.19 (σ_k² = 3/ℓ²) is inconsistent with I.18; the spectral width √3ℓ implies an envelope σ_k = 1/(√3ℓ). Not encoded in tests pending Max's confirmation.
8. **App. I.4 cutoff (new).** The active band is \|m\| < √((1/β) ln(2S0/(νλ))). The text prints (1/β) ln(…) without the square root (test `test_TT9_gaussian_psd_active_cutoff_has_square_root`).
9. **D2/D6, joint optimum.** With κ = λ/2, the closed form (8.9) gives T\* = 58, 70, 80 for B0 = 5, 20, 80 (Gaussian PSD, β = 0.01, λ = 1e-2, N = 128). Direct numeric minimization of E\*(T) over even T gives T\* = 54, 66, 78. T\* = 2M\* does hold at the numeric optimum, but the closed form overshoots by 2–4 samples. Eq. 8.9 has no solution in general, because M\*(ν) is a step function; the residual is bounded by λ.
10. **Travelling steady state, sign convention.** With a forward DFT using e^{−inθ}, an input I(θ − ωt) gives r̂_n = Î_n e^{−inωt}/(γ_n − inωτ). Spec 6.4's "+inωτ" is the same statement in the opposite n convention. Verified by simulation: the opposite sign is wrong by more than 10×.

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
- Python 3.12 venv via `uv` (spec: ≥ 3.11). The project is not a git repository yet, so `meta.json` cannot record a commit hash.

## Next

CLI and configs, then the Fig. 2–9 reproductions (Figs. 6, 8, 9 quantitative), then the README. After that: main-package `conventions.py`, `targets.py`, `braun_map.py` with T1–T8 (spec §16.1 step 1), then WP2.1/WP2.3.
