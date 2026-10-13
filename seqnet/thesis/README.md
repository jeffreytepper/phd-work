# seqnet.thesis

A standalone rebuild of the computational framework of Tepper (2026), *Connecting Manifold Rank to Computational
Capacity through Structured Kernel Inhibition* (MSc thesis, Göttingen), from the thesis text alone (design spec §6A).

The subpackage imports only the standard library, NumPy and SciPy (matplotlib for figures) and never another `seqnet`
module (test TT17). It runs from a bare copy of `seqnet/thesis/` (test TT18). It uses thesis notation, and every
formula function cites its equation number.

## Conventions you must know

- **Operational kernel power (D10).** `a_m = |K_m|^2 = N |k̂_m|^2`, the squared singular value of the design matrix at
  `T_obs = N`. Here `K_m` is the unnormalized DFT and `k̂_m` is the unitary eq. 4.1 coefficient. Budgets count
  signed modes: `K0 = Σ_m a_m = N Σ_i k_i²`.
- **Units of the thesis figures.** The thesis quotes λ, K0 and B0 in unitary units (`a = |k̂|²`). Multiply λ, K0
  and B0 by N to get the D10 values used by this code; the error formulas are invariant under this joint rescaling.
  Fig. 9's printed T\* grid is reproduced exactly this way.
- **Half spectra.** Allocations and PSDs are stored on `m = 0..⌊N/2⌋`; `mode_multiplicity(N)` gives the signed-mode
  weights.
- **Delta-rule decay.** `DeltaRule(decay=λ)` takes the *ridge* λ and uses λ/T internally, so that its fixed point is
  eq. 3.13. Eq. 3.18 taken literally converges to ridge with T·λ (REPORT discrepancy 2).

## Minimal example

```python
import numpy as np
import seqnet.thesis as th

N = 128
S = th.TargetClass.narrowband(N // 2, xi0=24, ell=3).psd      # target class by its PSD (eq. 6.3, I.17)
res = th.water_fill(S, decay=0.1 * N, K0=10 * N, N=N)         # eq. 7.11-7.13 (thesis units x N)
print(res.M_star, res.rank, 2 * res.M_star)                   # 33, 67, T* = 66

kernel = th.Kernel.from_alloc(res.a, N)                       # Wiener-optimal kernel, positive signs
X = th.design_matrix(kernel, th.Schedule(N, 1))               # X_{t,i} = k(r_i - c_t), eq. 2.4
y = th.TargetClass(psd=S).sample(np.random.default_rng(0), N, th.Schedule(N, 1).positions)
w = th.ridge_readout(X, y, 0.1 * N)                           # eq. 3.13
print(th.error_terms(res.a, S, 0.1 * N, N, N))                # eq. 6.21-6.22
```

## Public API (`__all__`)

| Area | Names | Thesis source |
| --- | --- | --- |
| Ring, schedule | `Ring`, `Schedule` | §2.1, §2.3 |
| Kernels | `Kernel` (`from_family`, `from_alloc`, `alloc`, `energy`, `effective_rank`, `autocorrelation`), `FAMILIES`, `analytic_spectrum_ratio` | §4.5, App. D |
| Fourier | `unitary_dft`, `unitary_idft`, `circulant_eigs`, `alloc_from_kernel`, `mode_multiplicity`, `full_from_half`, `kernel_from_half_spectrum` | eq. 4.1, D10 |
| Design | `activity`, `design_matrix`, `design_svd`, `observation_subspace`, `sampled_effective_rank` | eq. 2.1–2.6, 5.4 |
| Targets | `TargetClass` (six App. I classes, `sample`, `coeffs`, `evaluate`), `fixed_target` | eq. 6.3, App. I |
| Readout | `pinv_readout`, `ridge_readout`, `per_mode_ridge`, `spectral_filter`, `delta_rule_rates`, `DeltaRule`, `RLSReadout` | §3, §4.3 |
| Coherence | `autocorrelation`, `coherence`, `pairwise_coherence`, `spectral_energy_distribution`, `spectral_spread`, `spatial_width`, `uncertainty_product`, `independence_distance` | §5, App. C–D |
| Errors | `error_terms` (thesis), `sampled_error_exact` (corrected), `expected_error` (exact expectation), `measure_error` (Monte Carlo), `per_mode_error` | §6, App. G |
| Optimum | `water_fill`, `ridge_fill`, `min_bias_error`, `joint_optimum`, `marginal_returns`, `ridge_marginal_returns`, `optimality_gap`, `implied_psd`, `appendix_I_cases` | §7, §8, App. I |
| Learning | `ThesisKernelRule`, `RidgeObjectiveRule`, `LearningSystem`, `History`, `spatial_error_power`, `temporal_error_power`, `EMAEstimator`, `observable_gradient`, `analytic_gradient` | §9, App. K.1; spec §10.1 |
| Field | `simulate_field`, `som_chain`, `inverse_som_chain`, `steady_state`, `effective_damping`, `continuum_coeffs`, `branching_ratios` | App. K.2, App. L |
| Perturbation | `position_dependent_design`, `amplitude_modulated`, `position_averaged_kernel`, `coupling_matrix`, `bridge_energy`, `expected_ridge_error`, `coupling_error` | App. E |

Slow-loop rules have two options beyond the thesis, both off by default:

- `precondition='newton'` steps by the inverse of the observable Hessian. Use it when a/λ is large, where the literal
  rule is stiff.
- `budget='project'` keeps Σ a = K0 at every step. The default `'dual'` updates ν instead.

## Command line

```bash
python -m seqnet.thesis fig8                     # execute a figure notebook in place, print its acceptance table
python -m seqnet.thesis summary                  # acceptance tables of all figures
python -m seqnet.thesis water_fill --config configs/thesis/example_water_fill.json
python -m seqnet.thesis learn --config configs/thesis/example_learn.json
```

## Figures

`figures/figN.ipynb` reproduces thesis Fig. N panel by panel; there is one notebook per figure, and the parameters
are in `configs/thesis/figN.yaml`. Each panel has a compute cell that writes `results/thesis/data/figN/<panel>.npz`
(plus a JSON with the parameters), and a plot cell that reads only that file and writes
`results/thesis/figures/figN/<panel>.png`. Each notebook ends with an acceptance table in
`results/thesis/data/figN/acceptance.json`. Running notebooks needs `nbconvert` and `ipykernel`; the
`notebooks` extra in `pyproject.toml` installs them.

## Tests

```bash
python -m pytest tests/thesis
```

These are TT1–TT18 of the spec plus regression tests for the documented discrepancies. They take about one minute.
