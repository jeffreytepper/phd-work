# Checkpoint 1 — 2026-10-14

> **2026-10-15:** the WP-T thesis framework (`seqnet.thesis`, `configs/thesis/`, `tests/thesis/`, `results/thesis/`)
> has moved to its own repository, **msc-thesis** (package `msc_thesis`). Paths below refer to the old layout.

A review of what exists, what works, what should change, and how to proceed. Details and numbers are in
`results/thesis/REPORT.md` (WP-T) and `results/wp0/REPORT.md` (WP0). The spec is
`Design spec from readout function to sequence connectivity.md`.

## 1. What has been built

| Part | State | Where |
| --- | --- | --- |
| WP-T: thesis framework (`seqnet.thesis`) | Complete except App. L (stub). Standalone, CLI, README | `seqnet/thesis/`, 77 tests in `tests/thesis/` |
| Thesis Figs. 2–9 | One notebook per figure, panel by panel; 20 checks pass, 3 fail (all thesis discrepancies) | `seqnet/thesis/figures/`, `results/thesis/` |
| Main-package core (spec §6.1–6.3, 6.7, 6.8, 6.10, 6.13–6.17) | Done | `seqnet/` |
| Data layer: bat (proxy) and MEC (skeleton) | Bat working; MEC behind one loader stub | `seqnet/sequences/datasets/`, `configs/wp0/data.yaml` |
| Core tests T1–T9, T15, T17, data tests | 16 pass | `tests/core/` |
| WP0 (Braun reproduction) | Every synthetic and bat panel; 14 notebooks; 34 checks pass, 8 fail (findings or settings) | `notebooks/wp0/`, `results/wp0/` |
| Sequences (`seqnet.sequences`) | All targets behind one interface: Braun waves, the Lehr et al. 2024 family (parametric and ring network), datasets; one 8.5-s training clock; demo notebook, 10 checks pass | `seqnet/sequences/`, `notebooks/sequences/`, `results/sequences/` |
| Run infrastructure | Cached parallel runs with `meta.json` (config, commit, versions, seeds, time) | `seqnet/experiments/` |

Not built yet: `seqnet/bridge.py`; `models/driven_ring.py`, `gain_ring.py`, `ei_ring.py`; `training/kernel_rule.py`;
the `spectral/thesis_opt.py` and `spectral/kernels.py` wrappers; WP1–WP6.

## 2. What works as expected

- **Thesis closed forms.** Water-filling, App. I, KKT, joint optimum and the error terms are exact against simulation.
  Fig. 9's 36 printed T\* values are reproduced exactly once its target is identified. Fig. 7's M\* values match within
  ±1 with one calibrated λ.
- **Braun Fig. 1 at the paper's settings.** Matrix correlation 0.063 (paper 0.07), profile correlation 0.971
  (0.97–0.98). The mean matrix regenerates the sequence.
- **Parameter trends.** All Fig. 2 / Supp. 3–5 trends reproduce, as do sparsity variants (b) and (c) and the
  perturbation phenomenology (reset, restart, current-magnitude dependence).
- **Bat data.** Training (0.94 vs 0.90) and the data-vs-synthetic profile agreement (0.90 vs 0.83) reproduce, and the
  column banding appears only in the data-trained network.
- **Suppl. Cor. 3.2 on learned networks (WP0.8).** Holds at k = 1–2 (median error 1.6–3%).
- **Stability.** Every trained network has a strongly attracting orbit (transverse multipliers < 0.01).

## 3. What did not work, or needs changing

### 3a. Source discrepancies (reported, not fixed)

| # | Source | Issue | Consequence for the plan |
| --- | --- | --- | --- |
| 1 | Thesis D10 | Singular values are √N\|k̂\|; operational a_m = N\|k̂\|² | Spec §3.4 fact 6 needs the factor N; thesis figure units = D10 / N |
| 2 | Thesis eq. 3.18 | Delta rule with decay λ converges to ridge(T·λ) | WP3 fast loop: use λ/T (done in code) |
| 3 | Thesis eq. 6.7–6.9 | Missing alias-contamination term and (T/N) scaling | `sampled_error_exact` replaces them where T_obs < N |
| 4 | Thesis eq. 9.10 | Stiff when a/λ is large: ratio ((a+λ)/λ)⁴ | WP3: decide whether the Newton preconditioner is in or out (decision D-B) |
| 5 | Thesis App. I, D1 | Kernel is *narrower* than the target by √3; I.19 inconsistent | Needs Max's confirmation |
| 6 | Thesis App. G.7 | Noise lowers ν\*; active set does not shrink | Preprint correction |
| 7 | Thesis §5 | Gabor does not saturate the uncertainty bound; σ_x definition | Preprint correction |
| 8 | Braun eq. 8 (A2) | Printed FORCE step ≠ RLS; only the standard step equals ridge centred on J0 | **WP1 rationale (fact 4) holds only for the standard step** |
| 9 | Braun Supp. 8 claim | Reset distance follows the profile *zero crossing*, not its minimum | WP4.2 input; a possible result |
| 10 | Braun A1 | Training toward x\* runs 1–4% slow; y\* gives the exact period | Choose the convention for later WPs (decision D-A) |
| 11 | Braun noise | Self-connections ≈ −1.3 are noise correction; they set the learned eigenvalues above the active modes | WP0.8/WP1/WP2.2 comparisons must handle the diagonal |
| 12 | Braun eq. 22 | Spec 6.3's noise perturbation is 2× the Methods | Code follows the Methods |
| 13 | Spec 6.14 / T15 | Off-diagonal motifs of a circulant are −1/(N−1), not 0 | Spec corrected in the tests |

### 3b. Reproduction gaps (likely settings or data, not code)

- **Fig. 5 downstream:** 0.93 vs ≥ 0.99 with a noisy upstream; 0.99 noise-free. The Methods are ambiguous.
- **Supp. 2 mask (a):** degrades beyond 75% sparsity (paper: 95%). Dev settings (100 training iterations).
- **τ_rec range:** 0.01–0.47 vs 0.55–0.8. Possibly the diagonal or dev settings; unexamined.
- **Bat bump width:** 1.19 s vs 1.72 s, a proxy-data limitation (166 cells, correct trials only).
- **Thesis Figs. 2–4:** parameters not recoverable from the figures. Calibrated or defaulted, and disclosed.

### 3c. Engineering items to change

- **Notebook builders.** The notebooks were generated by builder scripts in a temporary scratchpad. The notebooks
  are now the source of truth; the builders will be lost. Either archive them in `tools/` or edit notebooks directly
  from now on (decision D-F).
- **Repository size.** `results/wp0/runs/` is 1.5 GB (gitignored, regenerable). Tracked figure data is 35 MB and
  growing. Consider ignoring `*.npz` panel data above ~1 MB, or Git LFS.
- **Compute.** FORCE at N = 501 is memory-bound on this 6-core machine: about 50 min per condition at final settings
  with 6 workers. Final reruns of all 60+ WP0 sweep conditions would take ≈ 2 days. Options: a cluster, a float32
  backend (spec allows it with validation), fewer realisations (e.g. 20) for sweeps, or N = 251 for exploratory WPs
  (decision D-C).
- **Cache-hash pitfall.** New `ForceConfig` fields must be registered in `runner._LATE_DEFAULTS`. A test should
  guard this.
- **Test runtime** is about 2 min in total, within the spec's 5-minute budget.

## 4. Decisions needed

| ID | Decision | Recommendation |
| --- | --- | --- |
| D-A | Training convention for WP1–WP4 (`methods` = Braun as printed, or `supplement`) | Run WP1 under `methods` (comparability with Braun); use `supplement` for WP2.2/WP4.1, where the analytic map assumes y\* |
| D-B | Kernel-rule preconditioner in WP3 | Report both. The literal rule is the thesis claim; the Newton variant shows the fixed point is reachable |
| D-C | Compute strategy | Dev settings for exploration; final settings only for results that go into an abstract or paper. A cluster account would help |
| D-D | WP1 RLS variant | Run both the printed and the standard step; fact 4 is only exact for the standard one |
| D-E | Spec decisions still open: D1 (√3 direction), D3 (objective), D4 (DC in budget), D11 (thesis figure parameters) | Max to confirm D1 and send D11 if available; defaults otherwise |
| D-F | Notebook builders | Edit notebooks directly; archive the builders once in `tools/nb_builders/` for provenance |

## 5. How to proceed

COSYNE deadline: **2026-10-18 (4 days)**. Remaining items of the §16.1 slice:

| Step | Work | Compute | Gives |
| --- | --- | --- | --- |
| 1 | `bridge.py` (thesis ↔ Braun units, kernel ↔ temporal profile) + WP2.1 atlas + WP2.3 implied spectra for the bat fit and Braun's MEC/bat fits | minutes (closed form) | Claim A: normative profiles predicted from readout spectra; rank = 2M\* + 1; odd part = ωτ ∂ even |
| 2 | WP1 at dev settings: baseline vs λ_J ∈ {1e-4, 1e-3, 3e-3} on the Fig. 1 target, standard and printed RLS | ≈ 1.5 h | Claim B: does decay collapse the degeneracy? |
| 3 | WP3a: thesis vs ridge-objective rule, exponent fit (1/3 vs 1/2) | minutes | Claim C: the slow-loop objective sets the water-filling exponent |
| 4 | Write-up: one-page results summary per claim, with figures | — | Abstract draft |

Results from this checkpoint that could also go into an abstract, if Max wants them:
- Braun's common profile is fixed only on the strongly represented modes.
- The learned diagonal is a noise-correction term.
- Reset location follows the profile's zero crossing.

After COSYNE, in spec order: WP0 final reruns where a result depends on them → WP2.2 training check → WP3b–d → WP4 →
WP5.2–5.3 → MEC-dependent items (WP0.7 mouse, WP5.1, WP5.4, WP6) once the data arrive.
