"""seqnet.thesis -- standalone rebuild of the Tepper (2026) MSc thesis framework (spec Section 6A).

Imports only the standard library, NumPy, SciPy and (for figures) matplotlib; never another ``seqnet``
module (test TT17).  Uses thesis notation and equation numbers throughout.  The operational kernel power
is a_m = |K_m|^2 = N |k_hat_m|^2 (D10 resolution; see :mod:`seqnet.thesis.fourier`).
"""
from .coherence import (
    autocorrelation,
    coherence,
    independence_distance,
    pairwise_coherence,
    spatial_width,
    spectral_energy_distribution,
    spectral_spread,
    uncertainty_product,
)
from .design import DesignSVD, activity, design_matrix, design_svd, observation_subspace, sampled_effective_rank
from .errors import ErrorTerms, error_terms, expected_error, measure_error, per_mode_error, sampled_error_exact
from .field import (
    branching_ratios,
    continuum_coeffs,
    effective_damping,
    inverse_som_chain,
    simulate_field,
    som_chain,
    steady_state,
)
from .fourier import (
    alloc_from_kernel,
    circulant_eigs,
    full_from_half,
    kernel_from_half_spectrum,
    mode_multiplicity,
    unitary_dft,
    unitary_idft,
)
from .kernels import FAMILIES, Kernel, analytic_spectrum_ratio
from .learning import (
    EMAEstimator,
    History,
    LearningSystem,
    RidgeObjectiveRule,
    ThesisKernelRule,
    analytic_gradient,
    observable_gradient,
    spatial_error_power,
    temporal_error_power,
)
from .optimum import (
    JointResult,
    WaterFillResult,
    appendix_I_cases,
    implied_psd,
    joint_optimum,
    marginal_returns,
    min_bias_error,
    noise_floor_estimate,
    noisy_alloc,
    optimality_gap,
    ridge_fill,
    ridge_marginal_returns,
    snr,
    water_fill,
    water_fill_noisy,
)
from .perturbation import (
    amplitude_modulated,
    bridge_energy,
    coupling_error,
    coupling_matrix,
    expected_ridge_error,
    position_averaged_kernel,
    position_dependent_design,
)
from .readout import (
    DeltaRule,
    RLSReadout,
    delta_rule_rates,
    per_mode_ridge,
    pinv_readout,
    ridge_readout,
    spectral_filter,
)
from .ring import Ring, Schedule
from .targets import TargetClass, fixed_target

__all__ = [
    # ring and kernels
    "Ring", "Schedule", "Kernel", "FAMILIES", "analytic_spectrum_ratio",
    # Fourier / normalization
    "unitary_dft", "unitary_idft", "circulant_eigs", "alloc_from_kernel", "mode_multiplicity",
    "full_from_half", "kernel_from_half_spectrum",
    # design
    "activity", "design_matrix", "design_svd", "DesignSVD", "observation_subspace", "sampled_effective_rank",
    # targets
    "TargetClass", "fixed_target",
    # readout
    "pinv_readout", "ridge_readout", "per_mode_ridge", "spectral_filter", "delta_rule_rates", "DeltaRule",
    "RLSReadout",
    # coherence
    "autocorrelation", "coherence", "pairwise_coherence", "spectral_energy_distribution", "spectral_spread",
    "spatial_width", "uncertainty_product", "independence_distance",
    # errors
    "ErrorTerms", "error_terms", "per_mode_error", "measure_error", "sampled_error_exact", "expected_error",
    # perturbation (App. E)
    "position_dependent_design", "amplitude_modulated", "position_averaged_kernel", "coupling_matrix",
    "bridge_energy", "expected_ridge_error", "coupling_error",
    # optimum
    "WaterFillResult", "water_fill", "ridge_fill", "min_bias_error", "JointResult", "joint_optimum",
    "marginal_returns", "ridge_marginal_returns", "optimality_gap", "implied_psd", "appendix_I_cases",
    "water_fill_noisy", "noisy_alloc", "snr", "noise_floor_estimate",
    # learning
    "ThesisKernelRule", "RidgeObjectiveRule", "spatial_error_power", "temporal_error_power", "EMAEstimator",
    "History", "LearningSystem", "observable_gradient", "analytic_gradient",
    # field
    "continuum_coeffs", "effective_damping", "som_chain", "inverse_som_chain", "steady_state", "simulate_field",
    "branching_ratios",
]
