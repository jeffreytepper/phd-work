"""TT6, TT7: spectral coherence (5.6) vs pairwise overlap (5.2); the uncertainty bound (Theorem 5.1)."""
import numpy as np
import pytest

import seqnet.thesis as th

N = 128


@pytest.mark.parametrize(
    "name,params",
    [("gaussian", {"sigma": 0.3}), ("box", {"R": 0.7}), ("bspline", {"p": 3, "h": 0.2}),
     ("gabor", {"sigma": 0.3, "xi0": 6.0}), ("point", {})],
)
def test_TT6_spectral_coherence_equals_pairwise(name, params):
    k = th.Kernel.from_family(name, N, **params)
    d = np.linspace(0, 1.5, 37)  # includes off-grid displacements
    np.testing.assert_allclose(th.coherence(k, d), th.pairwise_coherence(k, d), atol=1e-10)
    # on the grid it is also the discrete autocorrelation of eq. 5.1
    R = k.autocorrelation()
    np.testing.assert_allclose(th.autocorrelation(k, 2 * np.pi * np.arange(N) / N), R, atol=1e-10 * R[0])


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3, 0.5])
def test_TT7_gaussian_saturates_uncertainty(sigma):
    """Gaussian: sigma_x sigma_xi = 1/2 (to discretization/periodization error) with sigma_x = sigma/sqrt(2)."""
    k = th.Kernel.from_family("gaussian", 512, sigma=sigma)
    assert abs(th.uncertainty_product(k) - 0.5) < 1e-3
    assert abs(th.spatial_width(k) - sigma / np.sqrt(2)) < 1e-3 * sigma


@pytest.mark.parametrize(
    "name,params",
    [("box", {"R": 0.7}), ("bspline", {"p": 1, "h": 0.3}), ("bspline", {"p": 3, "h": 0.2}),
     ("gabor", {"sigma": 0.3, "xi0": 6.0}), ("gaussian", {"sigma": 0.25})],
)
def test_TT7_bound_holds_for_localized_families(name, params):
    """sigma_x sigma_xi >= 1/2 for kernels localized well inside the ring.

    The box's sigma_xi is dominated by its sinc sidelobes and grows with N (discrete ring), so its product
    exceeds 1/2 by a wide margin; the Gabor does NOT saturate the bound (contrary to Table 5.1)."""
    k = th.Kernel.from_family(name, 512, **params)
    p = th.uncertainty_product(k)
    assert p >= 0.5 - 1e-6
    if name == "gabor":
        assert p > 0.5 + 0.1


def test_TT7_bound_fails_for_broad_kernels_on_ring():
    """Documented limitation: on the periodic ring the bound fails for kernels comparable to the ring
    (e.g. a nearly constant kernel has sigma_xi -> 0 while sigma_x stays ~ pi/sqrt(3))."""
    k = th.Kernel.from_family("gaussian", 256, sigma=3.0)
    assert th.uncertainty_product(k) < 0.5


def test_independence_distance_gaussian_closed_form():
    """d*_Gauss(eps) = 2 sigma sqrt(ln 1/eps) (Thesis eq. 5.12, App. C.3, D.4)."""
    sigma = 0.25
    k = th.Kernel.from_family("gaussian", 512, sigma=sigma)
    for eps in (0.5, 0.1, 0.01):
        assert abs(th.independence_distance(k, eps) - 2 * sigma * np.sqrt(np.log(1 / eps))) < 2e-3


def test_independence_distance_box_closed_form():
    """d*_box(eps) = R (1 - eps) (Thesis eq. D.4), up to the grid discretization of the box."""
    R = 0.8
    k = th.Kernel.from_family("box", 1024, R=R)
    for eps in (0.5, 0.2):
        assert abs(th.independence_distance(k, eps) - R * (1 - eps)) < 0.02
