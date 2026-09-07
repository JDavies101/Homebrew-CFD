# D2Q9 lattice constants: weights, opposites, isotropy
import numpy as np

from src.lbm import lattice


def test_direction_count():
    assert lattice.Q == 9
    assert lattice.E.shape == (9, 2)
    assert lattice.W.shape == (9,)
    assert lattice.OPP.shape == (9,)


def test_weights_sum_to_one():
    # Required for correct macroscopic density; the core normalization.
    assert np.isclose(lattice.W.sum(), 1.0)


def test_weights_positive():
    assert np.all(lattice.W > 0)


def test_rest_direction_is_zero():
    assert np.array_equal(lattice.E[0], [0, 0])


def test_opposites_reverse_direction():
    # Bounce-back correctness: E[OPP[i]] must equal -E[i] for every i.
    for i in range(lattice.Q):
        assert np.array_equal(lattice.E[lattice.OPP[i]], -lattice.E[i])


def test_opposite_is_involution():
    # Taking the opposite twice returns the original direction.
    for i in range(lattice.Q):
        assert lattice.OPP[lattice.OPP[i]] == i


def test_lattice_sound_speed():
    assert np.isclose(lattice.CS2, 1.0 / 3.0)


def test_first_order_velocity_moment_vanishes():
    # sum_i w_i * e_i = 0  (lattice isotropy; zero mean momentum at rest).
    weighted = (lattice.W[:, None] * lattice.E).sum(axis=0)
    assert np.allclose(weighted, [0.0, 0.0])


def test_second_order_velocity_moment_is_isotropic():
    # sum_i w_i * e_ia * e_ib = CS2 * delta_ab  (recovers correct pressure/viscosity).
    tensor = np.einsum("i,ia,ib->ab", lattice.W, lattice.E, lattice.E)
    assert np.allclose(tensor, lattice.CS2 * np.eye(2))
