# D2Q9 lattice constants: weights, opposites, isotropy
import numpy as np
from src.lbm import lattice as lt

# test 1: array shapes match the direction count
def test_direction_count():

    assert lt.direction_count == 9
    assert lt.lattice_velocities.shape == (9, 2)
    assert lt.lattice_weights.shape == (9,)
    assert lt.opposite_direction.shape == (9,)

# test 2: weights sum to one (correct macroscopic density)
def test_weights_sum_to_one():

    assert np.isclose(lt.lattice_weights.sum(), 1.0)

# test 3: weights positive
def test_weights_positive():

    assert np.all(lt.lattice_weights > 0)

# test 4: rest direction is zero
def test_rest_direction_is_zero():

    assert np.array_equal(lt.lattice_velocities[0], [0, 0])

# test 5: opposite direction reverses the velocity (bounce-back correctness)
def test_opposite_reverses_direction():

    for q in range(lt.direction_count):
        assert np.array_equal(lt.lattice_velocities[lt.opposite_direction[q]], -lt.lattice_velocities[q])

# test 6: taking the opposite twice returns the original direction
def test_opposite_is_involution():

    for q in range(lt.direction_count):
        assert lt.opposite_direction[lt.opposite_direction[q]] == q

# test 7: lattice sound speed squared is 1/3
def test_sound_speed_squared():

    assert np.isclose(lt.sound_speed_squared, 1.0 / 3.0)

# test 8: sum_q w_q c_q = 0 (zero mean momentum at rest)
def test_first_order_moment_vanishes():

    weighted_velocity_sum = (lt.lattice_weights[:, None] * lt.lattice_velocities).sum(axis=0)

    assert np.allclose(weighted_velocity_sum, [0.0, 0.0])

# test 9: sum_q w_q c_qa c_qb = c_s^2 delta_ab (correct pressure and viscosity)
def test_second_order_moment_is_isotropic():

    second_moment = np.einsum("q,qa,qb->ab", lt.lattice_weights, lt.lattice_velocities, lt.lattice_velocities)

    assert np.allclose(second_moment, lt.sound_speed_squared * np.eye(2))
