# BGK collision: conservation, tau independence, equilibrium fixed point
import numpy as np
from src.lbm.collision import collide
from src.lbm.moments import macroscopic
from src.lbm.equilibrium import equilibrium

rng = np.random.default_rng(0)
# test 1: mass and momentum conservation
def test_conservation():

    f = rng.uniform(0.5, 1.5, (9, 4, 3))

    f_coll = collide(f, 1)

    rho, u = macroscopic(f)
    rho_coll, u_coll = macroscopic(f_coll)

    assert np.allclose(rho, rho_coll)
    assert np.allclose(u, u_coll)

# test 2: mass and momentum conservation for any valid tau
def test_conservation_tau():

    for i in range(5):
        tau = rng.uniform(0.5, 2)
        f = rng.uniform(0.5, 1.5, (9, 4, 3))

        f_coll = collide(f, tau)

        rho, u = macroscopic(f)
        rho_coll, u_coll = macroscopic(f_coll)

        assert np.allclose(rho, rho_coll)
        assert np.allclose(u, u_coll)

# test 3: idempotence at equilibrium
def test_idempotence():

    f = rng.uniform(0.5, 1.5, (9, 4, 3))
    rho, u = macroscopic(f)
    f_eq = equilibrium(rho, u)
    f_coll = collide(f_eq, 1)

    assert np.allclose(f_coll, f_eq)