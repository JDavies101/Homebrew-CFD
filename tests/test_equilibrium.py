import numpy as np
from src.lbm.moments import macroscopic
from src.lbm.equilibrium import equilibrium
from src.lbm import lattice as lt

nx = 4
ny = 3
# test 1: round trip
def test_round_trip():
    
    rho = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    u = np.full((2, nx, ny), 0.05)
    f_eq = equilibrium(rho, u)
    rho_check, u_check = macroscopic(f_eq)
    assert np.allclose(rho_check, rho)
    assert np.allclose(u_check, u)

# test 2: zero velocity gives baseline
def test_zero_velocity():

    u = np.zeros((2, nx, ny))
    rho = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    f_eq = equilibrium(rho, u)
    assert np.allclose(lt.W[:, None, None] * rho, f_eq)

# test 3: hand compute value
def test_computed():

    rho = np.array([[1]], dtype=np.int32)
    u = np.zeros((2, 1, 1))
    u[0, 0, 0] = 0.1
    f_eq = equilibrium(rho, u)
    # rho=1, u=(0.1,0), direction 1 (East): (1/9)*(1 + 0.3 + 0.045 - 0.015) = 1.33/9
    assert np.allclose(f_eq[1, 0, 0], 1.33 / 9)