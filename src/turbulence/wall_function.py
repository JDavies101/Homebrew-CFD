# log-law wall function: Newton inversion for the friction velocity given the first-node speed at distance y1
# host copy of Simulation3D.wall_utau (the kernel can't call numpy), keep the two in sync
import numpy as np

def friction_velocity(first_node_speed, y1, viscosity, von_karman=0.41, log_law_constant=5.2, iterations=50, tolerance=1e-8):
    """
    Solve u1 = u_tau (ln(y1 u_tau / nu) / kappa + B) for u_tau by Newton iteration.

    Returns u_tau (0 for non-positive first-node speed).
    """

    if first_node_speed <= 0:
        return 0.0

    inverse_von_karman = 1 / von_karman
    # initial guess: viscous sublayer estimate
    u_tau = np.sqrt(viscosity * first_node_speed / y1)

    for _ in range(iterations):
        log_term = inverse_von_karman * np.log(y1 * u_tau / viscosity) + log_law_constant
        residual = u_tau * log_term - first_node_speed

        if abs(residual) < tolerance:
            break

        residual_derivative = log_term + inverse_von_karman
        u_tau_next = u_tau - residual / residual_derivative

        u_tau = u_tau_next if u_tau_next > 0 else u_tau / 2  # guard the proposed value

    return u_tau
