# log-law wall function: newton inversion for u_tau given the first-node speed u1 at distance y1
# host copy of Simulation3D.wall_utau (the kernel can't call numpy), keep the two in sync
import numpy as np

def friction_velocity(u1, y1, nu, k=0.41, B=5.2, iters=50, atol=1e-8):

    if u1 <= 0:
        return 0.0

    u_tau = np.sqrt(nu * u1 / y1) # initial guess of the viscous/gradient estimate

    for _ in range(iters):
        f = u_tau * ((1 / k) * np.log(y1 * u_tau / nu) + B) - u1
        
        if abs(f) < atol:
            break

        f_prime = (1 / k) * np.log(y1 * u_tau / nu) + B + 1 / k
        next = u_tau - f / f_prime

        u_tau = next if next > 0 else u_tau / 2 # guard the proposed value
    
    return u_tau