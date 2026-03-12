from UAV_dynamics import aero
import numpy as np

def analytic_trim(V, P):

    '''
    Force balance at level flight:
    Lift = Weight  →  q̄*S*CL = m*g  →  CL = m*g/(q̄*S)
    alpha = (CL - CL0) / CLa
    Drag  = q̄*S*CD  →  Thrust = Drag
    '''

    qbar     = 0.5 * P.rho * V**2
    CL_req   = (P.m * P.g) / (qbar * P.S)          # required lift coeff
    alpha    = (CL_req - P.CL0) / P.CLa             # angle of attack (rad)

    CD       = P.CD0 + P.CDa * alpha**2             # drag coefficient
    Drag     = qbar * P.S * CD                      # drag force (N)

    Drag_total = qbar * P.S * CD + P.m * P.g * np.sin(alpha)
    throttle   = np.clip(Drag_total / P.T_max, 0, 1)   # throttle to balance drag

    # Elevator for moment balance: Cm = 0
    # Cm0 + Cma*alpha + Cmde*delta_e = 0
    delta_e  = -(P.Cm0 + P.Cma * alpha) / P.Cmde

    print(f"\n=== Analytic Trim (V={V} m/s) ===")
    print(f"  alpha    = {np.degrees(alpha):+.3f} deg")
    print(f"  delta_e  = {np.degrees(delta_e):+.3f} deg")
    print(f"  throttle = {throttle:.4f}")
    print(f"  CL_req   = {CL_req:.4f}")

    return alpha, delta_e, throttle