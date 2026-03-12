import numpy as np

def normalize(q):
    ''' Quaternion to unit length '''
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_to_DCM(q):
    ''' Rotation matrix from quaternion '''
    q0, q1, q2, q3 = q
    R = np.array([
        [1-2*(q2**2+q3**2),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [  2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2),   2*(q2*q3-q0*q1)],
        [  2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])
    return R

def omega_matrix(omega):
    ''' Quaternion kinematics matrix '''
    p, q, r = omega
    matrix = np.array([
        [ 0, -p, -q, -r],
        [ p,  0,  r, -q],
        [ q, -r,  0,  p],
        [ r,  q, -p,  0]
    ], dtype=float)
    return matrix

def quat_to_euler(q):
    ''' Quaternion → [roll, pitch, yaw] in radians '''
    q0, q1, q2, q3 = q
    roll  = np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2))
    pitch = np.arcsin(np.clip(2*(q0*q2-q3*q1), -1, 1))
    yaw   = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))
    return np.array([roll, pitch, yaw])

def aero(x, u, P):
    """
    Full 6-DOF aerodynamic forces and moments.
    x = [u,v,w, p,q,r, q0,q1,q2,q3, pN,pE,pD]
    u = [delta_e, delta_a, delta_r, throttle]
    Returns: F[3], M[3] in body frame
    """
    ub, vb, wb = x[0], x[1], x[2]
    p,  q,  r  = x[3], x[4], x[5]
    de, da, dr, thr = u[0], u[1], u[2], u[3]

    V     = np.sqrt(ub**2 + vb**2 + wb**2)
    V     = max(V, 1e-3)          # prevent division by zero
    alpha = np.arctan2(wb, ub)    # angle of attack
    beta  = np.arcsin( vb / V)    # sideslip angle

    qbar  = 0.5 * P.rho * V**2

    # ── Longitudinal aerodynamics ──────────────────
    CL   = P.CL0 + P.CLa * alpha
    CD   = P.CD0 + P.CDa * alpha**2
    Cm   = P.Cm0 + P.Cma * alpha + P.Cmde * de

    # Wind-axis → body-axis
    ca, sa = np.cos(alpha), np.sin(alpha)
    Fx_aero = qbar * P.S * (-CD * ca + CL * sa)
    Fz_aero = qbar * P.S * (-CD * sa - CL * ca)

    # Thrust (along body x)
    Fx_thrust = thr * P.T_max
    Fx = Fx_aero + Fx_thrust
    Fz = Fz_aero

    # Pitch moment
    My = qbar * P.S * P.c * Cm

    # Lateral-directional aerodynamics
    # Normalised rates
    pb_2V = p * P.b / (2 * V)
    rb_2V = r * P.b / (2 * V)

    # Side force
    Fy = qbar * P.S * (P.Cyb * beta + P.Cydr * dr)

    # Roll moment
    Mx = qbar * P.S * P.b * (
        P.Clb  * beta  +
        P.Clp  * pb_2V +
        P.Clr  * rb_2V +
        P.Clda * da    +
        P.Cldr * dr
    )

    # Yaw moment
    Mz = qbar * P.S * P.b * (
        P.Cnb  * beta  +
        P.Cnp  * pb_2V +
        P.Cnr  * rb_2V +
        P.Cnda * da    +
        P.Cndr * dr
    )

    F = np.array([Fx, Fy, Fz])
    M = np.array([Mx, My, Mz])
    return F, M

# 6-DOF dynamics

def dynamics(x, u, P):
    ub,vb,wb = x[0],x[1],x[2]
    p, q, r  = x[3],x[4],x[5]
    q0,q1,q2,q3 = x[6],x[7],x[8],x[9]

    F, M = aero(x, u, P)

    # Translational acceleration (body frame)
    # Coriolis terms: ω × V
    du = F[0]/P.m - q*wb + r*vb
    dv = F[1]/P.m - r*ub + p*wb
    dw = F[2]/P.m - p*vb + q*ub

    # Gravity in body frame via quaternion rotation
    gx = 2*(q1*q3 - q0*q2) * P.g
    gy = 2*(q2*q3 + q0*q1) * P.g
    gz = (q0**2 - q1**2 - q2**2 + q3**2) * P.g
    du += gx;  dv += gy;  dw += gz

    # Rotational acceleration (with Ixz coupling) 
    Gamma = P.Ix*P.Iz - P.Ixz**2
    dp = (P.Ixz*(P.Ix - P.Iy + P.Iz)*p*q
          - (P.Iz*(P.Iz - P.Iy) + P.Ixz**2)*q*r
          + P.Iz*M[0] + P.Ixz*M[2]) / Gamma

    dq = ((P.Iz - P.Ix)*p*r - P.Ixz*(p**2 - r**2)
          + M[1]) / P.Iy

    dr = ((P.Ix*(P.Ix - P.Iy) + P.Ixz**2)*p*q
          - P.Ixz*(P.Ix - P.Iy + P.Iz)*q*r
          + P.Ixz*M[0] + P.Ix*M[2]) / Gamma

    # Quaternion kinematics
    dq0 = 0.5*(-q1*p - q2*q - q3*r)
    dq1 = 0.5*( q0*p - q3*q + q2*r)
    dq2 = 0.5*( q3*p + q0*q - q1*r)
    dq3 = 0.5*(-q2*p + q1*q + q0*r)

    # NED position kinematics
    # Rotate body velocity to NED via quaternion
    R11 = q0**2+q1**2-q2**2-q3**2
    R12 = 2*(q1*q2-q0*q3)
    R13 = 2*(q1*q3+q0*q2)
    R21 = 2*(q1*q2+q0*q3)
    R22 = q0**2-q1**2+q2**2-q3**2
    R23 = 2*(q2*q3-q0*q1)
    R31 = 2*(q1*q3-q0*q2)
    R32 = 2*(q2*q3+q0*q1)
    R33 = q0**2-q1**2-q2**2+q3**2

    dpN = R11*ub + R12*vb + R13*wb
    dpE = R21*ub + R22*vb + R23*wb
    dpD = R31*ub + R32*vb + R33*wb

    return np.array([du,dv,dw, dp,dq,dr,
                     dq0,dq1,dq2,dq3,
                     dpN,dpE,dpD])

def rkf_step(x, u_ctrl, dt, P):

    x =np.asarray(x, dtype=float).ravel()

    k1 = dynamics(x, u_ctrl, P)
    k2 = dynamics(x + 0.5*dt*k1, u_ctrl, P)
    k3 = dynamics(x + 0.5*dt*k2, u_ctrl, P)
    k4 = dynamics(x + dt*k3, u_ctrl, P)

    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x_next[6:10] = normalize(x_next[6:10])   # keep quaternion unit length

    return x_next