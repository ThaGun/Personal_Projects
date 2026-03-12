import numpy as np
from scipy.linalg import solve_continuous_are
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from UAV_dynamics import rkf_step, quat_to_euler
from UAV_trim     import analytic_trim

# ═══════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════
class Params:
    def __init__(self):
        self.m=1.5; self.g=9.81; self.rho=1.225
        self.S=0.25; self.b=1.2; self.c=0.20
        self.Ix=0.03; self.Iy=0.04; self.Ixz = 0.002; self.Iz=0.05
        self.CL0=0.30; self.CLa=5.50
        self.CD0=0.02; self.CDa=0.30
        self.Cm0=0.02; self.Cma=-1.50; self.Cmde=-1.00
        self.T_max=15.0
        # Side force
        self.Cyb  = -0.30    # side force / sideslip
        self.Cydr =  0.20    # side force / rudder

        # Roll moment
        self.Clb  = -0.12    # dihedral effect (negative = stable)
        self.Clp  = -0.40    # roll damping    (negative = stable)
        self.Clr  =  0.08    # roll from yaw rate
        self.Clda =  0.08    # aileron effectiveness
        self.Cldr =  0.02    # rudder-to-roll coupling

        # Yaw moment
        self.Cnb  =  0.10    # weathercock stability (positive = stable)
        self.Cnp  = -0.05    # yaw from roll rate
        self.Cnr  = -0.10    # yaw damping    (negative = stable)
        self.Cnda = -0.02    # adverse yaw from aileron
        self.Cndr = -0.08    # rudder effectiveness

#  LONGITUDINAL STATE  [u, w, q, theta, h]

def extract_lon(x_full):
    u     = x_full[0]
    w     = x_full[2]
    q     = x_full[4]
    _, theta, _ = quat_to_euler(x_full[6:10])
    return np.array([u, w, q, theta])

def full_from_lon(x_lon, x_trim_full):
    x         = x_trim_full.copy()
    u,w,q,theta = x_lon
    x[0]  = u
    x[2]  = w
    x[4]  = q
    x[6]  = np.cos(theta/2)
    x[7]  = 0.0
    x[8]  = np.sin(theta/2)
    x[9]  = 0.0
    return x

def lon_step(x_lon, u_lon, x_trim_full, u_trim_full, P, dt=0.01):
    x_full    = full_from_lon(x_lon, x_trim_full)
    u_full    = u_trim_full.copy()
    u_full[0] = u_lon[0]    # delta_e
    u_full[3] = u_lon[1]    # throttle
    x_next = rkf_step(x_full, u_full, dt, P)
    return extract_lon(x_next)

def linearize(x_trim_full, u_trim_full, P, dt=0.01):
    x0  = extract_lon(x_trim_full)
    u0  = np.array([u_trim_full[0], u_trim_full[3]])
    NX_LON = len(x0)
    NU_LON = len(u0)
    ex  = np.array([0.1, 0.05, 0.01, 0.001])
    eu  = np.array([0.01, 0.01])

    # A discrete
    Ad = np.zeros((NX_LON, NX_LON))
    for i in range(NX_LON):
        dx=np.zeros(NX_LON); dx[i]=ex[i]
        xp = lon_step(x0+dx, u0, x_trim_full, u_trim_full, P, dt)
        xm = lon_step(x0-dx, u0, x_trim_full, u_trim_full, P, dt)
        Ad[:,i] = (xp-xm)/(2*ex[i])

    # B discrete
    Bd = np.zeros((NX_LON, NU_LON))
    for i in range(NU_LON):
        du=np.zeros(NU_LON); du[i]=eu[i]
        xp = lon_step(x0, u0+du, x_trim_full, u_trim_full, P, dt)
        xm = lon_step(x0, u0-du, x_trim_full, u_trim_full, P, dt)
        Bd[:,i] = (xp-xm)/(2*eu[i])

    # Discrete → Continuous
    Ac = (Ad - np.eye(NX_LON)) / dt
    Bc = Bd / dt
    return Ac, Bc


def design_lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


if __name__ == "__main__":
    P = Params()

    V_trim = 15.0
    alpha, delta_e, throttle = analytic_trim(V_trim, P)
    x_trim = np.zeros(13)
    x_trim[0] = V_trim * np.cos(alpha)
    x_trim[2] = V_trim * np.sin(alpha)
    x_trim[6] = np.cos(alpha/2)
    x_trim[8] = np.sin(alpha/2)
    u_trim = np.array([delta_e, 0.0, 0.0, throttle])

    print("=" * 55)
    print("  UAV LONGITUDINAL LQR DESIGN")
    print("=" * 55)
    print(f"\nTrim: V={V_trim} m/s  alpha={np.degrees(alpha):.3f}°")
    print(f"      de={np.degrees(delta_e):.3f}°  thr={throttle:.4f}")

    # Linearize
    print("\nLinearizing dynamics at trim...")
    A, B = linearize(x_trim, u_trim, P)

    labels = ['u','w','q','θ']
    print("\nA matrix:")
    print(f"  {'':5s}", end='')
    for l in labels: print(f"  {l:>8s}", end='')
    print()
    for i,l in enumerate(labels):
        print(f"  {l:5s}", end='')
        for j in range(4): print(f"  {A[i,j]:8.4f}", end='')
        print()

    print("\nB matrix:")
    print(f"  {'':5s}  {'de':>8s}  {'thr':>8s}")
    for i,l in enumerate(labels):
        print(f"  {l:5s}  {B[i,0]:8.4f}  {B[i,1]:8.4f}")

    # Eigenvalues open-loop
    eigs_ol = np.linalg.eigvals(A)
    print("\nOpen-loop eigenvalues:")
    for i,e in enumerate(eigs_ol):
        s = "stable ✓" if e.real < 0 else "UNSTABLE ✗"
        print(f"  λ{i+1} = {e.real:8.4f} + {e.imag:7.4f}j  {s}")

    # LQR weights  [u, w, q, theta, h]
    Q_lqr = np.diag([10.0, 1.0, 10.0, 50.0])
    R_lqr = np.diag([500.0, 1.0])

    print("\nSolving Riccati equation...")
    K_lon = design_lqr(A, B, Q_lqr, R_lqr)

    print("\nLQR Gain Matrix K (2×5):")
    print(f"  {'':8s}  {'u':>8s}  {'w':>8s}  {'q':>8s}  {'θ':>8s}  {'h':>8s}")
    for i,cn in enumerate(['delta_e','throttle']):
        print(f"  {cn:8s}", end='')
        for j in range(4): print(f"  {K_lon[i,j]:8.4f}", end='')
        print()

    # Closed-loop check
    A_cl   = A - B @ K_lon
    eigs_cl = np.linalg.eigvals(A_cl)
    print("\nClosed-loop eigenvalues:")
    all_ok = True
    for i,e in enumerate(eigs_cl):
        s = "stable ✓" if e.real < 0 else "UNSTABLE ✗"
        if e.real >= 0: all_ok = False
        print(f"  λ{i+1} = {e.real:8.4f} + {e.imag:7.4f}j  {s}")
    print(f"\n  {'✓ LQR VALID' if all_ok else '✗ CHECK WEIGHTS'}")

'''Lateral gain'''

#  LATERAL SUBSPACE
#  State:   x_lat = [v, p, r, phi, psi]   (5 states)
#  Control: u_lat = [delta_a, delta_r]     (2 controls)

def extract_lat(x_full):
    """Full 13-state → lateral 5-state."""
    v   = x_full[1]
    p   = x_full[3]
    r   = x_full[5]
    roll, _, yaw = quat_to_euler(x_full[6:10])
    return np.array([v, p, r, roll, yaw])


def full_from_lat(x_lat, x_trim_full):
    """Lateral 5-state → full 13-state (longitudinal held at trim)."""
    x = x_trim_full.copy()
    v, p, r, phi, psi = x_lat
    x[1] = v
    x[3] = p
    x[5] = r
    # Reconstruct quaternion from trim pitch + current roll/yaw
    _, theta_trim, _ = quat_to_euler(x_trim_full[6:10])
    # Roll-Pitch-Yaw → quaternion
    cr, sr = np.cos(phi/2),   np.sin(phi/2)
    cp, sp = np.cos(theta_trim/2), np.sin(theta_trim/2)
    cy, sy = np.cos(psi/2),   np.sin(psi/2)
    x[6] = cr*cp*cy + sr*sp*sy
    x[7] = sr*cp*cy - cr*sp*sy
    x[8] = cr*sp*cy + sr*cp*sy
    x[9] = cr*cp*sy - sr*sp*cy
    return x


def lat_step(x_lat, u_lat, x_trim_full, u_trim_full, P, dt=0.01):
    """One RK4 step in lateral subspace."""
    x_full    = full_from_lat(x_lat, x_trim_full)
    u_full    = u_trim_full.copy()
    u_full[1] = u_lat[0]   # delta_a
    u_full[2] = u_lat[1]   # delta_r
    x_next    = rkf_step(x_full, u_full, dt, P)
    return extract_lat(x_next)


def linearize_lat(x_trim_full, u_trim_full, P, dt=0.01):
    """Numerical linearization of lateral subspace."""
    x0     = extract_lat(x_trim_full)
    u0     = np.array([u_trim_full[1], u_trim_full[2]])  # [da, dr]
    NX_LAT = len(x0)   # = 5
    NU_LAT = len(u0)   # = 2

    ex = np.array([0.05, 0.01, 0.01, 0.01, 0.01])
    eu = np.array([0.01, 0.01])

    Ad = np.zeros((NX_LAT, NX_LAT))
    for i in range(NX_LAT):
        dx = np.zeros(NX_LAT); dx[i] = ex[i]
        xp = lat_step(x0+dx, u0, x_trim_full, u_trim_full, P, dt)
        xm = lat_step(x0-dx, u0, x_trim_full, u_trim_full, P, dt)
        Ad[:, i] = (xp - xm) / (2*ex[i])

    Bd = np.zeros((NX_LAT, NU_LAT))
    for i in range(NU_LAT):
        du = np.zeros(NU_LAT); du[i] = eu[i]
        xp = lat_step(x0, u0+du, x_trim_full, u_trim_full, P, dt)
        xm = lat_step(x0, u0-du, x_trim_full, u_trim_full, P, dt)
        Bd[:, i] = (xp - xm) / (2*eu[i])

    Ac = (Ad - np.eye(NX_LAT)) / dt
    Bc = Bd / dt
    return Ac, Bc


# Lateral LQR weights
#  States:   [v,    p,    r,    phi,   psi  ]
Q_lat = np.diag([1.0,  8.0,  8.0,  30.0,  80.0])
#  Controls: [da,   dr  ]
R_lat = np.diag([30.0, 15.0])

print("\nLinearizing lateral dynamics at trim...")
Ac_lat, Bc_lat = linearize_lat(x_trim, u_trim, P)

print("Solving lateral CARE...")
P_lat = solve_continuous_are(Ac_lat, Bc_lat, Q_lat, R_lat)
K_lat = np.linalg.solve(R_lat, Bc_lat.T @ P_lat)

# Closed-loop check
eigs_lat = np.linalg.eigvals(Ac_lat - Bc_lat @ K_lat)
print("Lateral closed-loop eigenvalues:")
for i, ev in enumerate(eigs_lat):
    stability = "STABLE" if ev.real < 0 else "UNSTABLE ←"
    print(f"  λ{i} = {ev.real:+.4f} {ev.imag:+.4f}j  {stability}")

labels_lat = ['v', 'p', 'r', 'φ', 'ψ']
print("\nLateral K matrix:")
for i, row in enumerate(K_lat):
    vals = "  ".join(f"{v:+8.4f}" for v in row)
    print(f"  K[{'da' if i==0 else 'dr'}] = [ {vals} ]  "
          f"← [{', '.join(labels_lat)}]")

gains = {
    # Longitudinal (existing)
    "K_lon"        : K_lon.tolist(),
    "x_trim_lon"   : extract_lon(x_trim).tolist(),
    "u_trim_lon"   : [float(u_trim[0]), float(u_trim[3])],
    # Lateral (new)
    "K_lat"        : K_lat.tolist(),
    "x_trim_lat"   : extract_lat(x_trim).tolist(),
    "u_trim_lat"   : [float(u_trim[1]), float(u_trim[2])],
    # Shared
    "x_trim_full"  : x_trim.tolist(),
    "u_trim_full"  : u_trim.tolist(),
    "V_trim"       : float(V_trim),
}
with open("lqr_gains.json", "w") as f:
    json.dump(gains, f, indent=2)
print("\nSaved lqr_gains.json  (lon + lat)")

import json
with open("lqr_gains.json") as f:
    g = json.load(f)
print(list(g.keys()))