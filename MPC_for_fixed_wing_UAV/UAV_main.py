import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from UAV_dynamics import rkf_step, quat_to_euler
from UAV_trim import analytic_trim
import UAV_lqr as LQR
from mpc_path import PathMPC, WaypointManager, OBS_RADIUS

''' Run UAV_lqr_design.py before running UAV_main,py '''

''' LQR Compile C++ '''
LQR.compile_lqr()

''' Parameters '''
class Params:
    def __init__(self):
        # Physical
        self.m   = 1.5      # mass (kg)
        self.g   = 9.81     # gravity (m/s²)
        self.rho = 1.225    # air density (kg/m³)

        # Geometry
        self.S = 0.25       # wing area (m²)
        self.b = 1.2        # wingspan (m)
        self.c = 0.20       # mean chord (m)

        # Inertia (kg·m²)
        self.Ix  = 0.03
        self.Iy  = 0.04
        self.Iz  = 0.05
        self.Ixz = 0.002  

        # Lift:  CL = CL0 + CLa*alpha
        self.CL0 =  0.30
        self.CLa =  5.50

        # Drag:  CD = CD0 + CDa*alpha²
        self.CD0 =  0.02
        self.CDa =  0.30

        # Pitch: Cm = Cm0 + Cma*alpha + Cmde*delta_e
        self.Cm0  =  0.02
        self.Cma  = -1.50
        self.Cmde = -1.00

        # Propulsion
        self.T_max = 15.0   # max thrust (N)

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

''' Setup '''
P = Params()
dt = 0.05
dt_mpc = 0.2
mpc_ratio  = int(dt_mpc / dt)
T = 120.0
steps = int(T/dt)

''' Initial state '''
# x = [u, v, w, p, q, r, q0, q1, q2, q3, pN, pE, pD]
x = np.zeros(13)

''' LQR inner loop '''
gains = LQR.init()
x_trim_full = gains["x_trim_full"]
u_trim_full = gains["u_trim_full"]
V_trim      = gains["V_trim"]
_, theta_trim, _ = quat_to_euler(x_trim_full[6:10])
_, _, throttle   = analytic_trim(V_trim, P)

''' State extract functions '''
def extract_lon(x_full):
    _, theta, _ = quat_to_euler(x_full[6:10])
    return np.array([
        x_full[0],     # u
        x_full[2],     # w
        x_full[4],     # q
        theta,         # pitch angle
    ])

def extract_lat(x_full):
    roll, _, yaw = quat_to_euler(x_full[6:10])
    return np.array([x_full[1], x_full[3], x_full[5], roll, yaw])

def get_kinematic_state(x_full):
    # Extract MPC kinematic state from full 13-state.
    # z = [x_NED, y_NED, h, V, psi]
    x    = x_full[10]               # North
    y    = x_full[11]               # East
    h    = -x_full[12]              # altitude
    V    = np.linalg.norm(x_full[0:3])
    _, _, psi = quat_to_euler(x_full[6:10])
    return np.array([x, y, h, V, psi])

''' Waypoints and Obstacles '''
waypoints = [
    (  400,   0,  20, 15),   # WP0: straight north
    (  1000,   0,  20, 15),   # WP1: continue north
    (  1400,   100,  10, 15),   # WP3: continue north along east
    (  1800,   -100,  5, 15),   # WP4: continue north along west
]

obstacles = [
    # (cx,   cy,  cz, radius)
    (700,    0,   20,    30),   # dead centre of path 
]

wp_manager = WaypointManager(waypoints, capture_radius=30.0)
path_mpc   = PathMPC(obstacles=obstacles, dt_mpc=dt_mpc)

''' Initial state with trim '''
V_trim = 15.0
alpha, delta_e, throttle = analytic_trim(V_trim, P)
x_trim = np.zeros_like(x)

x_trim[0] = V_trim * np.cos(alpha)    # u (forward body velocity)
x_trim[2] = V_trim * np.sin(alpha)    # w (downward body velocity)
x_trim[6] = np.cos(alpha / 2)         # q0 (quaternion scalar)
x_trim[8] = np.sin(alpha / 2)         # q2 (quaternion y-component, pure pitch)

''' Fixed control (Initial)'''
# [delta_e, delta_a, delta_r, throttle]
u = np.array([0.0, 0.0, 0.0, 0.5])

u_trim = np.array([delta_e, 0.0, 0.0, throttle])

print(f"\nInitial state:")
print(f"  u={x_trim[0]:.3f} m/s  w={x_trim[2]:.3f} m/s")
print(f"  q=[{x_trim[6]:.4f}, {x_trim[7]:.4f}, {x_trim[8]:.4f}, {x_trim[9]:.4f}]")
print(f"u_trim = {u_trim}")

# ── Verify trim holds ────────────────────────
x_test = x_trim.copy()
print("\nTrim stability check:")
for i in range(5):
    x_test = rkf_step(x_test, u_trim, dt, P)
    print(f"  step {i+1}: alt={-x_test[12]:.5f}m  "
          f"u={x_test[0]:.4f}  "
          f"thr_used={throttle:.4f}  "
          f"pitch={np.degrees(quat_to_euler(x_test[6:10])[1]):.4f}°")

''' NED: climb to 20m → pD = -20m '''
ALT_TARGET = 30.0
pD_target  = -ALT_TARGET

''' Reference state '''
x_ref      = x_trim.copy()
x_ref[12]  = pD_target


''' Outer loop for Longitude'''
# Simple PI controller
Kh         = 0.02    # altitude to pitch gain (rad/m)
theta_max  = np.radians(15.0)   # max pitch command
h_integrator = 0.0
Ki           = 0.002

''' Heading controller gains '''
K_psi   = 0.8    # heading error → roll rate
K_roll  = 0.8    # roll angle → aileron
bank_max = np.radians(20.0)    # max bank angle

''' Initial state '''
xt    = x_trim_full.copy()
xt[12] = 0.0
u_seq = None

''' Outer loop MPC references — start at trim '''
h_ref   = waypoints[0][2]      # current altitude
V_ref   = V_trim
psi_ref = 0.0

''' Storage '''
hist_x   = [xt.copy()]
hist_u   = []
hist_href = []
hist_zref = []

# Print initial state
z0 = get_kinematic_state(xt)
print(f"Initial kinematic state:")
print(f"  x(North)={z0[0]:.2f}  y(East)={z0[1]:.2f}")
print(f"  h={z0[2]:.2f}  V={z0[3]:.2f}")
print(f"  psi={np.degrees(z0[4]):.2f}°")
print(f"WP0 target: N={waypoints[0][0]}  E={waypoints[0][1]}")
psi_to_wp0 = np.degrees(np.arctan2(waypoints[0][1]-z0[1],
                                    waypoints[0][0]-z0[0]))
print(f"Required heading to WP0: {psi_to_wp0:.2f}°")

print(f"\n{'t':>6} | {'x':>7} | {'y':>7} | {'alt':>7} | {'h_ref':>7} | "
      f"{'V':>6} | {'de°':>6} | {'thr':>5} | {'WP':>3}")
print("-" * 80)

''' Simulation '''

for k in range(steps):
    t = k *dt

    if not np.all(np.isfinite(x)):
        print(f"Diverged at t={t:.2f}s")
        break

    ''' Kinematic state '''
    z_now = get_kinematic_state(xt)

    ''' MPC outer loop: runs every mpc_ratio steps (5 Hz) '''
    if k % mpc_ratio == 0:

        # Update waypoint manager
        wp_manager.update(z_now)

        # Hold position at final waypoint
        if not wp_manager.completed:
            z_ref  = wp_manager.get_reference_state(z_now)
            mpc_out = path_mpc.update(z_now, z_ref)
            psi_ref  = mpc_out['psi_ref']    # ← from waypoint manager directly
            wp_now   = wp_manager.get_target()
            h_ref   = wp_now[2]
            V_ref   = wp_now[3]
            # MPC only for altitude/speed planning, not heading
            mpc_out  = path_mpc.update(z_now, z_ref)
        else:
            z_ref    = wp_manager.get_reference_state(z_now, obstacles)
            mpc_out  = path_mpc.update(z_now, z_ref)
            wp_now   = wp_manager.get_target()
            h_ref    = wp_now[2]
            V_ref    = wp_now[3]
            psi_ref  = mpc_out['psi_ref']
        
        hist_zref.append(np.array([t, h_ref, V_ref,
                                    np.degrees(psi_ref)]))
    
    ''' Longitudinal LQR Inner loop: runs everu step (50Hz) '''
    x_lon  = extract_lon(xt)        # [u, w, q, theta]
    h_err   = h_ref - (-xt[12])

    ''' Outer altitude loop: h error → theta reference '''
    h_integrator += h_err * dt
    h_integrator  = float(np.clip(h_integrator, -50.0, 50.0))

    theta_ref = float(np.clip(
        theta_trim + Kh * h_err + Ki * h_integrator,
        theta_trim - theta_max,
        theta_trim + theta_max
    ))

    ''' Throttle direct law: proportional to altitude + speed error '''
    V_now = np.linalg.norm(xt[0:3])
    thr_ref = float(np.clip(
        throttle + 0.8*(h_err/max(abs(h_ref)+1, 20))
                 + 0.3*(V_ref - V_now)/V_ref,
        0.04, 1.0
    ))

    ''' LQR: pitch + airspeed tracking '''
    x_ref_lon = np.array([V_ref, 0.0, 0.0, theta_ref])
    u_lon     = LQR.lon_step(x_lon, x_ref_lon)

    ''' Lateral LQR loop '''
    x_lat = extract_lat(xt)
    x_ref_lat = np.array([
        0.0,      # no sideslip
        0.0,      # wings level (p=0)
        0.0,      # no yaw rate
        0.0,      # wings level (phi=0)
        psi_ref   # ← heading from MPC outer loop
    ])
    u_lat     = LQR.lat_step(x_lat, x_ref_lat)  # [da, dr]
    
    ''' Full control vector '''
    u_full = np.array([u_lon[0], u_lat[0], u_lat[1], thr_ref])

    hist_u.append(u_full.copy())
    hist_href.append(h_ref)
    
    ''' Integrate 6-DOF dynamics '''
    xt = rkf_step(xt, u_full, dt, P)
    hist_x.append(xt.copy())

    ''' Print every 1s '''
    if k % int(1.0/dt) == 0:
        alt = -xt[12]
        V   = np.linalg.norm(xt[0:3])
        wp  = wp_manager.current_idx
        print(f"{t:6.1f} | {z_now[0]:7.1f} | {z_now[1]:7.1f} | "
              f"{alt:7.2f} | {h_ref:7.2f} | {V:6.2f} | "
              f"{np.degrees(u_full[0]):6.2f} | {u_full[3]:5.3f} | WP{wp}")
    if wp_manager.completed and k % mpc_ratio == 0:
        print(f"t={t:.1f} MISSION COMPLETE — holding at WP3")
        break    # ← stop sim when done

hist_x   = np.array(hist_x)
hist_u   = np.array(hist_u)
hist_href = np.array(hist_href)
n        = len(hist_x)
time_x   = np.linspace(0, T, n)
time_u   = np.linspace(0, T, len(hist_u))

''' Derived '''
altitude  = -hist_x[:, 12]
airspeed  = np.linalg.norm(hist_x[:, 0:3], axis=1)
pN        = hist_x[:, 10]
pE        = hist_x[:, 11]
euler_all = np.array([quat_to_euler(hist_x[i, 6:10]) for i in range(n)])
pitch_deg = np.degrees(euler_all[:, 1])
yaw_deg   = np.degrees(euler_all[:, 2])

''' Plots '''
# ── 2D Plots ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 10))

axes[0,0].plot(time_x, altitude,  'b',   lw=2, label='Altitude')
axes[0,0].plot(time_u, hist_href, 'r--', lw=1.5, label='MPC Altitude Reference')
axes[0,0].set_ylabel('Altitude (m)')
axes[0,0].set_ylim(0, 40)
axes[0,0].set_xlabel('Time (s)')
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(pE, pN, 'b', lw=2, label='X Y phase path')
axes[0,1].scatter(pE[0],  pN[0],  s=100, c='green',
           zorder=5, label='Start')
axes[0,1].scatter(pE[-1], pN[-1], s=100, c='red',
           zorder=5, label='End')
wp_N = [wp[0] for wp in waypoints]
wp_E = [wp[1] for wp in waypoints]
axes[0,1].scatter(wp_E, wp_N, s=120, c='orange',
           marker='^', zorder=5, label='Waypoints')
for i, wp in enumerate(waypoints):
    axes[0,1].text(wp[1]+5, wp[0]+5, f'WP{i}', fontsize=9)
for (cx, cy, cz, r) in obstacles:
    axes[0,1].scatter(cy, cx, s=40, c='darkred', zorder=5)
    axes[0,1].text(cy+3, cx+3, 'OBS', color='darkred',
            fontsize=9, fontweight='bold')
axes[0,1].set_ylabel('North X-axis (m)')
axes[0,1].set_xlabel('East Y-axis (m)')
axes[0,1].set_xlim(-1000, 1000)
axes[0,1].legend(); axes[0,1].grid(True)

axes[0,2].plot(time_x, pitch_deg,  'darkorange', lw=2, label='Pitch')
axes[0,2].plot(time_x, yaw_deg,    'purple',     lw=1.5, label='Heading')
axes[0,2].set_ylabel('Angle (°)')
axes[0,2].set_xlabel('Time (s)')
axes[0,2].legend(); axes[0,2].grid(True)

axes[1,0].step(time_u, np.degrees(hist_u[:,0]), 'r',
             where='post', lw=2, label='δe Elevator control (°)')
axes[1,0].step(time_u, hist_u[:,3], 'g',
             where='post', lw=2, label='Throttle control')
axes[1,0].set_ylabel('Control')
axes[1,0].set_xlabel('Time (s)') 
axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(time_x, airspeed, 'navy', lw=2)
axes[1,1].axhline(V_trim, color='red', ls='--', lw=1.5, label='V_trim')
axes[1,1].set_ylabel('Airspeed (m/s)')
axes[1,1].set_xlabel('Time (s)'); axes[1,1].grid(True)

axes[1,2].step(time_u, np.degrees(hist_u[:,1]), 'r',
             where='post', lw=2, label='δa Ailerons control (°)')
axes[1,2].step(time_u, np.degrees(hist_u[:,2]), 'g',
             where='post', lw=2, label='δr Rudder control (°)')
axes[1,2].set_xlabel('Time (s)')
axes[1,2].set_ylabel('Control')
axes[1,2].legend()
axes[1,2].grid(True)

fig.suptitle("MPC for fixed wing UAV movement", fontsize='16')
plt.tight_layout()

# 3D flight path
fig3 = plt.figure(figsize=(11, 8))
ax3  = fig3.add_subplot(111, projection='3d')

ax3.plot(pN, pE, altitude, 'royalblue', lw=2, label='Flight path')
ax3.scatter(pN[0], pE[0], altitude[0], color='green', s=100,
            zorder=5, label='Start')
ax3.scatter(pN[-1], pE[-1], altitude[-1], color='red', s=100,
            zorder=5, label='End')

# Plot waypoints
for i, wp in enumerate(waypoints):
    ax3.scatter(wp[0], wp[1], wp[2], color='orange',
                s=120, marker='^', zorder=5)
    ax3.text(wp[0]+2, wp[1]+2, wp[2]+2, f'WP{i}', fontsize=9)

# ── Plot obstacles as scatter spheres ────────────────────────
for (cx, cy, cz, r) in obstacles:
    n_pts  = 800
    phi    = np.random.uniform(0,    np.pi,   n_pts)
    theta  = np.random.uniform(0, 2*np.pi,    n_pts)
    r_plot = r * 0.001    # visual radius (slightly smaller than avoidance)

    sx = cx + r_plot * np.sin(phi) * np.cos(theta)
    sy = cy + r_plot * np.sin(phi) * np.sin(theta)
    sz = cz + r_plot * np.cos(phi)

    # Mark centre
    ax3.scatter([cx], [cy], [cz],
                c='darkred', s=100, marker='o', zorder=10)

    # Safety radius ring at obstacle altitude
    theta_ring = np.linspace(0, 2*np.pi, 60)
    ax3.plot(cx + (r + OBS_RADIUS)*np.cos(theta_ring),
             cy + (r + OBS_RADIUS)*np.sin(theta_ring),
             np.full(60, cz),
             'r--', lw=1.0, alpha=0.5, label='Safety radius')

ax3.set_xlabel('North (m)'); ax3.set_ylabel('East (m)')
ax3.set_zlabel('Altitude (m)')
ax3.set_title('3D Flight Path — Waypoints + Obstacles')
ax3.set_zlim(-40, 80)
ax3.set_ylim(-1000, 1000)
ax3.legend(); ax3.grid(True)

plt.tight_layout()
plt.show()

print(f"\nFinal position: N={pN[-1]:.1f}m  E={pE[-1]:.1f}m  "
      f"h={altitude[-1]:.1f}m")
print(f"Final airspeed: {airspeed[-1]:.2f} m/s")
print(f"Waypoint: {wp_manager.current_idx}/{len(waypoints)}  "
      f"({'COMPLETE' if wp_manager.completed else 'in progress'})")

## How it all fits together
'''
Every sim step (50 Hz, dt=0.05s):
│
├─ k % 10 == 0?  → YES: run PathMPC.update()
│                        - kinematic rollout N=15 steps × 0.2s = 3s horizon
│                        - SLSQP optimizer avoids obstacles
│                        - outputs h_ref, V_ref, psi_ref
│
└─ Every step: LQR.step()
               - altitude error → theta_ref  (P+I outer loop)
               - LQR tracks [V_ref, 0, 0, theta_ref]
               - outputs delta_e, throttle
               - rk4_step() integrates full 6-DOF
               
'''