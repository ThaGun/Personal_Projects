import numpy as np
from scipy.optimize import minimize

#  MPC OUTER LOOP — PATH PLANNER
#
#  Runs at 5 Hz (every 10 sim steps).
#  Uses a simple 3D kinematic model — NOT full 6-DOF.
#  Plans a smooth path through waypoints, avoids obstacles.
#  Outputs [h_ref, V_ref, psi_ref] to LQR inner loop.
#
#  KINEMATIC MODEL (point mass, NED frame):
#    x_dot   =  V * cos(psi)          North velocity
#    y_dot   =  V * sin(psi)          East velocity
#    h_dot   =  h_cmd                 altitude rate (direct)
#    V_dot   =  V_cmd                 speed rate    (direct)
#    psi_dot =  psi_rate_cmd          heading rate  (direct)
#
#  STATE:   z = [x, y, h, V, psi]    (5 states)
#  CONTROL: u = [h_rate, V_rate, psi_rate]  (3 controls)

NZ = 5         # MPC state
NU = 3         # MPC control
N = 10          # Prediction horizon

''' State cost '''
Q_PATH  = np.diag([15.0, 15.0, 20.0, 5.0, 3.0])

''' Terminal cost '''
Qf_PATH = np.diag([30.0, 30.0, 40.0, 5.0, 3.0])

''' Control cost '''
R_PATH  = np.diag([1.0, 1.0, 2.0])

''' Physical limits '''
H_RATE_MAX   =  2.0    # m/s   max climb/descend rate
V_RATE_MAX   =  2.0    # m/s²  max accel/decel
PSI_RATE_MAX =  0.5    # rad/s max heading rate (~17°/s)
V_MIN        =  10.0   # m/s   minimum airspeed
V_MAX        =  25.0   # m/s   maximum airspeed
H_MIN        =   2.0   # m     minimum safe altitude
H_MAX        = 200.0   # m     maximum altitude

''' Obstacle avoidance '''
OBS_RADIUS  = 40.0    # m     safety margin around obstacles
OBS_PENALTY  = 5e6     # cost added per obstacle violation

def path_dynamics(z, u, dt):

    x, y, h, V, psi = z
    h_rate, V_rate, psi_rate = u

    # Clamp control to physical limits
    h_rate   = np.clip(h_rate,   -H_RATE_MAX,   H_RATE_MAX)
    V_rate   = np.clip(V_rate,   -V_RATE_MAX,   V_RATE_MAX)
    psi_rate = np.clip(psi_rate, -PSI_RATE_MAX, PSI_RATE_MAX)

    # Integrate
    x_new   = x   + V * np.cos(psi) * dt
    y_new   = y   + V * np.sin(psi) * dt
    h_new   = np.clip(h + h_rate * dt, H_MIN, H_MAX)
    V_new   = np.clip(V + V_rate * dt, V_MIN, V_MAX)
    psi_new = psi + psi_rate * dt

    return np.array([x_new, y_new, h_new, V_new, psi_new])

#  OBSTACLE AVOIDANCE COST
#  Each obstacle is a cylinder: (cx, cy, radius, h_min, h_max)
#  Cost spikes when UAV enters obstacle + margin

def obstacle_cost(z, obstacles):

    cost = 0.0
    x, y, h = z[0], z[1], z[2]

    cost = 0.0
    x, y, h = z[0], z[1], z[2]
    for (cx, cy, cz, r) in obstacles:
        dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (h-cz)**2)
        r_safe = r + OBS_RADIUS
        if dist < r_safe:
            cost += OBS_PENALTY * (r_safe - dist)**2
    
    return cost

#  WAYPOINT MANAGER
#  Sequences through waypoints automatically.
#  Switches to next waypoint when within capture radius.

class WaypointManager:
    """
    Manages a list of 3D waypoints.

    Each waypoint: (x, y, h, V)
      x, y : NED position (m)
      h    : target altitude (m)
      V    : target airspeed (m/s)
    """
    def __init__(self, waypoints, capture_radius=8.0):
        self.waypoints      = waypoints
        self.capture_radius = capture_radius
        self.current_idx    = 0
        self.completed      = False
    
    def update(self, z):
        """Check if current waypoint reached, advance if so."""
        if self.completed:
            return

        wp = self.waypoints[self.current_idx]
        dx = z[0] - wp[0]
        dy = z[1] - wp[1]
        dh = z[2] - wp[2]

        # 3D distance to waypoint
        dist = np.sqrt(dx**2 + dy**2 + dh**2)

        if dist < self.capture_radius:
            print(f"  [WP] Reached waypoint {self.current_idx} "
                  f"({wp[0]:.0f},{wp[1]:.0f},{wp[2]:.0f}m) "
                  f"at dist={dist:.1f}m")
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints):
                self.current_idx = len(self.waypoints) - 1
                self.completed   = True
                print("  [WP] All waypoints reached!")
    
    def get_target(self):
        """Return current target waypoint [x, y, h, V]."""
        return self.waypoints[self.current_idx]
    
    def get_reference_state(self, z, obstacles=None):
        """Point toward a lookahead point, not the waypoint directly."""
        wp   = self.get_target()
        dx   = wp[0] - z[0]
        dy   = wp[1] - z[1]
        dist  = np.sqrt(dx**2 + dy**2)

        # Lookahead
        lookahead = max(dist * 0.4, 60.0)
        scale     = lookahead / max(dist, 1e-3)
        psi_d = np.arctan2(dy * scale, dx * scale)

        return np.array([wp[0], wp[1], wp[2], wp[3], psi_d])

def mpc_cost(u_flat, z0, z_ref, obstacles, dt_mpc):

    z    = z0.copy()
    cost = 0.0
    u_prev = np.zeros(NU)

    for k in range(N):
        # Extract control at step k
        uk = u_flat[k*NU : (k+1)*NU]

        # State error  (heading uses shortest-path wrap)
        e      = z - z_ref
        e[4]   = np.arctan2(np.sin(e[4]), np.cos(e[4]))  # wrap heading

        # State cost
        cost  += e @ Q_PATH @ e

        # Control cost
        cost  += uk @ R_PATH @ uk

        # Obstacle cost
        cost  += obstacle_cost(z, obstacles)

        u_prev = uk

        # Propagate
        z = path_dynamics(z, uk, dt_mpc)

    # Terminal cost
    e    = z - z_ref
    e[4] = np.arctan2(np.sin(e[4]), np.cos(e[4]))
    cost += e @ Qf_PATH @ e
    cost += obstacle_cost(z, obstacles)

    return cost

class PathMPC:
    """
    Outer loop MPC path planner.

    Call update() at 5 Hz.
    Reads h_ref, V_ref from get_refs() for LQR inner loop.
    """
    def __init__(self, obstacles=None, dt_mpc=0.2):
        """
        dt_mpc : MPC timestep (s) — should be 10 × sim dt
        """
        self.obstacles = obstacles or []
        self.dt_mpc    = dt_mpc
        self.u_seq     = np.zeros(N * NU)   # warm start
        self.last_ref  = None

        # Actuator bounds for optimizer
        lo, hi = [], []
        for _ in range(N):
            lo += [-H_RATE_MAX,   -V_RATE_MAX,   -PSI_RATE_MAX]
            hi += [ H_RATE_MAX,    V_RATE_MAX,    PSI_RATE_MAX]
        self.bounds = list(zip(lo, hi))

    def update(self, z, z_ref):
        # warm start
        u_init = np.roll(self.u_seq, -NU)
        u_init[-NU:] = self.u_seq[-NU:]

        # Optimize
        res = minimize(
            mpc_cost,
            u_init,
            args=(z, z_ref, self.obstacles, self.dt_mpc),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter':100, 'ftol': 1e-4}
        )

        if res.success or res.fun < 1e6:
            self.u_seq = res.x
        
        # First control extraction
        u0 = self.u_seq[:NU]

        # Predict next state for reference output
        z_next = path_dynamics(z, u0, self.dt_mpc)

        self.last_ref = {
            'h_ref'   : float(z_next[2]),
            'V_ref'   : float(z_next[3]),
            'psi_ref' : float(z_next[4]),
            'cost'    : float(res.fun),
            'u0'      : u0.copy(),
        }

        return self.last_ref
    
    def get_refs(self):
        """Return last computed references."""
        return self.last_ref
