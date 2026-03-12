import numpy as np
import ctypes, os, platform, subprocess, sys, json

# ═══════════════════════════════════════════════════════════
#  PYTHON WRAPPER FOR C++ LQR CONTROLLER
# ═══════════════════════════════════════════════════════════

_base = os.path.dirname(os.path.abspath(__file__))
_dll  = os.path.join(_base, "lqr_core.dll" if platform.system()=="Windows"
                             else "lqr_core.so")

# ── Auto-compile ─────────────────────────────────────────
def compile_lqr():
    src = os.path.join(_base, "lqr_core.cpp")
    out = _dll
    if (os.path.exists(out) and
        os.path.getmtime(out) > os.path.getmtime(src)):
        print("[LQR] lqr_core.dll up to date.")
        return
    print("[LQR] Compiling lqr_core.cpp ...")
    gxx = r"C:\mingw64\bin\g++.exe"
    cmd = [gxx, "-O3", "-std=c++17", "-shared", "-o", out, src]
    r   = subprocess.run(cmd, capture_output=True, text=True, cwd=_base)
    if r.returncode != 0:
        print("Compile FAILED:"); print(r.stderr); sys.exit(1)
    print(f"[LQR] Compiled → {out}")

# ── Load library ─────────────────────────────────────────
compile_lqr()
_lib = ctypes.CDLL(_dll)

_dp = ctypes.POINTER(ctypes.c_double)

_lib.lqr_init.restype  = None
_lib.lqr_init.argtypes = [_dp, _dp, _dp, _dp, _dp]

_lib.lqr_step.restype  = None
_lib.lqr_step.argtypes = [_dp, _dp, _dp]

_lib.lqr_reset.restype  = None
_lib.lqr_reset.argtypes = []

def _p(a): return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(_dp)


# ── Public API ───────────────────────────────────────────

def init():
    """Load gains, initialise both LQR controllers."""
    global _K_lon, _K_lat
    global _x_trim_lon, _u_trim_lon
    global _x_trim_lat, _u_trim_lat
    global _x_trim_full, _u_trim_full, _V_trim
    global _umin_lon, _umax_lon
    global _umin_lat, _umax_lat

    base = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base, "lqr_gains.json")
    
    print(f"Loading gains from: {json_path}")   # ← add this to confirm path
    
    with open(json_path) as f:
        g = json.load(f)
    
    print(f"Keys found: {list(g.keys())}")       # ← add this to confirm keys

    _K_lon        = np.array(g["K_lon"])
    _x_trim_lon   = np.array(g["x_trim_lon"])
    _u_trim_lon   = np.array(g["u_trim_lon"])

    _K_lat        = np.array(g["K_lat"])
    _x_trim_lat   = np.array(g["x_trim_lat"])
    _u_trim_lat   = np.array(g["u_trim_lat"])

    _x_trim_full  = np.array(g["x_trim_full"])
    _u_trim_full  = np.array(g["u_trim_full"])
    _V_trim       = float(g["V_trim"])

    # Longitudinal limits: [delta_e, throttle]
    _umin_lon = np.array([-0.15, 0.04])
    _umax_lon = np.array([ 0.15, 1.00])

    # Lateral limits: [delta_a, delta_r]
    _umin_lat = np.array([-0.40, -0.30])
    _umax_lat = np.array([ 0.40,  0.30])

    return g


def lon_step(x_lon, x_ref_lon):
    """
    Longitudinal LQR step.
    x_lon     = [u, w, q, theta]
    x_ref_lon = [V_ref, 0, 0, theta_ref]
    Returns   = [delta_e, throttle]
    """
    e   = x_lon - x_ref_lon
    u   = _u_trim_lon - _K_lon @ e
    return np.clip(u, _umin_lon, _umax_lon)


def lat_step(x_lat, x_ref_lat):
    """
    Lateral LQR step.
    x_lat     = [v, p, r, phi, psi]
    x_ref_lat = [0, 0, 0, 0, psi_ref]
    Returns   = [delta_a, delta_r]
    """
    e    = x_lat - x_ref_lat

    # Wrap heading error to ±π
    e[4] = float(np.arctan2(np.sin(e[4]), np.cos(e[4])))

    u    = _u_trim_lat - _K_lat @ e
    return np.clip(u, _umin_lat, _umax_lat)


def reset():
    _lib.lqr_reset()