// ═══════════════════════════════════════════════════════════
//  LQR LONGITUDINAL CONTROLLER  —  C++ inner loop
//
//  u = u_trim - K @ (x_lon - x_ref_lon)
//
//  x_lon[5]  = [u, w, q, theta, h]
//  u_lon[2]  = [delta_e, throttle]
//
//  No optimizer — just one matrix multiply. Ultra fast.
// ═══════════════════════════════════════════════════════════

#include <cmath>
#include <cstdio>
#include <algorithm>

static constexpr int NX = 4;
static constexpr int NU = 2;

// ── Controller state ────────────────────────────────────────
static double g_K[NU][NX]   = {};   // gain matrix
static double g_x_trim[NX]  = {};   // trim state
static double g_u_trim[NU]  = {};   // trim control
static double g_u_min[NU]   = {};   // actuator limits
static double g_u_max[NU]   = {};
static bool   g_ready       = false;

// ── Math helpers ────────────────────────────────────────────
static inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ═══════════════════════════════════════════════════════════
//  C INTERFACE
// ═══════════════════════════════════════════════════════════
extern "C" {

// ── lqr_init ─────────────────────────────────────────────
//  K_flat[10]     : K row-major  [K_de_u, K_de_w, ..., K_thr_h]
//  x_trim_lon[5]  : trim [u, w, q, theta, h]
//  u_trim_lon[2]  : trim [delta_e, throttle]
//  u_min[2]       : [de_min, thr_min]
//  u_max[2]       : [de_max, thr_max]
void lqr_init(const double* K_flat,
              const double* x_trim_lon,
              const double* u_trim_lon,
              const double* u_min,
              const double* u_max) {

    for (int i = 0; i < NU; i++)
        for (int j = 0; j < NX; j++)
            g_K[i][j] = K_flat[i * NX + j];

    for (int i = 0; i < NX; i++) g_x_trim[i] = x_trim_lon[i];
    for (int i = 0; i < NU; i++) g_u_trim[i] = u_trim_lon[i];
    for (int i = 0; i < NU; i++) g_u_min[i]  = u_min[i];
    for (int i = 0; i < NU; i++) g_u_max[i]  = u_max[i];

    g_ready = true;

    printf("[LQR] Initialized\n");
    printf("      K[de ] = [%7.4f %7.4f %7.4f %7.4f %7.4f]\n",
           g_K[0][0],g_K[0][1],g_K[0][2],g_K[0][3],g_K[0][4]);
    printf("      K[thr] = [%7.4f %7.4f %7.4f %7.4f %7.4f]\n",
           g_K[1][0],g_K[1][1],g_K[1][2],g_K[1][3],g_K[1][4]);
    printf("      u_trim = [de=%.4f  thr=%.4f]\n",
           g_u_trim[0], g_u_trim[1]);
}

// ── lqr_step ─────────────────────────────────────────────
//  One LQR control step.
//
//  x_lon[5]    : current longitudinal state  [u,w,q,theta,h]
//  x_ref_lon[5]: reference longitudinal state [u_ref,w_ref,q_ref,theta_ref,h_ref]
//  u_out[2]    : control output [delta_e, throttle]  (write-back)
//
//  Law:  e = x_lon - x_ref_lon
//        u = u_trim - K @ e
//        u = clamp(u, u_min, u_max)
void lqr_step(const double* x_lon,
              const double* x_ref_lon,
                    double* u_out) {

    if (!g_ready) {
        u_out[0] = g_u_trim[0];
        u_out[1] = g_u_trim[1];
        return;
    }

    // State error
    double e[NX];
    for (int i = 0; i < NX; i++)
        e[i] = x_lon[i] - x_ref_lon[i];

    // u = u_trim - K @ e
    for (int i = 0; i < NU; i++) {
        double Ke = 0.0;
        for (int j = 0; j < NX; j++)
            Ke += g_K[i][j] * e[j];
        u_out[i] = clamp(g_u_trim[i] - Ke,
                         g_u_min[i],
                         g_u_max[i]);
    }
}

// ── lqr_reset ────────────────────────────────────────────
void lqr_reset() {
    g_ready = false;
    printf("[LQR] Reset.\n");
}

} // extern "C"