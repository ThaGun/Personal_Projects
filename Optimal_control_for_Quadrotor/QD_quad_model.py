import numpy as np
import casadi as ca

def quad_model_symbolic(par):
    ''' Parameters '''
    m = par["m"]
    g = par["g"]
    Ixx = par["Ixx"]
    Iyy = par["Iyy"]
    Izz = par["Izz"]
    
    ''' State Vector '''
    x = ca.SX.sym('x', 12)

    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    #phi = theta = psi = 0

    ''' Control vector '''
    u = ca.SX.sym('u', 4)

    T = u[0]
    tau_phi = u[1]
    tau_theta = u[2]
    tau_psi = u[3]

    ''' Rotation matrix '''
    cphi = ca.cos(phi)
    sphi = ca.sin(phi)
    ctheta = ca.cos(theta)
    stheta = ca.sin(theta)
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)

    R = ca.vertcat(
        ca.horzcat(ctheta*cpsi,
                   ctheta*spsi,
                   -stheta),
        ca.horzcat(sphi*stheta*cpsi - cphi*spsi,
                   sphi*stheta*spsi + cphi*cpsi,
                   sphi*ctheta),
        ca.horzcat(cphi*stheta*cpsi + sphi*spsi,
                   cphi*stheta*spsi - sphi*cpsi,
                   cphi*ctheta)
    )
    
    ''' Translational dynamics '''
    thurst_body = ca.vertcat(0, 0, T)

    accel = (1/m) * R @ thurst_body + ca.vertcat(0, 0, -g)

    ''' Euler kinematics '''
    W = ca.vertcat(
        ca.horzcat(1, sphi*ca.tan(theta), cphi*ca.tan(theta)),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi/ctheta, cphi/ctheta)
    )

    euler_dot = W @ ca.vertcat(p, q, r)

    ''' Rotational dynamics '''
    omega = ca.vertcat(p, q, r)
    tau = ca.vertcat(tau_phi, tau_theta, tau_psi)

    I = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
    omega_dot = ca.solve(I, tau - ca.cross(omega, I @ omega))

    ''' State derivation '''
    xdot = ca.vertcat(
        vx, vy, vz,
        accel,
        euler_dot,
        omega_dot
    )
    f = ca.Function('f', [x, u], [xdot])

    return f

def discrete_dynamic(f, dt):

    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 4)

    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)


    x_next = x + dt/6 *(k1 + 2*k2 + 2*k3 + k4)

    F = ca.Function('F', [x, u], [x_next])

    return F