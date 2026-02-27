import numpy as np
import casadi as ca
from QD_control_theory import LQR, LQR_control, Integral_state, controllability
from QD_trajectory import Circle_trajectory, Spiral_trajectory, Sinusoidal_trajectory, Waypoint_trajectory, Square_trajectory


def LQR_sim(A, B, F, x0, N, dt, u_hover, Rc, Omega):

    ''' LQR Weight matrix '''
    Q = np.diag([10, 10, 100,       # Position
                1, 1, 50,          # Vecocity
                200, 200, 10,     # Angles
                1, 1, 100])         # Angular rates

    R = np.diag([10, 10, 10, 10])

    # print("A matrix:", A)
    # print("B matrix:", B)

    C_rank = controllability(A, B)

    print("Controllability rank:", C_rank)

    K = LQR(A, B, Q, R)

    # print("LQR gain K:\n", K)
    # print("K shape:", K.shape)

    x_ref = np.zeros(12)

    x_h = np.zeros((12, N+1))
    x_h[:,0] = x0
    u_h = np.zeros((4, N+1))
    x_error = np.zeros((12, N+1))
    x_refer = np.zeros((12, N+1))

    # Integral action
    integral_pos = np.zeros(3)
    Ki = np.array([2.0, 2.0, 5.0])

    ''' LQR control '''
    for k in range(N):

        ''' Open loop sim '''
        # x0 = F(ca.DM(x0), ca.DM(u_hover))
        # x0 = np.array(x0.full()).flatten()
        # x_h[:,k+1] = x0

        ''' Closed loop sim '''
        #x_ref = Circle_trajectory(Rc, Omega, x_ref, k, dt)

        #x_ref = Spiral_trajectory(Rc, Omega, x_ref, k, dt)

        #x_ref = Sinusoidal_trajectory(Rc, Omega, x_ref, k, dt)

        x_ref = Square_trajectory(Rc, Omega, x_ref, k, dt)

        wps = [(0,0), (2,0), (2,2), (0,2), (0,0)]

        #x_ref = Waypoint_trajectory(wps, x_ref, k, dt)

        u_current = LQR_control(x0, x_ref, u_hover, K)

        error_x = Integral_state(integral_pos[0], Ki[0], x0[0], x_ref[0], dt)
        error_y = Integral_state(integral_pos[1], Ki[1], x0[1], x_ref[1], dt)
        error_z = Integral_state(integral_pos[0], Ki[2], x0[2], x_ref[2], dt)

        u_current[0] -= error_z      # Thurst integral
        u_current[1] -= error_y      # Roll integral
        u_current[2] -= error_x      # Pitch integral

        # Nonlinear probagation
        x_next = F(ca.DM(x0), ca.DM(u_current))
        x0 = np.array(x_next.full()).flatten()

        x_h[:,k+1] = x0
        u_h[:,k+1] = u_current

        x_error[:,k+1] = x0 - x_ref
        x_refer[:,k+1] = x_ref
    
    return x_h, u_h, x_error, x_refer