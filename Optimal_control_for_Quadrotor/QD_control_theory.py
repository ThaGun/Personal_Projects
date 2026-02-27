import casadi as ca
from numpy.linalg import matrix_rank, matrix_power
import numpy as np
from scipy.linalg import solve_continuous_are

def linerization(f):

    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 4)

    xdot = f(x, u)             # non linear matrix

    ''' Jacobian '''
    A_sym = ca.jacobian(xdot, x)
    B_sym = ca.jacobian(xdot, u)

    ''' Function '''
    A_fun = ca.Function('A_fun', [x, u], [A_sym])
    B_fun = ca.Function('B_fun', [x, u], [B_sym])

    return A_fun, B_fun

def controllability(A, B):

    Ctrb = B

    for i in range(1,12):
        Ctrb = np.hstack((Ctrb, matrix_power(A,i) @ B))
    
    return matrix_rank(Ctrb)

def LQR(A, B, Q, R):

    # Q and R are weighting matrices

    ''' Solve Riccati '''
    P = solve_continuous_are(A, B, Q, R)

    ''' Compute gain '''
    K = np.linalg.inv(R) @ B.T @ P

    return K

def LQR_control(x, x_hover, u_hover, K):

    u_current = u_hover - K @ (x - x_hover)

    return u_current

def Integral_state(I, Ki, x0, x_ref, dt):

    error = x0 - x_ref
    I += error * dt

    total_error = Ki * I

    return total_error