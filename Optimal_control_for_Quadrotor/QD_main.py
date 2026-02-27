from QD_quad_model import quad_model_symbolic, discrete_dynamic
from QD_control_theory import linerization
from QD_lqr_sim import LQR_sim
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

''' Parameters '''
par = {
    "m" : 1.5,
    "g" : 9.81,
    "Ixx" : 0.02,
    "Iyy" : 0.02,
    "Izz" : 0.04
}

''' Simulation parameters '''
T_sim = 20.0
dt= 0.02
N = int(T_sim/dt)

''' Quadrotor model call'''
f = quad_model_symbolic(par)
F = discrete_dynamic(f, dt)
A_fun, B_fun = linerization(f)

''' Initial conditions '''
x0 = np.zeros(12)
#x0[6] = np.deg2rad(20)

x_ref = np.zeros(12)

''' Hover Condition '''
x_hover = np.zeros(12)
u_hover = np.array([(par["m"]*par["g"]), 0, 0, 0])

u_min = np.array([0, -0.5, -0.5, -0.5])
u_max = np.array([2*par["m"]*par["g"], 0.5, 0.5, 0.5])

''' LQR parameters'''
A = np.array(A_fun(x0, u_hover))
B = np.array(B_fun(x0, u_hover))

# ''' MPC parameters '''
# N_mpc = 10
# u_prev=None

''' Circular trajectory '''
R_c = 1.0
omega = 0.5

''' Simulation '''
print("Sim Start!!")

''' LQR control '''
x_h, u_h, x_error, x_refer = LQR_sim(A, B, F, x0, N, dt, u_hover, R_c, omega)

print("Done!!")

''' Plots '''
time = np.linspace(0, T_sim, N+1)

fig, axes = plt.subplots(2, 3, figsize=(13, 10))

axes[0,0].plot(time, x_h[2,:], label="Actual altitude")
axes[0,0].plot(time, x_refer[2,:], linestyle=':', color='r', label="Reference altitude")
axes[0,0].set_ylabel("Altitude (m)")
axes[0,0].set_xlabel("Time (s)")
axes[0,0].legend()
axes[0,0].grid(True)

axes[0,1].plot(x_h[0,:], x_h[1,:], label="Actual trajectory")
axes[0,1].plot(x_refer[0,:], x_refer[1,:], linestyle=':', color='r', label="Reference trajectory")
axes[0,1].set_ylabel("X Trajectary (m)")
axes[0,1].set_xlabel("Y Trajectory (m)")
axes[0,1].legend()
axes[0,1].grid(True)

axes[0,2].plot(time, x_h[3,:], label="vx")
axes[0,2].plot(time, x_h[4,:], label="vy")
axes[0,2].plot(time, x_h[5,:], label="vz")
axes[0,2].set_ylabel("Velocities (m/s)")
axes[0,2].set_xlabel("Time (s)")
axes[0,2].legend()
axes[0,2].grid(True)

axes[1,0].plot(time, x_h[6,:], label="phi")
axes[1,0].plot(time, x_h[7,:], label="theta")
axes[1,0].plot(time, x_h[8,:], label="psi")
axes[1,0].set_ylabel("Euler angles (rad)")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].legend()
axes[1,0].grid(True)

axes[1,1].plot(time, u_h[0,:], label="T")
axes[1,1].set_ylabel("Thurst (N)")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].legend()
axes[1,1].grid(True)

axes[1,2].plot(time, u_h[1,:], label="phi")
axes[1,2].plot(time, u_h[2,:], label="theta")
axes[1,2].plot(time, u_h[3,:], label="psi")
axes[1,2].set_ylabel("Angular inputs (rad)")
axes[1,2].set_xlabel("Time (s)")
axes[1,2].legend()
axes[1,2].grid(True)

fig.suptitle("Optimal control for quadrotor movement (Square trajectory)", fontsize='16')
plt.tight_layout()

plt.show()

""" 3D plot """
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_h[0,:], x_h[1,:], x_h[2,:], linestyle=':', label=" Controlled trajectory")
ax.set_xlabel('X (roll)')
ax.set_ylabel('Y (pitch)')
ax.set_zlabel('Z (altitude)')
ax.set_title('Quadrotor Trajectory')

xa = x_h[0,:]
ya = x_h[1,:]
za = x_h[2,:]

line, = ax.plot([], [], [], lw=2, color='orangered', label="Actual trajectory")
point, = ax.plot([], [], [], marker='o', markersize=15, color='teal', label="Quadrotor")

ax.set_xlim(min(xa), max(xa))
ax.set_ylim(min(ya), max(ya))
ax.set_zlim(min(za), max(za))

def update(frame):
    line.set_data(xa[:frame], ya[:frame])
    line.set_3d_properties(za[:frame])

    #point._offsets3d = ([xa[frame]], [ya[frame]], [za[frame]])
    point.set_data([xa[frame]], [ya[frame]])
    point.set_3d_properties([za[frame]])
    return line, point


anim = FuncAnimation(fig, update, frames=len(xa), interval=1)
plt.legend()
plt.show()
