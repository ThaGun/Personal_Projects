# MPC for fixed wing UAV movement

<img width="258" height="205" alt="Airplane_axes_(PSF)" src="https://github.com/user-attachments/assets/d9cd689a-48dc-432f-bf89-5fe662c9a190" />

## Techologies💻:
1. Python
2. NumPy
3. SciPy
4. C++
5. MPC (Model predictive control)
6. LQR controller
7. PI COntroller

## Objective🎯:
1. To model the dynamics of a fixed wing UAV with quaternions, positions, velocities, angular rates and thurst.
2. Convert the dynamics for discrete time and linear estimation for a state space model.
3. Dvelope and tune MPC for optimal path and obstacle avoidance of the UAV.

## Methodology⚙️:
1. Runge - Kutta function for the conversion of discrete time dynamics,
  
  $$x_{k+1} = x_k + \frac{\Delta t}{6}\left( K_1 + 2K_2 + 2K_3 + K_4 \right)$$
  
  $$
  \begin{aligned}
  K_1 &= f(x, u) \\
  K_2 &= f\left(x + \frac{dt}{2}*K_1, u\right) \\
  K_3 &= f\left(x + \frac{dt}{2}*K_2, u\right) \\
  K_4 &= f\left(x + dt * K_3, u \right) \\
  \end{aligned}
  $$

3. Lineaerization with Jacobian matrix,

  $$\dot{x} = f(x,u)$$

  $$\dot{x} = Ax + Bu$$

  $$A = \frac{\partial f}{\partial x}, B = \frac{\partial f}{\partial u}$$

4. LQR controller with Q and R weight matrices,

  $$J = \int_{0}^{\infty}\left(x^{\top}(t) Q x(t)+u^{\top}(t) R u(t)\right) dt$$

  $$K = R^{-1} B^{\top} P$$

  $$u(t) = -K x(t)$$

5. Riccati equation to solve and obtain P,

  $$A^{\top} P + P A-P B R^{-1} B^{\top} P+Q = 0$$

6. MPC (Model Predictive control) for non-linear system,

The system is represented using a nonlinear discrete-time state-space model:

  $$x_{k+1} = f(x_k, u_k)$$

  $$y_k = h(x_k, u_k)$$
  
    Future states are predicted over the prediction horizon Np.

  $$x_{k+1} = f(x_k, u_k)$$

  $$x_{k+2} = f(x_{k+1}, u_{k+1})$$

    General prediction:

  $$x_{k+i+1} = f(x_{k+i}, u_{k+i})$$

for $$i = 0,1,\dots,N_p-1$$

    Predicted outputs:

  $$y_{k+i} = h(x_{k+i}, u_{k+i})$$

    MPC minimizes a nonlinear cost function over the prediction horizon.

$$
J =
\sum_{i=0}^{N_p-1}
(x_{k+i} - x_{ref})^T Q (x_{k+i} - x_{ref})
+
\sum_{i=0}^{N_c-1}
u_{k+i}^T R u_{k+i}
+
(x_{k+N_p} - x_{ref})^T P (x_{k+N_p} - x_{ref})
$$

Where:

- $x_{ref}$ : reference state
- $Q$ : state tracking weight matrix
- $R$ : control effort weight matrix
- $P$ : terminal weight matrix
- $N_p$ : prediction horizon
- $N_c$ : control horizon

7. Obstacle avoidance:

$$
\text{dist} = \sqrt{(x - c_x)^2 + (y - c_y)^2 + (h - c_z)^2}
$$


## Developement▶️:
Dynamic model of the quadrotor:

State vector:

$$x = [u, v, w, p, q, r, q_0, q_1, q_2, q_3, x_N, y_E, z_D]$$

$u, v, w \rightarrow \text{velocities}$  

$p, q, r \rightarrow \text{angular velocities}$  

$q_0, q_1, q_2, q_3 \rightarrow \text{quaternion}$  

$x_N, y_E, z_D \rightarrow \text{NED position}$


Control vector:

$$u = [\delta_{e} \, \delta_{a} \, \delta_{r}, T]$$

$\delta_{e} \rightarrow \text{Elevator deflection}$  

$\delta_{a} \ \rightarrow \text{Aileron deflection}$  

$\delta_{r} \rightarrow \text{Rudder deflection}$

$T \rightarrow \text{Total thrust}$

$c_x, c_y, c_z$ - Obstacle coordinates

Continuous nonlinear dynamics:

Aerodynamic angles:

    Airspeed:
  
$$
V = \sqrt{u^2 + v^2 + w^2}
$$

    Angle of attack:

$$
\alpha = \arctan\left(\frac{w}{u}\right)
$$

    Sideslip:

$$
\beta = \arcsin\left(\frac{v}{V}\right)
$$

    Dynamic Pressure:

$$
\bar{q} = \frac{1}{2}\rho V^2
$$

Forces (body frame):

$$
C_L = C_{L0} + C_{L\alpha}\alpha
$$

$$
C_D = C_{D0} + C_{D\alpha}\alpha^2
$$

$$
\text{Lift} = \bar{q} \, S \, C_L
$$

(Perpendicular to velocity)

$$
\text{Drag} = \bar{q} \, S \, C_D
$$

(Parallel to velocity and opposing)

    X-axis force:
$$
F_x = \bar{q}\, S \left(-C_D c_a + C_L s_a\right)
$$

    Y-axis force:
$$
F_y = \bar{q}\, S \left(C_{Y\beta}\beta + C_{Y\delta_r}\delta_r\right)
$$

    Z-axis force:
$$
F_z = \bar{q}\, S \left(-C_D s_a - C_L c_a\right)
$$

Gravity in body frame (NED)

    NED grvity vector:

$$
\mathbf{g}_{NED} =
\begin{bmatrix}
0 \\
0 \\
g
\end{bmatrix}
$$

    Body frame gravity:

$$
\mathbf{g}_{body} = R^{T}\mathbf{g}_{NED}
$$

    For pure pitch θ = alpha:

$$
g_{x,body} = g \sin(\alpha)
$$

$$
g_{z,body} = g \cos(\alpha)
$$

Traslational dynamics:

$$
\dot{u} = \frac{F_x}{m} + r v - q w
$$

$$
\dot{v} = \frac{F_y}{m} + p w - r u
$$

$$
\dot{w} = \frac{F_z}{m} + q u - p v
$$

$$
\dot{\mathbf{v}}_{\text{body}} =
\frac{\mathbf{F}_{\text{total}}}{m} -
\boldsymbol{\omega} \times \mathbf{v}_{\text{body}}
$$


Rotational dynamics:

$$
I \, \dot{\boldsymbol{\omega}} =
\mathbf{M} - \boldsymbol{\omega} \times (I \boldsymbol{\omega})
$$

$$
\dot{\boldsymbol{\omega}} =
I^{-1}\left[
\mathbf{M} - \boldsymbol{\omega} \times (I \boldsymbol{\omega})
\right]
$$
 
$$\boldsymbol{\omega} = \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$ 



Quaternion Kinematics:

    Avoids gimbal lock of Euler angles.

$$\dot{\mathbf{q}} = \frac{1}{2}\,\Xi(\boldsymbol{\omega})\,\mathbf{q}$$

$$
\Xi(\boldsymbol{\omega}) =
\begin{bmatrix}
0 & -p & -q & -r \\
p & 0 & r & -q \\
q & -r & 0 & p \\
r & q & -p & 0
\end{bmatrix}
$$


## Process🪜:
1. Continuous nonlinear dynamics is computed to produce $\dot{x}$ using NumPy python library.

$$\dot{x} = f(x,u)$$

2. RKF is implemented to compute discrete time approximation $x_{next}$ using NumPy python libraby.

$$x_{next} = F(x,u)$$

3. Linearization of the nonlinear function with Jacobian matrix.

$$A = \frac{\partial f}{\partial x}, B = \frac{\partial f}{\partial u}$$

4. For longitudinal control, a cascade control system is designed with LQR in inner loop and PI in outer loop.
5. Weights for LQR controller Q and R is probagated for respected states and control varibles.
6. PI is tuned in respect to fast response.
7. Riccati equation is solved to find matrix $P$ using SciPy python library.

                     P = solve_continuous_are(A, B, Q, R)
8. Gain matrix $K$ is calculated and fed-forward to the control matrix $B$.

$$K = R^{-1} B^{\top} P$$

$$u(t) = -K x(t)$$

7. Weight matrices of LQR is tuned for optimal performance.
8. Another LQR controller is designed for Lateral control and tuned with the same procedures.
9. MPC is designed in the outer loop of Longitudinal and Lateral controllers.
10. MPC predicts an optimal path and feed the optimal reference signals to both Longitudinal and Lateral controllers.
11. MPC cost is formulated and minimized with SciPy Python library for an optimal path with respect to waypoint following and obstacle avoidance,

## Results✅:
Result for UAV movement with MPC:

<img width="1536" height="754" alt="3d_with_obs2" src="https://github.com/user-attachments/assets/7165a42d-b611-4c3d-adf7-bda12d9f3c62" />

<img width="1536" height="754" alt="results" src="https://github.com/user-attachments/assets/055fe0e9-76ad-4a2d-acb0-ee2932e02f30" />
