# Optimal Control for a Quadrotor

<img width="420" height="308" alt="DeriveQuadrotorDynamicsForNonlinearMPCExample_01" src="https://github.com/user-attachments/assets/6f389cb9-dc6a-4e6e-a4ed-aba26e61915a" />

## Techologies💻:
1. Python
2. NumPy
3. SciPy
4. Casadi
5. LQR control

## Objective🎯:
1. To model the dynamics of a quardotor with euler angles, positions, velocities, angular rates and thurst.
2. Convert the dynamics for discrete time and linearize for a state space model.
3. Dvelope and tune LQR controller for optimal control and performance of the quadrotor.

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

3. Leniaerization with Jacobian matrix,

  $$\dot{x} = f(x,u)$$

  $$\dot{x} = Ax + Bu$$

  $$A = \frac{\partial f}{\partial x}, B = \frac{\partial f}{\partial u}$$

4. LQR controller with Q and R weight matrices,

  $$J = \int_{0}^{\infty}\left(x^{\top}(t) Q x(t)+u^{\top}(t) R u(t)\right) dt$$

  $$K = R^{-1} B^{\top} P$$

  $$u(t) = -K x(t)$$

5. Riccati equation to solve and obtain P,

  $$A^{\top} P + P A-P B R^{-1} B^{\top} P+Q = 0$$

## Developement▶️:
Dynamic model of the quadrotor:

State vector:

$$x = [p_x, p_y, p_z, v_x, v_y, v_z, \phi, \theta, \psi, p, q, r]$$

$p_x, p_y, p_z \rightarrow \text{position}$  

$v_x, v_y, v_z \rightarrow \text{velocity}$  

$\phi, \theta, \psi \rightarrow \text{roll, pitch, yaw}$  

$p, q, r \rightarrow \text{body angular rates}$


Control vector:

$$u = [T \, \tau_{\phi} \, \tau_{\theta} \, \tau_{\psi}]$$

$T \rightarrow \text{Total thrust}$  

$\tau_{\phi} \rightarrow \text{Roll torque}$  

$\tau_{\theta} \rightarrow \text{Pitch torque}$  

$\tau_{\psi} \rightarrow \text{Yaw torque}$


Continuous nonlinear dynamics:


Traslational motion:

$$\dot{p} = v$$

$$\begin{aligned}\dot{v} &= \frac{1}{m} R(\phi,\theta,\psi)\begin{bmatrix}0 ;0 ;T\end{bmatrix}-\begin{bmatrix}0 ;0 ;g\end{bmatrix}\\end{aligned}$$


Rotation matrix:

$$R = R_z(\psi)\, R_y(\theta)\, R_x(\phi)$$

$$  R =
    \left[
    \begin{matrix}
    \cos\theta \cos\psi &
    \sin\phi \sin\theta \cos\psi - \cos\phi \sin\psi &
    \cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi \\
    \cos\theta \sin\psi &
    \sin\phi \sin\theta \sin\psi + \cos\phi \cos\psi &
    \cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi \\
    -\sin\theta &
    \sin\phi \cos\theta &
    \cos\phi \cos\theta
    \end{matrix}
    \right]$$


Euler angle Kinematics:

$$  \begin{bmatrix}
    \dot{\phi} \\
    \dot{\theta} \\
    \dot{\psi}
    \end{bmatrix}
    =  
    W(\phi,\theta)
    \begin{bmatrix}
    p \\
    q \\
    r
    \end{bmatrix}$$

$$  W =
    \left[
    \begin{matrix}
    1 & \sin\phi \tan\theta & \cos\phi \tan\theta \\
    0 & \cos\phi & -\sin\phi \\
    0 & \dfrac{\sin\phi}{\cos\theta} & \dfrac{\cos\phi}{\cos\theta}
    \end{matrix}
   \right]$$


Rotationala dynamics:

$$I \dot{\omega} = \tau - \omega \times (I \omega)$$

$$\dot{\omega} = I^{-1} \left( \tau - \omega \times (I \omega) \right)$$

$$\omega = \left[ p, q, r \right]^T$$

## Process:
1. Continuous nonlinear dynamics is computed to produce $\dot{x}$ using Casadi python library.

$$\dot{x} = f(x,u)$$

2. RKF is implemented to compute discrete time approximation $x_{next}$ using Casadi python libraby.

$$x_{next} = F(x,u)$$

3. Linearization of the nonlinear function with Jacobian matrix.

$$A = \frac{\partial f}{\partial x}, B = \frac{\partial f}{\partial u}$$

4. Weights for LQR controller Q and R is probagated for respected states and control varibles.
5. Riccati equation is solved to find matrix $P$ using Casaadi python library.

                     P = solve_continuous_are(A, B, Q, R)
6. Gain matrix $K$ is calculated and fed-forward to the control matrix $B$.

$$K = R^{-1} B^{\top} P$$

$$u(t) = -K x(t)$$

7. Weight matrices of LQR is tuned for optimal performance.

## Results✅:
Result for circular hover motion:

<img width="1536" height="754" alt="Circular_hover" src="https://github.com/user-attachments/assets/b96ffb32-adb0-4645-9628-e13e48e95bc0" />

<img width="1536" height="754" alt="Circular_trajectory" src="https://github.com/user-attachments/assets/9aa59b84-83de-4af2-94e3-729c5c3ae996" />

Results of spiral evolve motion:

<img width="1536" height="754" alt="Spiral_evolve" src="https://github.com/user-attachments/assets/bb2e5737-f7ff-4e34-aab1-f1f08fd1d754" />

<img width="1536" height="754" alt="Spiral_trajectory" src="https://github.com/user-attachments/assets/46625cbc-06a6-4dec-a2d1-d566377f9f13" />

Results of sinusoidal hover motion:

<img width="1536" height="754" alt="Sinusoidal_hover" src="https://github.com/user-attachments/assets/b4fbdaaf-5b4a-4800-befc-0cd924da81fb" />

<img width="1536" height="754" alt="Sinusoidal_trajectory" src="https://github.com/user-attachments/assets/4f832b4b-0f13-4933-b955-3ccec187569e" />

Results of square evolve motion:

<img width="1536" height="754" alt="Square_evolve" src="https://github.com/user-attachments/assets/a045d970-54d0-46d5-8965-774f358730a5" />

<img width="1536" height="754" alt="Square_trajectory" src="https://github.com/user-attachments/assets/9777fe30-d6ed-45a0-87ee-b0f18dfeb1f5" />
