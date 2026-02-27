# Optimal Control for a Quadrotor

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
1. 

## Results✅:

