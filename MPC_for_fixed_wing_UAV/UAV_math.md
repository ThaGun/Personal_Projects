
## The Math First

### State Vector
```
x = [u, v, w,           ← body velocities (m/s)
     p, q, r,           ← angular rates (rad/s)
     q0, q1, q2, q3,    ← quaternion attitude
     pN, pE, pD]        ← NED position (m)
```

### Aerodynamic Angles
```
V     = sqrt(u² + v² + w²)     airspeed
alpha = arctan(w / u)           angle of attack
beta  = arcsin(v / V)           sideslip
q̄    = 0.5 * rho * V²          dynamic pressure
```

### Forces (body frame)
```
CL = CL0 + CLa*alpha
CD = CD0 + CDa*alpha²

Lift  = q̄ * S * CL     (perpendicular to velocity)
Drag  = q̄ * S * CD     (parallel to velocity, opposing)

Rotate wind→body frame via alpha:
  Fx =  -Drag*cos(alpha) + Lift*sin(alpha) + Thrust
  Fz =  -Drag*sin(alpha) - Lift*cos(alpha)
```

### Gravity in Body Frame (NED)
```
NED gravity vector: g_NED = [0, 0, +g]
                                      ↑ Z points DOWN in NED

Body frame gravity: g_body = Rᵀ @ g_NED
(R maps body→NED, so Rᵀ maps NED→body)

For pure pitch θ = alpha:
  gx_body =  g * sin(alpha)
  gz_body =  g * cos(alpha)
```

### Translational Dynamics (Newton in rotating body frame)
```
u̇ = Fx/m + rv - qw     ← Coriolis: ω × v
v̇ = Fy/m + pw - ru
ẇ = Fz/m + qu - pv

Vector form:  v̇_body = F_total/m  -  ω × v_body
```

### Rotational Dynamics (Euler's equations)
```
I * ω̇ = M  -  ω × (I*ω)

ω̇ = I⁻¹ * [M  -  ω × (I*ω)]
```

### Quaternion Kinematics
```
Avoids gimbal lock of Euler angles.

q̇ = 0.5 * Ξ(ω) * q

      ⎡  0  -p  -q  -r ⎤
Ξ(ω) =⎢  p   0   r  -q ⎥
      ⎢  q  -r   0   p ⎥
      ⎣  r   q  -p   0 ⎦
```

### Position Kinematics
```
ṗ_NED = R(q) * v_body

R rotates body-frame velocity into NED inertial frame
```

### RK4 Integration
```
k1 = f(x)
k2 = f(x + dt/2 * k1)
k3 = f(x + dt/2 * k2)
k4 = f(x + dt   * k3)

x_next = x + dt/6 * (k1 + 2k2 + 2k3 + k4)

Then normalize quaternion to prevent drift.

## MPC

Current state: x0
Reference:     x_ref  (where we want to be)

Solve this optimization:

  minimize:   J = Σ_{k=0}^{N-1} [ eₖᵀQeₖ + Δuₖᵀ R Δuₖ + δuₖᵀ Rd δuₖ ]
                                    ↑state    ↑input      ↑smoothness
                + eₙᵀ Qf eₙ         ← terminal cost

  where:
    eₖ   = x(k) - x_ref          state error at step k
    Δuₖ  = u_trim + du(k)         actual control
    δuₖ  = du(k) - du(k-1)        input rate (change per step)
    x(k+1) = RK4(x(k), Δuₖ)      predicted next state

  subject to:
    u_min ≤ u_trim + du(k) ≤ u_max    actuator limits

Apply ONLY du(0), then re-solve next timestep (receding horizon).

Q   penalizes state error      → pulls UAV toward reference
R   penalizes large inputs     → prevents aggressive deflections
Rd  penalizes input changes    → prevents oscillation/chattering
Qf  penalizes terminal error   → ensures convergence by end of horizon