import numpy as np
import matplotlib.pyplot as plt
from Function import function

# Penalized function
def penalized_function(x, y, penalty_weight=1):
    return function(x, y) + penalty_weight * (x - y)**2

# PSO parameters
n_particles = 30
max_iter    = 500
bounds      = (-5.12, 5.12)
w           = 0.7     # inertia weight
c1          = 1.5     # cognitive (personal) weight
c2          = 1.5     # social (global) weight
v_max       = 0.1     # max velocity clamp

# Positions — random in search space
positions = np.random.uniform(bounds[0], bounds[1], size=(n_particles, 2))
positions[0] = np.array([4.0, 3.5])    # seed on constraint line

# Velocities — start small and random
velocities = np.random.uniform(-v_max, v_max, size=(n_particles, 2))

# Personal best — each particle starts at its own position
pbest_pos  = positions.copy()
pbest_cost = np.array([penalized_function(p[0], p[1]) for p in pbest_pos])

# Global best — best among all personal bests
gbest_idx  = np.argmin(pbest_cost)
gbest_pos  = pbest_pos[gbest_idx].copy()
gbest_cost = pbest_cost[gbest_idx]

# History
history_x    = []
history_y    = []
history_cost = []
penalty_hist = []
found        = False

# Loop
for i in range(max_iter):
    # Update velocity and position for all particles
    r1 = np.random.rand(*velocities.shape)    # random matrix same shape as vel
    r2 = np.random.rand(*velocities.shape)

    cognitive = c1 * r1 * (pbest_pos - positions)   # pull toward personal best
    social    = c2 * r2 * (gbest_pos - positions)   # pull toward global best

    new_vel = w * velocities + cognitive + social
    velocities = np.clip(new_vel, -v_max, v_max)   # clamp velocity

    # Position
    new_pos = positions + velocities
    positions = np.clip(new_pos, bounds[0], bounds[1])

    # Evaluate fitness
    fitness = np.array([penalized_function(p[0], p[1]) for p in positions])

    # Update personal best
    for j in range(n_particles):
        if fitness[j] < pbest_cost[j]:
            pbest_cost[j] = fitness[j]
            pbest_pos[j]  = positions[j].copy()
    
    # Update global best
    best_idx = np.argmin(pbest_cost)
    if pbest_cost[best_idx] < gbest_cost:
        gbest_cost = pbest_cost[best_idx]
        gbest_pos  = pbest_pos[best_idx].copy()

    # Record history
    history_x.append(gbest_pos[0])
    history_y.append(gbest_pos[1])
    history_cost.append(gbest_cost)
    penalty_hist.append(abs(gbest_pos[0] - gbest_pos[1]))

    if (i+1) % 10 == 0:
        print(f"Iter {i+1:4d} | x={gbest_pos[0]:.4f}, y={gbest_pos[1]:.4f}")
    
    if penalized_function(gbest_pos[0], gbest_pos[1]) < 1e-6 and \
       abs(gbest_pos[0] - gbest_pos[1]) < 1e-4:
        print(f"Global minimum found at iteration {i+1}!")
        found = True
        break

if not found:
    print(f"Max iterations reached")

hx = np.array(history_x)
hy = np.array(history_y)

print(f"\nBest solution:")
print(f"   x      = {gbest_pos[0]:.6f}")
print(f"   y      = {gbest_pos[1]:.6f}")
print(f"   x - y  = {gbest_pos[0] - gbest_pos[1]:.6f}  (should be ~0)")
print(f"   f(x,y) = {function(gbest_pos[0], gbest_pos[1]):.6f}")

# Setup surface
x_m = np.linspace(-5.12, 5.12, 100)
y_m = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_m, y_m)
Z    = function(X, Y)
hz   = np.array([function(hx[i], hy[i]) for i in range(len(hx))])

# 3D path
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
ax.plot(hx, hy, hz, 'k', zorder=1, label='PSO path')
plt.plot(hx[0],  hy[0], hz[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(hx[-1], hy[-1], hz[-1], 'bo', markersize=5, label='End',   zorder=5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
plt.title('Particle Swarm', fontsize=16); plt.legend(); plt.show()

# Contour
x_line = np.linspace(-5.12, 5.12, 100)
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
plt.colorbar(label='f(x,y)')
plt.plot(x_line, x_line, 'm--', linewidth=2, label='Constraint x=y')
plt.plot(hx, hy, 'k', label='PSO path')
plt.plot(hx[0],  hy[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(hx[-1], hy[-1], 'bo', markersize=5, label='End',   zorder=5)
plt.xlim(-5.12, 5.12); plt.ylim(-5.12, 5.12)
plt.title('Particle Swarm')
plt.legend(); plt.show()

# # Convergence
# plt.plot(history_cost, color='blue')
# plt.xlabel('Iteration'); plt.ylabel('Best Cost')
# plt.title('PSO Convergence'); plt.grid(True); plt.show()

# # Constraint violation
# plt.plot(penalty_hist, color='orange')
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel('Iteration'); plt.ylabel('|x - y|')
# plt.title('Constraint Violation'); plt.grid(True); plt.show()