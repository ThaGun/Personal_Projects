import numpy as np
import matplotlib.pyplot as plt
from Function import function

# Penalized function
def penalized_function(x, y, penalty_weight=1):
    return function(x, y) + penalty_weight * (x - y)**2

# adaptive penalty
def penalized(x, y, i):
        progress = i / max_iter
        weight = 1 + 990 * progress
        return function(x, y) + weight * (x - y)**2

# SA parameters
T_start = 10000
T_min = 1e-8
cooling_rate = 0.999
max_iter = 50000
#step_size = 0.5
bounds = (-5.12, 5.12)

# Initial parameters
# curr_x, curr_y = [2.0, 2.0]
curr_x, curr_y = [np.random.uniform(*bounds), np.random.uniform(*bounds)]

# curr_cost = penalized_function(curr_x, curr_y)
curr_cost = penalized(curr_x, curr_y, 0)

# Best solution tracker
best_x, best_y = curr_x, curr_y
best_cost = curr_cost

# History
T = T_start
history_cost = []
history_x = []
history_y = []
history_T = []
penalty_hist = []

# Loop
for i in range(max_iter):
    # Generate neighbor
    # Adaptive step - large when hot, small when cold
    step_size = max(5.0 * (T / T_start), 0.1)   
    new_x = curr_x + np.random.uniform(-step_size, step_size)
    new_y = curr_y + np.random.uniform(-step_size, step_size)

    new_x = np.clip(new_x, bounds[0], bounds[1])
    new_y = np.clip(new_y, bounds[0], bounds[1])

    # new_cost = penalized_function(new_x, new_y)
    new_cost = penalized(new_x, new_y, i)

    # Acceptance rule
    delta = new_cost - curr_cost
    probability = np.exp(-delta / T)

    if delta < 0 or np.random.rand() < probability:
        curr_x, curr_y = new_x, new_y
        curr_cost = new_cost
    
    # Update best
    if curr_cost < best_cost:
        best_x, best_y = curr_x, curr_y
        best_cost = curr_cost
    
    # Cooldown
    T = max(T * cooling_rate, T_min)

    # History
    history_cost.append(best_cost)
    history_x.append(best_x)
    history_y.append(best_y)
    history_T.append(T)
    penalty_hist.append(abs(best_x - best_y))

    # Conditioned stop
    if best_cost < 1e-6:
            print(f"Global minimum found at iteration {i+1}!")
            break
    
    # Progress
    if (i+1) % 10 == 0:
            print(f"Iter {i+1:5d} | T={T:.4f} | x={best_x:.4f}, y={best_y:.4f}")

hx = np.array(history_x)
hy = np.array(history_y)
hcost = np.array(history_cost)
hT = np.array(history_T)
hpenalty = np.array(penalty_hist)

print(f"\nBest solution:")
print(f"   x      = {best_x:.6f}")
print(f"   y      = {best_y:.6f}")
print(f"   x - y  = {best_x - best_y:.6f}  (should be ~0)")
print(f"   f(x,y) = {function(best_x, best_y):.6f}")

x_m = np.linspace(-5.12, 5.12, 100)
y_m = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_m, y_m)
Z = function(X, Y)
hz = np.array([function(hx[i], hy[i]) for i in range(len(hx))])

# 3D path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
ax.plot(hx, hy, hz, 'k', zorder=1, label='SA path')
ax.plot(hx[0],  hy[0], hz[0], 'ro', markersize=5, label='Start', zorder=5)
ax.plot(hx[-1], hy[-1], hz[-1], 'bo', markersize=5, label='End',   zorder=5)
ax.set_xlabel('x');  ax.set_ylabel('y');  ax.set_zlabel('f(x,y)')
plt.title('Simulated Annealing', fontsize=16)
plt.legend();  plt.show()

# Contour path
x_line = np.linspace(-5.12, 5.12, 100)
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
plt.colorbar(label='Rastrigin f(x,y)')
plt.plot(x_line, x_line, 'm--', linewidth=2.0, label='Constraint x=y')
plt.plot(hx, hy, 'k', label='SA path')
plt.plot(hx[0],  hy[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(hx[-1], hy[-1], 'bo', markersize=5, label='End',   zorder=5)
plt.xlim(-5.12, 5.12);  plt.ylim(-5.12, 5.12)
plt.title('Simulated Annealing')
plt.legend();  plt.show()

# # --- Convergence ---
# plt.plot(hcost, color='blue')
# plt.xlabel('Iteration');  plt.ylabel('Best Cost')
# plt.title('SA Convergence');  plt.grid(True);  plt.show()

# # --- Temperature decay ---
# plt.plot(hT, color='red')
# plt.xlabel('Iteration');  plt.ylabel('Temperature')
# plt.title('Cooling Schedule');  plt.grid(True);  plt.show()

# # --- Constraint violation ---
# plt.plot(hpenalty, color='orange')
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel('Iteration');  plt.ylabel('|x - y|')
# plt.title('Constraint Violation');  plt.grid(True);  plt.show()