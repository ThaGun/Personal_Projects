import numpy as np
import matplotlib.pyplot as plt
from Function import function

# Penalized function
def penalized_function(x, y, penalty_weight=100):
    return function(x, y) + penalty_weight * (x - y)**2

# CT parameters
n_bacteria = 30
n_chemotaxis = 50
n_swim = 10
n_reproduction = 4
n_elimination = 2
step_size = 0.01
elim_prob = 0.25
bounds = (-5.12, 5.12)

# Population intialization
population = np.random.uniform(bounds[0], bounds[1], size=(n_bacteria, 2))
#population[0] = [4.5, 3.5]
best_x, best_y = population[0]
best_cost = penalized_function(best_x, best_y)

# History
history_x = []
history_y = []
history_cost = []
penalty_hist = []
iteration = 0

# Nested loop
found = False
for rep in range(n_reproduction):               # Reproduction loop
    if found: break
    for elim in range(n_elimination):           # Elimination loop
        if found: break
        for chem in range(n_chemotaxis):        # Chemotaxis loop
            # Chemotaxis
            curr_size = len(population)
            fitness = np.array([penalized_function(b[0], b[1]) for b in population])

            for i in range(curr_size):
                curr_cost = penalized_function(population[i][0], population[i][1])

                # Tumble - pick random direction
                direction = np.random.uniform(-1, 1, size=2)
                direction = direction / np.linalg.norm(direction) # normalizing to unit vector
                
                # Swim - keep moving if improving
                for swim in range(n_swim):
                    new_pos = population[i] + step_size * direction
                    new_pos = np.clip(new_pos, bounds[0], bounds[1])
                    new_cost = penalized_function(new_pos[0], new_pos[1])

                    if new_cost < curr_cost:
                        # Better, keep swimmimg in same direction
                        population[i] = new_pos
                        curr_cost = new_cost
                    else:
                        # Worst, stop swimming, wait for next direction
                        break
                
                fitness[i] = curr_cost
            
            # Track best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_cost:
                best_cost = fitness[best_idx]
                best_x = population[best_idx][0]
                best_y = population[best_idx][1]
            
            # Record history
            history_x.append(best_x)
            history_y.append(best_y)
            history_cost.append(best_cost)
            penalty_hist.append(abs(best_x - best_y))
            iteration += 1

            if penalized_function(best_x, best_y) < 1e-6 and abs(best_x - best_y) < 1e-4:
                print(f"Global minimum found at iteration {iteration}!")
                found = True
                break
        
        # Elimination and dispersal
        for i in range(len(population)):
            if np.random.rand() < elim_prob:
                population[i] = np.random.uniform(bounds[0], bounds[1], size=2)
    
    # Reproduction
    fitness = np.array([penalized_function(b[0], b[1]) for b in population])
    
    curr_size = len(population)
    half = curr_size // 2

    # Sort by fitness
    sorted_idx = np.argsort(fitness)
    survivors = population[sorted_idx[:half]]   # best half

    new_population = np.vstack([survivors, survivors.copy()])
    population = new_population

    print(f"Reproduction {rep+1}/{n_reproduction} | x={best_x:.4f}, y={best_y:.4f} | ")

hx = np.array(history_x)
hy = np.array(history_y)
hcost = np.array(history_cost)
hpenalty = np.array(penalty_hist)

print(f"\nBest solution:")
print(f"   x      = {best_x:.6f}")
print(f"   y      = {best_y:.6f}")
print(f"   x - y  = {best_x - best_y:.6f}  (should be ~0)")
print(f"   f(x,y) = {function(best_x, best_y):.6f}")

# Setup surface
x_m = np.linspace(-5.12, 5.12, 100)
y_m = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_m, y_m)
Z = function(X, Y)
hz = np.array([function(hx[i], hy[i]) for i in range(len(hx))])

# 3D path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
ax.plot(hx, hy, hz, 'k', zorder=1, label='CT path')
plt.plot(hx[0],  hy[0], hz[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(hx[-1], hy[-1], hz[-1], 'bo', markersize=5, label='End',   zorder=5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
plt.title('Chemotaxis', fontsize=16); plt.legend(); plt.show()

# Contour
x_line = np.linspace(-5.12, 5.12, 100)
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
plt.colorbar(label='f(x,y)')
plt.plot(x_line, x_line, 'm--', linewidth=2, label='Constraint x=y')
plt.plot(hx, hy, 'k', label='CT path')
plt.plot(hx[0],  hy[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(hx[-1], hy[-1], 'bo', markersize=5, label='End',   zorder=5)
plt.xlim(-5.12, 5.12); plt.ylim(-5.12, 5.12)
plt.title('Chemotaxis')
plt.legend(); plt.show()

# # --- Convergence ---
# plt.plot(hcost, color='blue')
# plt.xlabel('Iteration'); plt.ylabel('Best Cost')
# plt.title('BFO Convergence'); plt.grid(True); plt.show()

# # --- Constraint violation ---
# plt.plot(hpenalty, color='orange')
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel('Iteration'); plt.ylabel('|x - y|')
# plt.title('Constraint Violation'); plt.grid(True); plt.show()