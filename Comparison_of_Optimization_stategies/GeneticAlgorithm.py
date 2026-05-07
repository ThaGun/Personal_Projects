import numpy as np
import matplotlib.pylab as plt
from Function import function

# GA parameters
pop_size = 100
generations = 200
bounds = (-5.12, 5.12)
crossover_rate = 0.8
mutation_rate = 0.1
mutation_strength = 0.5
tournament_size = 3
elite_count = 2
alpha = 0.5

# Penalized function
def penalized_function(x, y, penalty_weight=10):
    return function(x, y) + penalty_weight * (x - y)**2

best_fitness_hist = []
x = []
y = []
z = []
# Initialize population
population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, 2))
population[0, :] = np.array([4.3, 3.5])

print(f"Initial params| x: {population[0, 0]} y: {population[0, 1]}")

for gen in range(generations):
    # Evaluation
    fitness = np.array([penalized_function(ind[0], ind[1]) for ind in population])

    best_fitness_hist.append(np.min(fitness))
    best_idx_a = np.argmin(fitness)
    x.append(population[best_idx_a][0])
    y.append(population[best_idx_a][1])
    best_value_a = penalized_function(population[best_idx_a][0], population[best_idx_a][1])
    z.append(best_value_a)

    # Parent selection
    selected = []
    pop_size = len(population)
    

    for _ in range(pop_size):
        # Random Competitors
        comp_idx = np.random.choice(pop_size, tournament_size, replace=False)
        comp_fitness = fitness[comp_idx]
        # Winners = lowest fitness
        win_idx = comp_idx[np.argmin(comp_fitness)]
        selected.append(population[win_idx])
    
    parents = np.array(selected)

    # Crossover
    children = []
    pop_size = len(parents)

    for i in range(0, pop_size, 2):
        p1 = parents[i]
        p2 = parents[(i+1) % pop_size]

        if np.random.rand() < crossover_rate:
            child1, child2 = [], []
            for g1, g2 in zip(p1, p2):
                d = abs(g1 - g2)
                lo = min(g1, g2) - alpha * d
                hi = max(g1, g2) + alpha * d
                child1.append(np.random.uniform(lo, hi))
                child2.append(np.random.uniform(lo, hi))
            children.extend([child1, child2])
        else:
            children.extend([p1.copy(), p2.copy()])
    
    children = np.array(children)

    # Mutate
    for i in range(len(children)):
        for j in range(len(children[i])):
            if np.random.rand() < mutation_rate:
                children[i][j] += np.random.normal(0, mutation_strength)
    
    children = np.clip(children, bounds[0], bounds[1])

    # Elitism
    elite_idcs = np.argsort(fitness)[:elite_count]  # Best from old
    worst_idcs = np.argsort(np.array([penalized_function(ind[0], ind[1]) 
                                      for ind in children]))[-elite_count:]
    
    for i, ei in enumerate(elite_idcs):
        children[worst_idcs[i]] = population[ei]
    
    population = children

    if (gen+1) % 1 == 0:
        best_idx = np.argmin(fitness)
        print(f"Gen {gen+1} | x={population[best_idx][0]:.4f}, y={population[best_idx][1]:.4f}")

    best_idx_a = np.argmin(fitness)
    
    

    if z[-1] < 1e-8:
        print(f"Global minimum found at Gen {gen+1} | f(x,y) = {z[-1]:.8f}")
        break


final_fitness = np.array([penalized_function(ind[0], ind[1]) for ind in population])
best_idx = np.argmin(final_fitness)
best_solution = population[best_idx]
best_value    = final_fitness[best_idx]

x = np.array(x)
y = np.array(y)
z = np.array(z)

print(f"\nBest solution found:")
print(f"   x = {best_solution[0]:.6f}")
print(f"   y = {best_solution[1]:.6f}")
print(f"   f(x,y) = {best_value:.6f}  (global min = 0.0)")

# 3D plot
x_m = np.linspace(-5.12, 5.12, 100)
y_m = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_m, y_m)
Z = function(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
ax.plot(x, y, z, 'k', zorder=1, label='GA path')
plt.plot(x[0],  y[0], z[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(x[-1], y[-1], z[-1], 'bo', markersize=5, label='End',   zorder=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title("Genetic algorithm", fontsize=16)
plt.legend()
plt.show()

x_line = np.linspace(-5.12, 5.12, 100)          # match your x-axis range

plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
plt.colorbar(label='Rastrigin f(x,y)')
plt.plot(x_line, x_line, 'm--', linewidth=2.0, label='Constraint x=y')
plt.plot(x, y, 'k', label='GA path')
plt.plot(x[0],  y[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(x[-1], y[-1], 'bo', markersize=5, label='End',   zorder=5)
plt.xlim(-5.12, 5.12);  plt.ylim(-5.12, 5.12)
plt.title("Genetic algorithm")
plt.legend();  plt.show()

# plt.plot(penalty_his, color='orange', linewidth=1.2)
# plt.axhline(0, color='k', linestyle='--', linewidth=1)
# plt.xlabel('Iteration')
# plt.ylabel('|x - y|')
# plt.title('Constraint Violation Over Iterations')
# plt.tight_layout()
# plt.show()