import numpy as np
import matplotlib.pyplot as plt
import time
from Function import function

def penalized_function(x, y, penalty_weight=1):
    return function(x, y) + penalty_weight * (x - y)**2

# Iterations setup

i_iter = 0
max_iter = 2000
S_iter = 0
max_S_iter = 2000
US_iter = 0
max_US_iter = 2000

# Initial values
x0 = np.random.uniform(-5.12, 5.12)
y0 = np.random.uniform(-5.12, 5.12)

param = [x0, y0]

# Gaussian vector with average of 0 and dispersion of 1
gVector = np.random.normal(0, 1, size=(max_iter, 1)).reshape(-1)
idx = np.random.randint(0, max_iter)
rand_value = gVector[idx]

print(f"Average: {np.average(gVector)}")
print(f"Deviation: {np.std(gVector)}")

# Learning rate/ Step size
deltaP = [0.001, 0.001]
path = []
func_his = []
penalty_his = []
# Start search
start_time = time.time()

while i_iter < max_iter:
    cand_guess = []

    prev_value = penalized_function(param[0], param[1])

    for i in range(0, len(param)):
        # next guess from old value
        next_guess = param[i] + (deltaP[i] * rand_value)
        cand_guess.append(next_guess)
        # Next random value for next canditate
        idx = np.random.randint(0, max_iter)
        rand_value = gVector[idx-1]
    
    # Constraint check
    if cand_guess[0] < -5.12 or cand_guess[0] > 5.12:
        print("x limit reached")
        cand_guess = param
        continue

    if cand_guess[1] < -5.12 or cand_guess[1] > 5.12:
        print("y limit reached")
        cand_guess = param
        continue
    
    curr_value = penalized_function(cand_guess[0], cand_guess[1])

    # Check for function value minimzation
    epsilon = prev_value - curr_value

    # Moving to next iteration
    if prev_value > curr_value:
        i_iter += 1
        S_iter += 1
        US_iter = 0
        param = cand_guess
        print(f"\nIteration: {i_iter}")
        print(f"Successful Iteration: {S_iter}")
        print(f"x: {param[0]}, y: {param[1]}")
        # Step size increase
        if S_iter % 10 == 0:
            deltaP[0] = deltaP[0] + (deltaP[0] * 0.2)
            deltaP[1] = deltaP[1] + (deltaP[1] * 0.2)
            print(f"Step size increased: {deltaP[0]}")
    else:
        i_iter += 1
        US_iter += 1
        S_iter = 0
        param = param
        print(f"\nIteration: {i_iter}")
        print(f"Unsuccessful Iteration: {US_iter}")
        print(f"x: {param[0]}, y: {param[1]}")
        # Step size decrese
        if US_iter % 10 == 0:
            deltaP[0] = deltaP[0] - (deltaP[0] * 0.2)
            deltaP[1] = deltaP[1] - (deltaP[1] * 0.2)
            print(f"Step size decreased: {deltaP[0]}")
    
    path.append(param)
    f_func = penalized_function(param[0], param[1])
    func_his.append(f_func)
    penalty_his.append(abs(param[0] - param[1]))

    if i_iter == max_iter:
        print("Stopping search...")
        break

end_time = time.time()

func_min = penalized_function(param[0], param[1])
print(f"x: {param[0]}, y: {param[1]}, Function: {func_min}")

path = np.array(path)
x = path[:,0]
y = path[:,1]
z = np.array(func_his)

# 3D plot
x_m = np.linspace(-5.12, 5.12, 100)
y_m = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_m, y_m)
Z = function(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
ax.plot(x, y, z, 'k', zorder=1, label='GD path')
ax.plot(x[0],  y[0], z[0],  'ro', markersize=5, label='Start', zorder=5)
ax.plot(x[-1], y[-1], z[-1], 'bo', markersize=5, label='End',   zorder=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title("Random search", fontsize=16)
plt.legend()
plt.show()

x_line = np.linspace(-5.12, 5.12, 100)          # match your x-axis range
y_line = x_line
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
plt.colorbar(label='Rastrigin f(x,y)')
plt.plot(x_line, y_line, 'm--', linewidth=2.0, label='Constraint')
plt.plot(x, y, 'k', label="RS path")
plt.plot(x[0],  y[0],  'ro', markersize=5, label='Start', zorder=5)
plt.plot(x[-1], y[-1], 'bo', markersize=5, label='End',   zorder=5)
# plt.plot(0, 0, 'ro', markersize=5, label='Global Minimum (0,0)', zorder=5)
plt.xlim(-5.12, 5.12)
plt.ylim(-5.12, 5.12)
plt.title("Random search")
plt.legend()
plt.show()

plt.plot(penalty_his, color='orange', linewidth=1.2)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('|x - y|')
plt.title('Constraint Violation Over Iterations')
plt.tight_layout()
plt.show()