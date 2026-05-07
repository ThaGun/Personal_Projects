import numpy as np
import matplotlib.pyplot as plt
from Function import function

def gradient_derivatives(params):
    # parameter 1: x
    dx = 2 * params[0] + 20 * np.pi * np.sin(2 * np.pi * params[0])
    # parameter 2: y
    dy = 2 * params[1] + 20 * np.pi * np.sin(2 * np.pi * params[1])
    return [dx, dy]

def project(x, y):
    # Project offset
    offset = (x - y) / 2
    return x - offset, y + offset

def penalty_loss(x, y, lam):
    return function(x, y) + lam * (x + y - 2)**2

def penalty_gradient(x, y, lam):
    dfdx = 2*x + 20*np.pi*np.sin(2*np.pi*x) + 2*lam*(x - y)
    dfdy = 2*y + 20*np.pi*np.sin(2*np.pi*y) + 2*lam*(x - y ) * (-1)
    return [dfdx, dfdy]

def lagrange_gradient(x, y, mu, lam):
    deriv = gradient_derivatives([x, y])
    constraint = x - y
    deriv[0] += mu + lam*constraint          # d/dx
    deriv[1] += -(mu + lam*constraint)       # d/dy → negative sign
    return deriv

def gradient_descent(params, learning_rate, iterations, lam=100):
    # # 3D plot initialise
    # x, y = np.linspace(-5.12, 5.12, 200), np.linspace(-5.12, 5.12, 200)
    # X, Y = np.meshgrid(x, y)
    # Z = function(X, Y)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # Initial values for optimization
    current_value = [params[0], params[1], function(params[0], params[1])]
    x_pos = [current_value[0]]
    y_pos = [current_value[1]]
    z_pos = [current_value[2]]
    penalty_his = []
    mu = 0.0

    for i in range(iterations):
        # Learning rate decay
        lr = learning_rate / (1 + 0.001 * i)
        # Gradient derivatives
        d_params = penalty_gradient(current_value[0], current_value[1], lam)
        #d_params = gradient_derivatives([current_value[0], current_value[1]])
        #d_params = lagrange_gradient(current_value[0], current_value[1], mu, lam)
        # Gradient clip
        clip = 5.12
        d_params[0] = np.clip(d_params[0], -clip, clip)
        d_params[1] = np.clip(d_params[1], -clip, clip)
        # Gradient descent with derivatives
        X_new = current_value[0] - lr * d_params[0]
        Y_new = current_value[1] - lr * d_params[1]
        # Lagrange multiplier update
        mu = mu + lam*(X_new - Y_new)
        # Constraint check with Project method
        #X_new, Y_new = project(X_new, Y_new)
        # Convergnce check
        if abs(X_new - current_value[0]) < 1e-6 and abs(Y_new - current_value[1]) < 1e-6:
            print(f"Converged at iteration {i}")
            break
        # Optimized value update
        current_value = [X_new, Y_new, function(X_new, Y_new)]
        # History for plotting
        x_pos.append(current_value[0])
        y_pos.append(current_value[1])
        z_pos.append(current_value[2])
        penalty_his.append(abs(current_value[0]-current_value[1]))

        # # 3D plot
        # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, zorder=0)
        # ax.plot(current_value[0], current_value[1], current_value[2], 'r.-', zorder=1)
        # plt.pause(0.001)
        # ax.clear()

        if i % 10 == 0:
            print(f"Iteration: {i}, x: {current_value[0]}, y: {current_value[1]}")
    
    return x_pos, y_pos, z_pos, penalty_his


if __name__ == '__main__':
    # Initial position/values
    params = [np.random.uniform(-5.12, 5.12), np.random.uniform(-5.12, 5.12)]
    print(params)

    # Learning rate of the gradient descent
    learning_rate = 0.001

    # No. of iterations
    iterations = 2000

    # Gradient descent
    # Random 50
    # best_result = None
    # best_final_z = float('inf')

    # for _ in range(50):  # Try 50 random starting points
    #     params = [np.random.uniform(-5.12, 5.12), 
    #               np.random.uniform(-5.12, 5.12)]
    #     x, y, z = gradient_descent(params, learning_rate, iterations)
        
    #     if z[-1] < best_final_z:
    #         best_final_z = z[-1]
    #         best_result = (x, y, z)
    #         best_params = params

    # print(f"Best final: f={best_final_z:.6f}")
    # x, y, z = best_result
    x, y, z, penalty_his = gradient_descent(params, learning_rate, iterations)
    print(f"Final values: x: {x[-1]}, y: {y[-1]}")

    # 3D plot
    x_m = np.linspace(-5.12, 5.12, 100)
    y_m = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x_m, y_m)
    Z = function(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=0)
    ax.plot(x, y, z, 'k', zorder=1, label='GD path')
    ax.plot(x[0],  y[0], z[0], 'ro', markersize=5, label='Start', zorder=5)
    ax.plot(x[-1], y[-1], z[-1], 'bo', markersize=5, label='End',   zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.title("Gradient Descent", fontsize=16)
    plt.legend()
    plt.show()

    x_line = np.linspace(-5.12, 5.12, 100)          # match your x-axis range
    y_line = x_line
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8, linewidth=0.4)
    plt.colorbar(label='Rastrigin f(x,y)')
    plt.plot(x_line, y_line, 'm--', linewidth=2.0, label='Constraint')
    plt.plot(x, y, 'k', label="GD path")
    plt.plot(x[0],  y[0],  'ro', markersize=5, label='Start', zorder=5)
    plt.plot(x[-1], y[-1], 'bo', markersize=5, label='End',   zorder=5)
    # plt.plot(0, 0, 'ro', markersize=5, label='Global Minimum (0,0)', zorder=5)
    plt.xlim(-5.12, 5.12)
    plt.ylim(-5.12, 5.12)
    plt.title("Gradient Descent")
    plt.legend()
    plt.show()

    plt.plot(penalty_his, color='orange', linewidth=1.2)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('|x - y|')
    plt.title('Constraint Violation Over Iterations')
    plt.tight_layout()
    plt.show()