import numpy as np
import matplotlib.pyplot as plt

# Rastrigin function with 2 varibles
def function(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

# # Function visualization
# x, y = np.linspace(-5.12, 5.12, 200), np.linspace(-5.12, 5.12, 200)
# X, Y = np.meshgrid(x, y)
# Z = function(X, Y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x,y)')
# plt.show()
