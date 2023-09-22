import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Generate some sample 3D data
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 2) * 10  # 2D input features
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, n_samples)

# Define the grid for interpolation
x1_grid = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_grid = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Perform 3D local regression (interpolation)
smoothed_y = griddata(X, y, (X1_grid, X2_grid), method='cubic')

# Create a 3D scatter plot of the original data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Original Data')

# Create a 3D surface plot of the smoothed data
ax.plot_surface(X1_grid, X2_grid, smoothed_y, cmap='viridis', alpha=0.7)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.title('3D Local Regression (Interpolation)')
plt.show()
