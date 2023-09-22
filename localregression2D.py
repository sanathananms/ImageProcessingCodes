import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + np.cos(X) + np.random.normal(0, 0.1, 100)

# Define the bandwidth (span) for the local regression
bandwidth = 0.3 

# Perform 2D local regression
lowess = sm.nonparametric.lowess(y, X, frac=bandwidth)

# Extract the smoothed values
smoothed_y = lowess[:, 1]

# Plot the original data and the smoothed curve
plt.scatter(X, y, label='Original Data', color='b', alpha=0.5)
plt.plot(X, smoothed_y, label='Smoothed Curve', color='r')
plt.legend()
plt.title('2D Local Regression (LOESS)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
