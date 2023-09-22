import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data generation (replace this with your data)
x = np.linspace(-5, 5, 100)
y = 2.0 * np.exp(-0.7 * ((x - 0.3) / 1.5) ** 2) + np.random.normal(0, 0.1, 100) # Example Gaussian curve

# Define a Gaussian function for fitting
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Fit the data to the Gaussian function
initial_guess = [1.0, 0.0, 1.0]  # Initial parameters for A, mu, and sigma
params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)

# Extract the fitted parameters
A_fit, mu_fit, sigma_fit = params

# Find the interpolated maximum value
interpolated_max_x = mu_fit  # The maximum occurs at the mean (mu)
interpolated_max_y = gaussian(interpolated_max_x, *params)

# Plot the original data and the fitted Gaussian curve
plt.plot(x, y, '.b', label='Original Data')
plt.plot(x, gaussian(x, *params), 'r', label='Fitted Gaussian')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Curve Fitting')
plt.legend()

# Display the interpolated maximum value
print(f"Interpolated Maximum Value (x): {interpolated_max_x}")
print(f"Interpolated Maximum Value (y): {interpolated_max_y}")

plt.show()
