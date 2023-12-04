# HW5 Problem 1

import numpy as np
from scipy.optimize import minimize

# Constants and data given
T = 20  # Temperature in °C
a1_water = 8.07131
a2_water = 1730.63
a3_water = 233.426
a1_dioxane = 7.43155
a2_dioxane = 1554.679
a3_dioxane = 240.337
x1_data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
p_data = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

# Calculate saturation pressures using the Antoine equation
p_sat_water = 10 ** (a1_water - (a2_water / (T + a3_water)))
p_sat_dioxane = 10 ** (a1_dioxane - (a2_dioxane / (T + a3_dioxane)))

# Function to calculate the pressure
def calculate_pressure(x1, A12, A21, p1_sat, p2_sat):
    x2 = 1 - x1
    p_calc = (x1 * np.exp(A12 * (A21*x2 / (A12*x1 + A21*x2))**2) * p1_sat) + \
             (x2 * np.exp(A21 * (A12*x1 / (A12*x1 + A21*x2))**2) * p2_sat)
    return p_calc

# Least squares function
def least_squares(parameters):
    A12, A21 = parameters
    p_calculated = np.array([calculate_pressure(x1, A12, A21, p_sat_water, p_sat_dioxane) for x1 in x1_data])
    return np.sum((p_data - p_calculated) ** 2)

# Optimization
initial_guess = [1, 1]
result = minimize(least_squares, initial_guess, method='Nelder-Mead')
A12_opt, A21_opt = result.x

# Calculate the predicted pressures with optimized parameters
p_predicted = np.array([calculate_pressure(x1, A12_opt, A21_opt, p_sat_water, p_sat_dioxane) for x1 in x1_data])

# Calculate R-squared
ss_res = np.sum((p_data - p_predicted) ** 2)
ss_tot = np.sum((p_data - np.mean(p_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Output
A12_opt, A21_opt, r_squared, p_predicted

print("Optimized A12:", A12_opt)
print("Optimized A21:", A21_opt)
print("R-squared value:", r_squared)
print("Predicted pressures:", p_predicted)
