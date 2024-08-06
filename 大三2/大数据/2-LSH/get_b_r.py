import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def error_function(params, threshold=0.8):
    r, b = params
    error = 0
    for t in np.linspace(0, 1, num=1000):
        approx_value = 1 - (1 - t**r)**b
        true_value = 1 if t >= threshold else 0
        error += (true_value - approx_value)**2
    return error
def step_function(t, r, b):
    return 1 - (1 - t**r)**b

initial_params = [10, 10]
result = minimize(error_function, initial_params, method='CG')
optimal_params = result.x
print("Optimal parameters (r, b):", optimal_params)

r = optimal_params[0]
b = optimal_params[1]
t_values = np.linspace(0, 1, num=1000)
y_values = step_function(t_values, r, b)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(t_values, y_values, label=f'r={r}, b={b}')
plt.axvline(x=0.8, color='red', linestyle='--', label='Threshold = 0.8')
plt.xlabel('t')
plt.ylabel('Function Value')
plt.title('1 - (1 - t^r)^b')
plt.legend()
plt.grid(True)
plt.show()