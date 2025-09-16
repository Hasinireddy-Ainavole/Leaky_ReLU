import numpy as np
import matplotlib.pyplot as plt

# Generate input values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Sigmoid activation function
# Range: (0, 1), suffers from vanishing gradients at extremes
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Tanh activation function  
# Range: (-1, 1), zero-centered but still has vanishing gradient issue
def tanh_func(x):
    return np.tanh(x)

# ReLU activation function
# Range: [0, inf), efficient but can cause dead neurons for negative inputs
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU activation function
# Range: (-inf, inf), solves dead neuron problem with small slope for negative values
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# Calculate outputs for each activation function
sig_output = sigmoid(x)
tanh_output = tanh_func(x)
relu_output = relu(x)
leaky_relu_output = leaky_relu(x)

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(x, sig_output, 'b-', label='Sigmoid', linewidth=2)
plt.plot(x, tanh_output, 'r-', label='Tanh', linewidth=2)
plt.plot(x, relu_output, 'g-', label='ReLU', linewidth=2)
plt.plot(x, leaky_relu_output, 'orange', label='Leaky ReLU', linewidth=2)

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions Comparison')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

print("Key differences:")
print("- Sigmoid/Tanh: smooth curves but vanishing gradients")
print("- ReLU: fast computation but dead neuron problem") 
print("- Leaky ReLU: fixes dead neurons while keeping efficiency")
