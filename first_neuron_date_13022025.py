def neuron(inputs, weights, bias=0.0, activation=lambda x: x):
    z = sum(x * w for x, w in zip(inputs, weights)) + bias
    return activation(z) #Calculates the neuron's output with an optional activation function.

# Example input
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

# Neuron output using the identity activation function
output = neuron(inputs, weights, bias)
print("Neuron output:", output)

