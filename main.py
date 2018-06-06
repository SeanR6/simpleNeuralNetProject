# based off of https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
import numpy as np

# input array
X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])

# output
y = np.array([[1], [1], [0]])


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)


# variable initialization
reps = 5000
learn_rate = 0.1
input_layer_neurons = X.shape[1]  # number of features in data set
hidden_layer_neurons = 6
output_neurons = 1

# weight and bias initialization
weight_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weight_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

for i in range(reps):
    # forward prop
    hidden_layer_input1 = np.dot(X, weight_hidden)  # dot products the weights with the input
    hidden_layer_input = hidden_layer_input1 + bias_hidden  # applies the bias to the inputs
    hidden_layer_activations = sigmoid(hidden_layer_input)  # normalizes data using sigmoid
    output_layer_input1 = np.dot(hidden_layer_activations, weight_output)  # repeats previous steps for the hidden ->out
    output_layer_input = output_layer_input1 + bias_output
    output = sigmoid(output_layer_input)

    # backwards prop
    E = y - output  # error
    slope_output_layer = sigmoid_prime(output)  # gets slope allows us to get our delta factors
    slope_hidden_layer = sigmoid_prime(hidden_layer_activations)
    delta_output = E * slope_output_layer  # gets the change of output from e and the slope
    # gets error of the hidden layer adjusted for the delta output
    E_at_hidden_layer = delta_output.dot(weight_output.T)
    delta_hidden_layer = E_at_hidden_layer * slope_hidden_layer  # gets change of hidden layer
    weight_output += hidden_layer_activations.T.dot(delta_output) * learn_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learn_rate
    weight_hidden += X.T.dot(delta_hidden_layer) * learn_rate
    bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learn_rate

    if i % 500 == 0:
        print(output)

print(output)
