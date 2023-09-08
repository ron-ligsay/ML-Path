
# Forward propagation code
import numpy as np
input_data = np.array([2,3])
weights = {
    'node_0': np.array([1,1]),
    'node_1': np.array([-1,1]),
    'output': np.array([2,-1])
    }

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)

# Activation function
import numpy as np
input_data = np.array([-1,2])
weights = {
    'node_0': np.array([-1,2]),
    'node_1': np.array([1,5]),
    'output': np.array([2,-1])
    }

node_0_value = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_value)
node_1_value = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_value)

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

hidden_layer_outputs = np.array([node_0_output, node_1_output])
model_output = (hidden_layer_outputs * weights['output']).sum()
print(model_output)

# Code to calculate slopes and update weights
weights = np.array([1,2])
input_data = np.array([3,4])
target = 6
learning_rate = 0.01
preds = (weights * input_data).sum()
error = preds - target
print(error)

# Slope calculation
gradient = 2 * input_data * error
print(gradient)

# Update weights
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error_updated)

# Test
print("initial weight: " , weights , ", updated: " , weights_updated)

