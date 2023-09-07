

# Deep Learning

## Introduction to Deep Learning in Python  (Datacamp)

### Intro
3 layers:
* input layer
* hidden layer
* output layer

### Forward Propagation
- Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for a neural network in order from the input layer to the output layer.
- from input layer to output layer (left to right)
- each connectin has a weight
- from input layer to hidden layer
  - each node in hidden layer is a sum of the inputs multiplied by the weights
  - each node in hidden layer has an activation function


### Activation Functions
linear vs. non-linear activation functions

#### ReLu (Rectified Linear Activation)
- most common activation function
- max(0, x)

### Deeper Networks

#### Multiple Hidden Layers

#### Representing learning
* deep networks internally build representations of patters in the data
* partially replace the need for feature engineering
* Subsequent layers build increasingly sophisticated representations of raw data

#### Deep Learning
* modeler doesn't need to specify the interactions
* neural networks learn the relevant patterns in the data
* last layer has more complex features


### The Need for Optimization
#### A baseline neural network
* the more closer to the actual target, the better the model
* the loss function is the distance between the prediction and the target (error = predicted - actual)
* changing the weights and biases changes the loss function (to be more acurate)

#### Predictions with multiple points
* making accurate predictions gets harder with more points
* at any set of weights, there are many values of the error

#### Loss function
* aggregate errors in predictions from many data points into single number
* measure of model's predictive performance

##### Squared error loss function
* most common loss function
* sum of squared errors between predictions and actual values
* total squared error/loss = sum of (predicted - actual)^2
* mean squared error (MSE) = average of squared errors

Plotting the loss function
* x-axis: weight 1
* y-axis: weight 2
* z-axis: loss function

Loss function
* lower loss function means a better model
* goal: find the weights that give the lowest value for the loss function

Gradient Descent
* imagining you are in a pitch dark field
* want to find the lowest point
* feel the ground to see how it slopes
* take a small step downhill
* repeat until it is uphill in every 

Gradient Descent Steps
* start at random point
* until you are somewhere flat:
  * find the slope
  * take a step downhill
  