### Note: this code is from Datacamp Courses: Introduction to Deep Learning in Python

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

**Loss function**
* lower loss function means a better model
* goal: find the weights that give the lowest value for the loss function

**Gradient Descent**
* imagining you are in a pitch dark field
* want to find the lowest point
* feel the ground to see how it slopes
* take a small step downhill
* repeat until it is uphill in every 

**Gradient Descent Steps**
* start at random point
* until you are somewhere flat:
  * find the slope
  * take a step downhill

* if the slope is positive
  * going opposite the slope means going  or to lower numbers
  * subtract the slope from the current 
  * too big a step might lead us astry
* solution: learning rate
  * update each weight by subtracting learning rate * slope
  * learning rate: controls how large a step we take downhill during gradient descent
  * too small: takes too long
  * too large: might not converge

**Slope calculation example**
* slope of loss function at a point = 2 * (predicted value - actual value) * input value

* to calculate the  slope for a weight, need to multiply:
  * slope of the loss function w.r.t value at the node we feed into
  * the value of the node that feeds into our weight
  * slope of the activation function w.r.t value we feed into (exisits only for nodes in hidden layers and output layers)

* weight * (slope of loss function w.r.t value at the node we feed into) predicted - actual * (slope of activation function w.r.t value we feed into) input data

* if learning rate is 0.01, the new weight would be
* weight - 0.01 * slope = new_weight (ex: 2 - 0.01 (-24) = 2.24)

note: w.r.t = with respect to

### Backpropagation
* allows gradient descent to update all weights in neural network (by getting gradients for all weights)
* comes from chain rule of calculus
* important to understand the process, but you will generally use a library that implements this

##### Backpropagation process
* tyring to estimate the slope of the loss function w.r.t each weight
* do forward propagation to calculate predictions and errors

* go back one layer at a time
* gradient for weight is product of:
  1. Node value feeding into that weight
  2. Slope of loss function w.r.t node it feeds into
  3. Slope of activation function at the node it feeds into

* Need to also keep track of the  slopes of the loss funciton w.r.t nonde values 
* slope of node values are the sum of the slopes for all weights that come out of them


Note:
* each time you generate predictions using forward propagation, you update the weights using backpropagation
* backpropagation iteration = forwardpropagation iteration

Q&A:
If your predictions were all exactly right, and your erros were all exactly 0, the slop of the loss fucntion with respect to your predictions would also be 0. In that circumstance, which of the following statemments would be correct?
* The updates to all weights in the network would also be 0
* The updates to all weights in the network would be dependent on the activation functions
* The updates to all weights in the network would be dependent on the slope of the activation functions

Answer: The updates to all weights in the network would also be 0
In this situation, the updates to all weights in the network wouldd indeed also be 0.

***Backpropagation***

Calculating slopes associated with any weight
* gradients for weight is product of
  1. Node value feeing into that weight
  2. Slope of activation funciton for the node being fed into
  3. Slope of loss function w.r.t output node

***Backpropagation recap***
* start at some random set of weights
* use forward propagation to make a prediction
* use backward propagation to calculate the slope of the loss function w.r.t each weight
* multiply that slope by the learning rate, and subtract from the current weights
* keep going with that cycle until we get to a flat part

#### Stochaastic Gradient descent
* it is common to calculate slops on only a subset of the data (a batch)
* use a different batch of data to calculate the next update
* start over from the beginning once all data is used
* each time through the training data is called an epoch
* when slopes are calculated on one batch at a time: stochasti gradient descent