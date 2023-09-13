# Improving Your Model Performancce
reference: Chapter 3 of Introduction to Deep Learning with Keras from Datacamp

### Learning Curves
**loss curves** - plots of loss function over time

**accuracy curves** - plots of accuracy over time

### Activation Functions
a = sum of **inputs*weights** + 
activation function: f(a) = y

**activation functions**
* sigmoid - squashes numbers to range of 0 to 1
* tanh - squashes numbers to range of -1 to 
* relu - returns 0 if input is negative, otherwise returns input, range of 0 to infinity
* Leaky relu - returns 0.01 if input is negative, otherwise returns input, range of -infinity to infinity


**effects of activation functions**
* sigmoid - creates a boundary in a linearly separable dataset
* tanh - smooth close to rounded edges
* relu - creates a sharper boundary
* Leaky relu - creates a similar boundaries with rely but with more corners

Comaparing activation functions
it is important to set a seed so that the results are reproducible

note: the sigmoid and tanh both take values close to -1 for big negative numbers.


### Batch size and Batch normalization
The network is fed with 3 mini-batches, 1 epoch = 3 weight , 1 for each mini-batch

**mini-batches**
Advantages
* networks train faster (more wieght updaates in same amount of time)
* requires less memory (less data to store), can train on larger datasets
* noise can help the network avoid local minima, reach a lower error
Disadvantages
* more iterations need to be run
* need to be adjusted, we need to find a good batch size

**normalization (standardization)**
(data - mean) / standard 
standardized sets the mean to 0 and the standard deviation to 1