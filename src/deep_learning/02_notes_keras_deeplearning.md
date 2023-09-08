### Note: this code is from Datacamp Courses: Introduction to Deep Learning in Python
# Creating a Keras model

### Model building steps
* specify architecture
* compile
* fit
* predict

#### Specifying a model


#### Why do you need to compile your model
* specify the optimizer - how to update the weights or finds
    * many options and mathematically complex
    * "Adam" is usually a good choice
* loss function
    * "mean_squared_error" common for regression

#### Fitting a model
* applying backpropagation and gradient descent with your data to update the weights
* scaling data before fitting can ease optimization


### Classification 
* 'categorical_crossentropy' loss function
* similar to log loss: lower is better
* add metrics = ['accuracy'] to compile step for easy-to-understand diagnostics
* output layer has separate node for each possible outcome, and uses 'softmax' activation function

# Using Models
* save
* reload
* make predictions


# Model Optimization

### Why optimization is hard
* simultaneously optimizing 10000s of parameters with complex relationships
* updates may not improve model meaningfully
* updates too small (learning rate too low) or too large (learning rate too high)

### Stochastic Gradient Descent (SGD)


### The dying neuron problem
* if a large gradient changes the weights such that the neuron never fires again, the gradient will forever be zero from then on

### Vanishing Gradients
* occurs when many layers have very small slopes (e.g. due to being on flat part of tanh curve)
* in deep networks, updates to backprop were close to 0


# Model 
**Validation in deep learning**
* commonly use validation split rather than cross-validation
* deep learning widely used on large datasets
* single validation score is based on large amount of data, and is reliable

### Early Stopping

### Experimentation
* experiment with different architectures
* more layers
* fewer layers
* layers with more nodes
* layers with fewer nodes
* creating a great model requires experimentation