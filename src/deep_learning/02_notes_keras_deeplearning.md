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