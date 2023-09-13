# Tensors, Layers, and autoencoders

# accessing the first layer of a Keras model
first_layer = model.layers[0]
# printing the layer, and its input, output and weights
print(first_layer.input)
print(first_layer.output)
print(first_layer.weights)

# Defining a rank 2 tensor (2 dimensions)
T2 = [[1,2,3],[4,5,6],[7,8,9]]

# Import Keras backend
import tensorflow.keras.backend as K
# Get the input and output tensors of a model layer
inp = model.layers[0].input
out = model.layers[0].output
# function that maps layer input to outputs
int_to_out = K.function([inp], [out])
# we pass and input and get the output we'd get in the first layer
print(inp_to_out([X_train]))


# Building a simple autoencoder

# instantiate a sequential model
autoencoder = Sequential()
# add a  hidden layer of 4 neurons and an input layer of 100
autoencoder.add(Dense(4, input_shape=(100,), activation="relu"))
# add an output layer of 100 neurons
autoencoder.add(Dense(100, activation="sigmoid"))
# compile your model 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# building a separate model to encode inputs
encoder = Sequential()
encoder.add(autoencoder.layers[0])
# Predicting returns the four hidden layer neuron outputs
encoder.predict(X_test)


# Exercise
# It's a flow of tensors
# Banknote Authentication
# Import tensorflow.keras backend
import tensorflow.keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.output[1]

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))


# Exercise
# Neural Separation
def plot():
   fig, ax = plt.subplots()
   plt.scatter(layer_output[:, 0], layer_output[:, 1],c = y_test,edgecolors='none')
   plt.title('Epoch: {}, Test Accuracy: {:3.1f} %'.format(i+1, test_accuracy * 100.0))
   plt.show()

for i in range(0, 21):
    # Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size = 16, epochs = 1, verbose = 0)
    if i%4==0:
      # Get the output of the first layer
      layer_output = inp_to_out([X_test])[0]
      
      # Evaluate model accuracy for this epoch
      test_accuracy = model.evaluate(X_test, y_test)[1] 
      
      # Plot 1st vs 2nd neuron output
      plot()

# Building an autoencoder
# Autoencoders have several interesting applications like anomaly detection or image denoising. 
# They aim at producing an output identical to its inputs. 
# The input will be compressed into a lower dimensional space, encoded. 
# The model then learns to decode it back to its original form.

# You will encode and decode the MNIST dataset of handwritten digits, 
# the hidden layer will encode a 32-dimensional representation of the image, 
# which originally consists of 784 pixels (28 x 28). The autoencoder will essentially 
# learn to turn the 784 pixels original image into a compressed 32 pixels image and 
# learn how to use that encoded representation to bring back the original 784 pixels image.


# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu"))

# Add an output layer with as many neurons as the orginal image pixels
autoencoder.add(Dense(784, activation = "sigmoid"))

# Compile your model with adadelta
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Summarize your model structure
autoencoder.summary()

# Okay, you have just built an autoencoder model. 
# Let's see how it handles a more challenging task.

# First, you will build a model that encodes images, 
# and you will check how different digits are represented with show_encodings(). 
# To build the encoder you will make use of your autoencoder, 
# that has already being trained. You will just use the first half of the network, 
# which contains the input and the bottleneck output. 
# That way, you will obtain a 32 number output which represents 
# the encoded version of the input image.

# Then, you will apply your autoencoder to noisy images from MNIST, 
# it should be able to clean the noisy artifacts.

# X_test_noise is loaded in your workspace. 
# The digits in this noisy dataset look like this:

# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)