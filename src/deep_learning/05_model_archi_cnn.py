# import COnv2D layer and Flatten from tensorflow keras layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(filters=32, 
                 input_shape=(28, 28, 1), 
                 kernel_size=3, 
                 activation='relu'))

# add another convolutional layer
model.add(Conv2D(8, kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(Flatten())

# End this multiclass model with 3 outputs and softmax
model.add(Dense(3, activation='softmax'))

# Pre-processing images for ResNet50
# import image from keras preprocessing
from tensorflow.keras.preprocessing import image
# import preprocess_input from keras applications resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input
# load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))
# turn it into an array
img = image.img_to_array(img)
# Exapnd the dimensions so that it's understood by our network:
# img.shape turns from (224, 224, 3) into (1, 224, 224, 3)
img = np.expand_dims(img, axis=0)
# pre-process the img in the same way training images were
img = preprocess_input(img)

# Using the ResNet50 model in Keras
# Import ResNet50 and decode_predictions from tensorflow.keras.apprications.resnet50
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
# Instantiate a ResNet50 model with imagenet weights
model = ResNet50(weights='imagenet')
# Predict with ResNet50 on your already processed img
preds = model.predict(img)
# Decode predictions and print it
print('Predicted:', decode_predictions(preds, top=3)[0])

# Excercises
# Classifying MNIST digit dataset
# Import the Conv2D and Flatten layers and instantiate model
from tensorflow.keras.layers import Conv2D  ,Flatten
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(32, kernel_size = 3, input_shape = (28, 28, 1), activation = 'relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(16, kernel_size = 3, activation = 'relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation = 'softmax'))

# Looking at convolutions
# Obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = Model(inputs = model.layers[0].input, outputs = first_layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# Plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[0,:,:,14], cmap = 'viridis')

# Do the same but for the 18th filter now
axs[1].matshow(activations[0,:,:,17], cmap = 'viridis')
plt.show()

# Hurrah! Each neuron filter of the first layer learned a different convolution. 
# The 15th filter (a.k.a convolutional mask) learned to detect horizontal traces in your digits. 
# On the other hand, filter 18th seems to be checking for vertical traces.

# Preparing your input  image

# Import image and preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)

# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])