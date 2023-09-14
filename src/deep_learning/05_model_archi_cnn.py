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

