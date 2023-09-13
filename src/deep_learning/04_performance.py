# Improving Performance

# Learning Curves

# Store initial model weights
init_weights = model.get_weights()

# Lists for storing accuracies
train_accs = []
tests_accs = []

for train_size in train_sizes:
    # Split a fraction according to train_size
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=0)

    # Set Model  initial weights
    model.set_weights(init_weights)
    # Fit model on trainig set fraction
    model.fit(X_train_frac, y_train_frac, epochs=100, callbacks=[EarlyStopping(monitor='loss',patience=1)], verbose=False)
    # get the accuracy for this training set fraction
    train_acc = model.evaluate(X_train_frac, y_train_frac, verbose=False)[1]
    train_accs.append(train_acc)
    # get the accuracy for this test set fraction
    test_acc = model.evaluate(X_test, y_test, verbose=False)[1]
    tests_accs.append(test_acc)

    print("Done with size: ", train_size)


# Exercise
# Learning the digits
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (64,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation='softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))


# Exercise
# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_test, y_test, epochs = 60, validation_data = (X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


# Exercise
for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(model.get_weights())
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train, y_train)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)



# Activation Functions
# Set a random  seed
np.random.seed(1)

# Return a new model with the given activation
def get_model(act_function):
   model = Sequential()
   model.add(Dense(4, input_shape = (2,), activation = act_function))
   model.add(Dense(1, activation = 'sigmoid'))
   return model

# Activation functions to try out
activations = ['relu', 'sigmoid', 'tanh']

# Dictionary to store results
activation_results = {}
for funct in activations:
   model = get_model(act_function=funct)
   history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, verbose = 0)
   activation_results[funct] = history


# Comparing activation functions

# Extract val_loss history of each activation function
val_loss_per_funct = {k:v.history['val_loss'] for k,v in activation_results.items()}

# turn the dictionary into a pandas dataframe
val_loss_curves = pd.DataFrame(val_loss_per_funct)

# plot the curves
val_loss_curves.plot(title = 'Validation Loss per Activation Function')


# Exercise
# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model and store the history results
  h_callback = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 20, verbose = 0)
  activation_results[act] = h_callback


# Exercise
# Comapring activation function
# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()

# batch size and batch normalization

# batch size in Keras
# fitting an already build and compiled model
model.fit(X_train, y_train, epochs=100, batch_size=128)

# batch normalization in keras
# import BatchNormalization from keras layers
from tensorflow.keras.layers import BatchNormalization

# instantiate a sequential model
model = Sequential()
# add an input layer
model.add(Dense(3, input_shape=(2,), activation='relu'))
# add batch normalization layer
model.add(BatchNormalization())
# add an output layer
model.add(Dense(1, activation='sigmoid'))

# Exercise
# Cchanging batch sizes
# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs=5, batch_size=1)
print("\n The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])

# using the whole training set as batch size
model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs=5, batch_size=X_train.shape[0])
print("\n The accuracy when using the whole training set as batch-size was: ",
      model.evaluate(X_test, y_test)[1])

# Batch normalizing a familiar model, digital dataset
# Import batch normalization from keras layers
from tensorflow.keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)