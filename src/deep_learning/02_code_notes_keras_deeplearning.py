# This codes are from Datacamp courses

# Model Specification
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# n_cols = number of predictors , number of nodes in the input later
predeictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1]

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# Compiling a model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting a model
model.fit(predictors, target)


# Classification
data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).values
target = to_categorical(data.shot_result)
n_cols = predictors.shape[1]

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)

# Saving, Reloading, and using your model
from tensoflow.keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:,1]

# Verifying model structure
my_model.summary()


# Stochastic Gradient Descent (SGD)
def get_new_momde(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)

lr_to_test = [.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    model.fit(predictors, target)


# Model Validation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target, validation_split=0.3)


# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

# with patience, the model will not stop until the validation loss has not improved for 2 epochs
early_stopping_monitor = EarlyStopping(patience=2)

# callbackcs is a list of callbacks to apply during training, it will be called at the end of each epoch
model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])