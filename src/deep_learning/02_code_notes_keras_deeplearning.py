
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

