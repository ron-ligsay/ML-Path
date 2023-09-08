
# Model Specification
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

predeictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1]

# n_cols = number of predictors , number of nodes in the input later
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

