import pandas as pd
from tensorflow import keras
from kerastuner import RandomSearch  # Or Hyperband, BayesianOptimization

def find_best_hyperparameters_and_plot(X_train, y_train, hyperparameter_space):
    """
    Finds the best hyperparameters for a TensorFlow Sequential model and creates a graph of results.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        hyperparameter_space (dict): Hyperparameter search space for Keras Tuner.

    Returns:
        tuple: (best_model, results)
            - best_model: The best fitted model with optimal hyperparameters.
            - results: Dictionary containing results for each hyperparameter trial.
    """

    def build_model(hp):
        model = keras.Sequential()
        # Define your model architecture here using hyperparameters from hp
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                      loss='categorical_crossentropy',  # Adjust loss as needed
                      metrics=['accuracy'])  # Adjust metrics as needed
        return model

    # Create RandomSearch tuner (adjust as needed)
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',  # Adjust objective as needed
        max_trials=10,  # Number of hyperparameter combinations to try
        directory='my_dir',  # Optional directory to store results
        project_name='hyperparameter_tuning'  # Optional project name
    )

    # Perform hyperparameter tuning
    tuner.search(X_train, y_train, epochs=5, validation_split=0.2)  # Adjust epochs and validation split

    # Get best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Get results as a pandas DataFrame
    results_df = pd.DataFrame(tuner.results_summary(num_trials=1))

    # Plot results for a chosen metric (e.g., val_accuracy)
    plt.figure()
    plt.plot(results_df["trial_id"], results_df["val_accuracy"])  # Adjust metric as needed
    plt.xlabel("Trial ID")
    plt.ylabel("Validation Accuracy")
    plt.title("Hyperparameter Tuning Results")
    plt.show()

    return best_model, results_df


# From tensorflow.org/tutorials/keras/keras_tuner
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# Instantiate the tuner and perform hypertuning
# Instantiate the tuner to perform the hypertuning. The Keras Tuner has four tuners available - RandomSearch, Hyperband, BayesianOptimization, and Sklearn. In this tutorial, you use the Hyperband tuner.

# To instantiate the Hyperband tuner, you must specify the hypermodel, the objective to optimize and the maximum number of epochs to train (max_epochs).

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# result:
# Trial 30 Complete [00h 00m 41s]
# val_accuracy: 0.8550833463668823

# Best val_accuracy So Far: 0.8900833129882812
# Total elapsed time: 00h 08m 43s

# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is 224 and the optimal learning rate for the optimizer
# is 0.001.


class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.input_shape))

        # Tune the number of hidden layers
        num_layers = hp.Int('num_layers', 2, 5)  # Allow 2-5 hidden layers
        for i in range(num_layers):
            model.add(keras.layers.Dense(
                units=hp.Int('units_' + str(i), 32, 128, step=32),
                activation='relu'
            ))
        model.add(keras.layers.Dense(10, activation='softmax'))  # Output layer

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
tuner = RandomSearch(
    hypermodel=MyHyperModel(input_shape=(28, 28, 1)),  # Example input shape
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='tuning_layers'
)

tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]