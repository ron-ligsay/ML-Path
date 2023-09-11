
# Introduction to Deep Learning with Keras
reference/source : Datacacmp Introduction to Deep Learning with Keras by Kevin Vecmanis

### When to use neural networks?
* dealing with unstructured data
* don't need e asily interpretable results
* you can benefit from a known architecture
Example: classify images of cats and dogs
* **images -> unstructured data**
* you don't care about why the network knows it's a cat or a dog
* you can benefit from cnn

### Coding keras
using functional or sequential API


## Surviving Meteor Strike
**compiling**
```model.compile()```

**training**
```model.fil(x,y,epochs=100)```

**predicting**
```preds = model.predict()```

**evaluating**
```model.evaluate(x,y)```

# Binary Classification
* **binary classification** - classification task with two possible outcomes

### Using sigmomid function
* **sigmoid function** - squashes numbers between 0 and 1
* **sigmoid activation function** - sigmoid function used to calculate the output of a neural network
* **sigmoid activation layer** - layer that applies sigmoid activation function to the input

neuron output -> sigmoid -> transformed output -> rounded output


**Exploring dollar bills**
You will practice building classification models in Keras with the Banknote Authentication dataset.

Your goal is to distinguish between real and fake dollar bills. In order to do this, the dataset comes with 4 features: variance,skewness,kurtosis and entropy. These features are calculated by applying mathematical operations over the dollar bill images. The labels are found in the dataframe's class column.


**Output layer**
using softmax to output probabilities
and each neuron represents a class or label

**Multi-class model**
instantiate a sequential model
add an input and hidden layer
add more hidden layers
add your output layer

**Preparing a dataset**
`from tensorflow.keras.utils import to_categorical`

turning response variable into labeled codes
df.response = pd.categorical(df.response)
df.response = df.response.cat.codes

turn response variable into one-hot response vector
y = to_categorical(df.response)

**one-hot encoding**
label encoding to one hot encoding
