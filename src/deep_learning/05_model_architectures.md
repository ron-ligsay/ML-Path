
## Tensors, Layers, and autoencoders

**What are tensors?**
tensors are the main data structures used in deep learning, inputs, outputs, and transformations in neural  networks are all represented using tensors and tensor multiplication
tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis).


### Autoencoders
are models that aim atproducing the same inputs as outputs

lower dimensional representation

**autoencoder use cases**
* dimensionality reduction
    * smaller dimensional space representation of our inputs
    * useful for data visualization
* denoising
    * if trained with clean data, irrelevant noise will be filtered out during reconstruction
* anomaly detection
    * a poor reconstruction will result when the model  is fed with unseen inputs


building a simple encoder


# Intro to CNNs

Example for object classification, car <br>
input (image of a car) > feature learning (convolution + relu > pooling > convolution + relu > pooling) > classfication (flatten > fully connected > softmax (bicycle, car, truck, van, ...)) 

##### Convolutional layers
convolutional layers perform feature learning, we then flatten the outputs into a vector and feed it to a fully connected layer for classification

building a cnn in keras

ResNet50 (50 layers) model that performs will in imagenet dataset, that can classify images into 1000 categories