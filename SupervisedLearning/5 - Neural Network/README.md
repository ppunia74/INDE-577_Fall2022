# Neural Network (Multilayer Perceptron Learning Algorithm)

This sub-repository mainly focuses on using Multilayer Perceptron Learning Algorithm on images to predict the labels.

Contents of **Neural Network**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Neural%20Network/Image): folder containing images used in README
* [MLP_MNIST.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Neural%20Network/MLP_MNIST.ipynb): Jupyter notebook file containing
  * a. Building Multilayer Perceptron Learning algorithm from scratch
  * b. Performing multilayer perceptron algorithm using MNIST Dataset to classify the handwritten digits
* [MLP.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Neural%20Network/MLP.ipynb): Jupyter notebook file containing
  * a. Introduction of the perceptron algorithm
  * b. Building Multilayer Perceptron Learning algorithm from scratch
  * c. Performing the multilayer perceptron algorithm using Fashion MNIST dataset to classify fashion categories
  * d. Increase the number of nodes in the hidden layers and compare
  * e. Add ReLU function as another activation function and compare

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Neural%20Network/Image/MLP_2.png)

### A Short Summary

# Multilayer Perceptron Learning Algorithm

Multilayer Perceptron Learning algorithm is an extension of the Single layer perceptron. Different from the perceptron, there are more than one layers of neurons. To be more specific, there could be one or more layers between the input and output layer, namely the 'hidden layer'. Each nodes in the input layer can be connected to each nodes in the hidden layer, and pass the input value with a certain weight (similar rules as perceptron). Each nodes in the hidden layer would then do the calculation based on all the inputs, including a nonlinear activation. The result will be further pass on the next hidden layer with the same rule, until it gets to the output layer.

### Activation Function

* Sigmoid Function:

<img src="https://latex.codecogs.com/svg.image?sigma(z)=\frac&space;{1}{1&plus;e^{-z}}" title="sigma(z)=\frac {1}{1+e^{-z}}" />

* Rectified Linear Units (ReLU)

<img src="https://latex.codecogs.com/svg.image?R(z)=&space;max(0,z)" title="R(z)= max(0,z)" />

### Loss Function

In our algorithm, we use the Mean Square Error:

<img src="https://latex.codecogs.com/svg.image?C(w,b;x,y)=\frac{1}{2}&space;\sum_{i=1}^n(a_i^{l-1}-y_i)^2" title="C(w,b;x,y)=\frac{1}{2} \sum_{i=1}^n(a_i^{l-1}-y_i)^2" />

### Output Error

The output error is

<img src="https://latex.codecogs.com/svg.image?\delta^{l-1}=\triangledown_{a^{l-1}}C&space;\otimes&space;\sigma'(z^{l-1})" title="\delta^{l-1}=\triangledown_{a^{l-1}}C \otimes \sigma'(z^{l-1})" />

### Neuron Error

According to the output error, for <img src="https://latex.codecogs.com/svg.image?l=L-2,...,1" title="l=L-2,...,1" />, the neuron error is

<img src="https://latex.codecogs.com/svg.image?\delta^{l}=\left&space;(&space;(w^{l&plus;1})^T&space;a^{l&plus;1}&space;\right&space;)&space;\otimes&space;\sigma'(z^{l})" title="\delta^{l}=\left ( (w^{l+1})^T a^{l+1} \right ) \otimes \sigma'(z^{l})" />

Please read [MLP.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Neural%20Network/MLP.ipynb) to learn more details.

---

### Datasets

There are two datasets used to implement Multilayer Perceptron Learning algorithm: The Iris Dataset is loaded from sklearn.datasets.

* MNIST Dataset

The MNIST Dataset is loaded from [*keras.dataset*](https://keras.io/api/datasets/). It consists of 70000 28 <img src="https://latex.codecogs.com/svg.image?\times" title="\times" /> 28 pixel images of hand written digits, 60000 of which are typically used as labeled training examples, where the other 10000 are used for testing your learning model on. 

* Fashion MNIST Dataset

The Fashion MNIST Dataset is also loaded from [*keras.dataset*](https://keras.io/api/datasets/). It consists of 70000 28 <img src="https://latex.codecogs.com/svg.image?\times" title="\times" /> 28 grayscale images of 10 fashion categories, 60000 of which are typically used as labeled training examples, where the other 10000 are used for testing your learning model on. 
