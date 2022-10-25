# Perceptron

This sub-repository illustrate the use of Perceptron algorithm to solve classification problems.

Contents of **Perceptron**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Perceptron/Image): contains images used in README
* [Data](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/Perceptron/Data): contains all data files used in this module
  * [banknote authentication.csv](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Perceptron/Data/BankNote_Authentication.csv): Banknote Authentication Dataset
* [Perceptron_iris.ipynb](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Perceptron/Perceptron_iris.ipynb): Jupyter notebook file contains:
  * a. Building Perceptron algorithm from scratch
  * b. Performing perceptron algorithm using Iris Dataset to classify the iris species
* [Perceptron.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Perceptron/Perceptron.ipynb): Jupyter notebook file contains
  * a. Introduction of the perceptron algorithm
  * b. Building Perceptron algorithm from scratch
  * c. Performing the perceptron algorithm using penguins dataset to classify Adelie penguins and Gentoo penguins

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Perceptron/Image/perceptron.jpeg)

### A Short Summary

# Perceptron

The perceptron is a supervised, single layer neural network binary classifier. In perceptron algorithm, each neuron takes inputs, weighs them separately, sums them up, and passes this sum through a nonlinear function (activation function) to produce output. Simply, it can be thought of as a mathematical model that draws a boundary to separate two groups in space.

The perceptron involves 4 components: Input values, Weights and bias, Weighted sum (net sum), and Activation function.

### Weighted Sum

Considering the bias, the weighted sum <img src="https://latex.codecogs.com/svg.image?\sum" title="\sum" /> multiplies all inputs of <img src="https://latex.codecogs.com/svg.image?X" title="X" /> by weight <img src="https://latex.codecogs.com/svg.image?w" title="w" /> and then adds them up, that

<img src="https://latex.codecogs.com/svg.image?\sum&space;=&space;w^T&space;\bar&space;X^i=w_1X^i_1&plus;w_2X^i_2&plus;...&plus;w_nX^i_n&plus;b\cdot&space;1.0" title="\sum = w^T \bar X^i=w_1X^i_1+w_2X^i_2+...+w_nX^i_n+b\cdot 1.0" />, where <img src="https://latex.codecogs.com/svg.image?\bar&space;X^i=\begin{bmatrix}X^i_1\\&space;\vdots\\&space;X^i_n&space;\\&space;1.0\end{bmatrix}" title="\bar X^i=\begin{bmatrix}X^i_1\\ \vdots\\ X^i_n \\ 1.0\end{bmatrix}" /> and <img src="https://latex.codecogs.com/svg.image?b" title="b" /> is the bias.

### Activation Function

The activation function is used to convert perceptron output. In our algorithm, sign function is used.

<img src="https://latex.codecogs.com/svg.image?\hat&space;y^i=sign(w^T&space;\bar&space;X^i)=\left\{\begin{matrix}1,&space;\space\space&space;w^T\bar&space;X^i>0\\&space;-1,\space\space&space;w^T\bar&space;X^i<0\end{matrix}\right." title="\hat y^i=sign(w^T \bar X^i)=\left\{\begin{matrix}1, \space\space w^T\bar X^i>0\\ -1,\space\space w^T\bar X^i<0\end{matrix}\right." />

### Loss Function

In our algorithm, 

<img src="https://latex.codecogs.com/svg.image?L(w,&space;\bar&space;X^i)=\frac{1}{2}\sum_{i=1}^n(\hat&space;y^i-y^i)^2&space;=&space;\frac{1}{2}\sum_{i=1}^n\left&space;(sign(w^T&space;\bar&space;X^i)-y^i\right)^2" title="L(w, \bar X^i)=\frac{1}{2}\sum_{i=1}^n(\hat y^i-y^i)^2 = \frac{1}{2}\sum_{i=1}^n\left (sign(w^T \bar X^i)-y^i\right)^2" />

The **stochastic gradient descent** will be used to optimize the algorithm. Since the sign function cannot be derived, we use an **approximate** gradient of the loss fuction that

<img src="https://latex.codecogs.com/svg.image?\triangledown&space;L(w,&space;\bar&space;X^i)=\left&space;(sign(w^T&space;\bar&space;X^i)-y^i\right)\bar&space;X_i" title="\triangledown L(w, \bar X^i)=\left (sign(w^T \bar X^i)-y^i\right)\bar X_i" />

And thus, to update the weights, <img src="https://latex.codecogs.com/svg.image?w_{n&plus;1}=w_n&space;-&space;\alpha&space;\triangledown&space;L(w,&space;\bar&space;X^i)" title="w_{n+1}=w_n - \alpha \triangledown L(w, \bar X^i)" />, where <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" /> is the learning rate.


---
### Dataset

There are two datasets used to implement Perceptron algorithm:

* Iris Dataset:

The Iris Dataset is loaded from sklearn.datasets. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

* Banknote Authentication Dataset:

The banknote authentication dataset contains data extracted from images that were taken from genuine and forged banknote-like specimens. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
