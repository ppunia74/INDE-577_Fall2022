# Logistic Regression

This sub-repository mainly focuses on using Logistic Regression algorithm to solve classification problems.

Contents of **Logistic Regression**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Logistic%20Regression/Image): folder containing images used in README
* [Data](): Folders containing all data files used in this module
  * [candidates_data.csv](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Logistic%20Regression/Data/candidates_data.csv): Candidates Dataset
  * [pima-indians-diabetes.txt](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Logistic%20Regression/Data/pima-indians-diabetes.txt): Pima Indians Diabetes Dataset
* [Logistic_Regression_candidates.ipynb](): Jupyter notebook file containing
  * a. Building Logistic Regression algorithm from scratch
  * b. Performing the algorithm using Candidates Dataset to predict whether the student can be admitted or rejected by the college
* [Logistic_Regression.ipynb](): Jupyter notebook file containing
  * a. Introduction of Logistic Regression algorithm
  * b. Building Logistic Regression algorithm from scratch 
  * c. Performing the algorithm using Pima Indians Diabetes Dataset to predict whether the person would have diabetes or not in the 5 years

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Logistic%20Regression/Image/logisticregression.JPG)

### A Short Summary
# Logistic Regression

In logistic regression, it uses a logistic function to modela binary dependent variable. We assume that 
* Each example (x,y) belongs to one of the two complemetary classes (0 or 1);
* The examples are generated under the same distribution and independent of each other.

### Sigmoid Function

The sigmoid function is used in the logistic regression, that

<img src="https://latex.codecogs.com/svg.image?\sigma(z)=\frac{1}{1&plus;e^{-z}}" title="\sigma(z)=\frac{1}{1+e^{-z}}" />

The shape of a sigmoid function is an "S"-shaped curve, that can take any real-valued number and map it into a value between 0 and 1.

### Cross Entropy Loss (CEL)

For a simple binary classification, we use Bernoulli Distribution, <img src="https://latex.codecogs.com/svg.image?\mathbb{P}(y|x)=&space;\hat&space;y^y&space;(1-\hat&space;y)^{1-y}" title="\mathbb{P}(y|x)= \hat y^y (1-\hat y)^{1-y}" />. 

To maximize the probability, maximum likelihood is used, and thus, we need to minimize <img src="https://latex.codecogs.com/svg.image?-Log[\mathbb{P}(y|x)]" title="-Log[\mathbb{P}(y|x)]" />, which is defined as the Cross Entropy Loss (CEL).

Then, we can use the concept of **Gradient Descent** to find the minimum of CEL.

<img src="https://latex.codecogs.com/svg.image?L(w;y)=-ylog&space;\sigma(z)-(1-y)&space;log&space;(1-\sigma(z))" title="L(w;y)=-ylog \sigma(z)-(1-y) log (1-\sigma(z))" />

To take the partial derivations,

<img src="https://latex.codecogs.com/svg.image?\triangledown&space;L[w;&space;(x,y)]=\begin{bmatrix}\frac{\partial&space;L}{\partial&space;w_j}=(\hat&space;y-y)x_j\\&space;\frac{\partial&space;L}{\partial&space;b}=\hat&space;y-y\end{bmatrix}" title="\triangledown L[w; (x,y)]=\begin{bmatrix}\frac{\partial L}{\partial w_j}=(\hat y-y)x_j\\ \frac{\partial L}{\partial b}=\hat y-y\end{bmatrix}" />

where <img src="https://latex.codecogs.com/svg.image?\hat&space;y&space;=&space;\sigma(z)" title="\hat y = \sigma(z)" />.

### Algorithm

* Step 1: Randomly select <img src="https://latex.codecogs.com/svg.image?(x,y)" title="(x,y)" /> from the training set
* Step 2: Feed-Forward into the Nueral Network
* Step 3: Update weights and bias (choose learning rate <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />)
    - <img src="https://latex.codecogs.com/svg.image?w_1&space;->&space;w_1&space;-&space;\alpha&space;(\hat&space;y-y)x_1" title="w_1 -> w_1 - \alpha (\hat y-y)x_1" />
    - <img src="https://latex.codecogs.com/svg.image?w_2&space;->&space;w_2&space;-&space;\alpha&space;(\hat&space;y-y)x_2" title="w_2 -> w_2 - \alpha (\hat y-y)x_2" />
    - <img src="https://latex.codecogs.com/svg.image?w_3&space;->&space;w_3&space;-&space;\alpha&space;(\hat&space;y-y)" title="w_3 -> w_3 - \alpha (\hat y-y)" />
* Step 4: Repeat Step 1 to Step 3 until desired loss in-sample is reached, or a maximum number of steps is reached

Please read [Logistic_Regression.ipynb]() to learn more details.

---
### Datasets

There are two datasets used to implement Logistic Regrssion algorithm:
* Candidates Dataset

The Candidates Dataset is provided by Dr. Randy Davila. It is a historical admissions records of the college, which contains 40 rows and 4 columns, including the GMAT score, GPA, work experience, and whether the student is admitted (marked as "1") or rejected (marked as "0") by the college.

* Pima Indians Diabetes Dataset

The Pima Indians Diabetes Dataset involves the medical details (e.g., plasma glucose concentration, Diastolic bolld pressure, BMI, etc.) of Pima Indians and the onset of diabetes within 5 years. There are 768 observations with 9 variables.
