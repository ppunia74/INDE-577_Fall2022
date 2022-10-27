# K Nearest Neighbors (KNN)

This sub-repository mainly focuses on using K Nearest Neighbors to solve classification problems.

Contents of **K Nearest Neighbors (KNN)**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/Image): folder containing images used in README
* [Data](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/Data): folder containing all data files used in this module
   * [Hawks.csv](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/Data/Hawks.csv): Hawks Dataset 
* [KNN_iris.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/KNN_iris.ipynb): Jupyter notebook file building KNN algorithm from scratch and performing the algorithm using Iris Dataset to classify the iris species
* [KNN.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/KNN.ipynb): Jupyter notebook file containing
  - 1) Introduction of K Nearest Neighbors algorithm
  - 2) Building KNN algorithm from scratch and performing KNN using Hawks Dataset to classify the hawks species
  - 3) Compare the built algorithm with the *KNeighborsClassifier* from sklearn


![image](https://github.com/cissyyang1014/INDE577_DataScience_and_MachineLearning/blob/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/Image/KNN_1.JPG)

### A Short Summary

# K Nearest Neighbors (KNN)

### Idea: 
Data with similar features "should" be close in space.

### Algorithm:
* Step 1: Choose k
* Step 2: Calculate the distance from your new point to every point in your training set (computational expensive)
* Step 3:
    - For Classification: of the k closest points predict the majority label
    - For Regression: predict the average label for the K closest points
* Step 4: Evaluation of the performance and optimize K

### Calculate distance:

Two-Dimention formular:

<img src="http://chart.googleapis.com/chart?cht=tx&chl= d(p, q) = \sqrt {(p_1-q_1)^2+(p_2-q_2)^2}" style="border:none;">


n-Dimention formular:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=d(p, q) = \sqrt {(p_1-q_1)^2+(p_2-q_2)^2+...+(p_n-q_n)^2}=\sqrt {\sum \limits_{i=1}^{n}(p_i-q_i)^2}" style="border:none;">


### How to choose K?
K should be odd for classification.

### How should we measure error?

<img src="http://chart.googleapis.com/chart?cht=tx&chl=E = \frac {1}{M} \sum \limits_{i=1}^{M} \left(y_i \ne \hat {y_i} \right)" style="border:none;">

Please read [KNN.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/K%20Nearest%20Neighbors%20(KNN)/KNN.ipynb) to learn more details.

---

### Datasets

There are two datasets used to implement K Nearest Neighbors (KNN) algorithm:
* Iris Dataset:

The Iris Dataset is loaded from *sklearn.datasets*. The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
  
* Hawks Dataset:

The Hawks dataset has 908 cases and 19 columns. The data were collected by the students and faculty from Cornell College at Lake MacBride near Iowa City, Iowa. There are three different species of hawks: Red-tailed, Sharp-shinned, and Cooper's hawks.
