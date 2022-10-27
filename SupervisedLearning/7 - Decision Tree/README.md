# Decision Tree

This sub-repository mainly focuses on using Decision Tree Algorithm to solve classification problems.

Contents of **Decision Tree**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Decision%20Tree/Image): folder containing images used in README
* [Data](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Decision%20Tree/Data): folder containing all data files used in this module
  * [penguins.csv](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Data/penguins.csv): Penguins Dataset
* [Decision_Tree_makemoons.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Decision_Tree_makemoons.ipynb): Jupyter notebook file containing
  * a. Performing decision tree algorithm using the make_moons Dataset from sklearn (artificial dataset)
  * b. Using confusion matrix to evaluate the algorithm
* [Decision Tree.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Decision%20Tree.ipynb): Jupyter notebook file containing
  * a. Introduction of the decision tree algorithm
  * b. Performing the decision tree algorithm using penguins dataset to classify penguins species
  * c. Using confusion matrix to evaluate the algorithm
  * d. Increasing the depth of the decision tree algorithm and compare

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Image/decision-tree-classification-algorithm.png)

### A Short Summary

# Decision Tree

Decision tree is a non-parametric supervised machine learning algorithm used for both classification and regression (Regression tree) tasks. The data is continuously split according to a certain parameter. It starts with the root node, which further gets divided into two or more homogeneous sets. The tree has two entities, decision nodes and leaves. The decision nodes are where the data is split, while the leaves are the decisions or the final outcomes.

### Entropy

Entropy is a metric to measures the impurity, or uncertainty in data. The formula of entropy is 

<img src="https://latex.codecogs.com/svg.image?Entropy&space;=&space;\sum_{i=1}^c&space;-P_i&space;log_2&space;(P_i)" title="Entropy = \sum_{i=1}^c -P_i log_2 (P_i)" />

where <img src="https://latex.codecogs.com/svg.image?c" title="c" /> is the total number of classes and <img src="https://latex.codecogs.com/svg.image?P_i" title="P_i" /> is the probability of class <img src="https://latex.codecogs.com/svg.image?i" title="i" /> in the node.

### Gini Index

Gini index is used to determine the impurity or purity when building a decision tree in the classification and regression tree (CART) algorithm. The formula of Gini index is

<img src="https://latex.codecogs.com/svg.image?Gini=&space;1&space;-&space;\sum_{i=1}^c&space;P_i^2" title="Gini= 1 - \sum_{i=1}^c P_i^2" />

where <img src="https://latex.codecogs.com/svg.image?c" title="c" /> is the total number of classes and <img src="https://latex.codecogs.com/svg.image?P_i" title="P_i" /> is the probability of class <img src="https://latex.codecogs.com/svg.image?i" title="i" /> in the node.

### Information Gain (IG)

Information gain is used to decide which feature to split at each step when building the decision tree. A decision tree algorhtim would maximize the value of information gain, and thus, the split with the highest value of information gain will be selected. If a node <img src="https://latex.codecogs.com/svg.image?V" title="V" /> is split into <img src="https://latex.codecogs.com/svg.image?V_l" title="V_l" /> and <img src="https://latex.codecogs.com/svg.image?V_r" title="V_r" />, the formula of information gain is

<img src="https://latex.codecogs.com/svg.image?IG&space;=&space;Entropy(V)-w_l&space;\times&space;Entropy(V_l)&plus;w_r&space;\times&space;Entropy(V_r)" title="IG = Entropy(V)-w_l \times Entropy(V_l)+w_r \times Entropy(V_r)" />

where <img src="https://latex.codecogs.com/svg.image?w" title="w" /> is the weight, that <img src="https://latex.codecogs.com/svg.image?w_l=\frac&space;{|V_l|}{V}" title="w_l=\frac {|V_l|}{V}" /> and <img src="https://latex.codecogs.com/svg.image?w_r=\frac&space;{|V_r|}{V}" title="w_r=\frac {|V_r|}{V}" />.

Please read [Decision Tree.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Decision%20Tree.ipynb) to learn more details.

# Confusion Matrix

A confusion matrix is a table used to describe the performance of a classification algorithom. It compares the actual values with those predicted by the model. 

There are four basic terms:

* True Positive: An outcome where the model correctly predicts the positive class.
* True Negative: An outcome where the model correctly predicts the negative class.
* False Positive: An outcome where the model incorrectly predicts the positive class.
* False Negative: An outcome where the model incorrectly predicts the negative class.

Based on these four terms, we can also calculate the Accuracy, Recall, Precious, and <img src="https://latex.codecogs.com/svg.image?F_1" title="F_1" /> score. These parameters can be used to evaluate the algorithm.

Please read [Decision Tree.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Decision%20Tree/Decision%20Tree.ipynb) to learn more details.

---

### Datasets

There are two datasets used to implement Decision Tree algorithm:

* make_moons Dataset:

The [make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset is an artificial data loaded from [sklearn.dataset](https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets). It can make two interleaving half circles. Adjusting the parameters (e.g., `noise`) can change the data distribution.

* Penguins Dataset:

The Penguins Dataset contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. These data were collected from 2007 - 2009 by Dr. Kristen Gorman's team. It consists of 344 rows and 7 columns. The three different species of penguins are Chinstrap, Adélie, and Gentoo penguins.
