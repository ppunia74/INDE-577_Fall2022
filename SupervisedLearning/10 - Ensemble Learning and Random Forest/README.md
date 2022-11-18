# Ensemble Learning and Random Forest

This sub-repository mainly focuses on using Ensemble Learning algorithms (including Random Forest) to solve classification problems.

Content of **Ensemble Learning and Random Forest**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Image): folder containing images used in README
* [Data](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Data): folder containing all data files used in this module
  * [penguins.csv](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Data/penguins.csv): Penguins Dataset
* [Ensemble_Learning_makemoons.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_makemoons.ipynb): Jupyter notebook file performing Random Forest and two other different Ensemble Learning algorithms using the make_moons Dataset from sklearn (artificial dataset)
* [Ensemble_Learning_and_Random_Forest.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_and_Random_Forest.ipynb): Jupyter notebook file containing
  * a. Introduction of the Ensemble Learning algorithm and the Random Forest algorithm
  * b. Performing Random Forest algorithm and three other different Ensemble Learning algorithms using penguins dataset to classify penguins species

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Image/10image001.png)

### A Short Summary

# Ensemble Learning

Ensemble learning is to combine multiple individual models together to improve the overall performance of the algorithm. It is designed to improve the stability and the accuracy of the machine learning algorithms. When implement the ensemble learning, it assumes all the predictions are completely independent.

### Hard Voting Classifier

Hard Voting Classifier is one of simple ensemble learning techniques. Multiple models are used to make predictions for each data point, and the predictions by each model are considered as a separate vote. The prediction which got from majority of the models would be selected as the final prediction.

### Bagging (Bootstrap Aggregation)

Bagging (Bootstrap Aggregation) is one of advanced ensemble learning techniques. To implement bagging, multiple subsets are generated from the original dataset, selecting observations with replacement; Each learner trains the random subset and returns its prediction; Then just like the hard voting classifier, the bagged algorithm counts the votes and assigns the class with the most votes as the final prediction.

Please read [Ensemble_Learning_and_Random_Forest.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_and_Random_Forest.ipynb) to learn more details.

# Random Forest

Random Forest is an extension over bagging. In random forest, multiple subsets are generated from the training dataset and trained by a group of dicision tree classifiers to make individual predictions according to the different subsets of the training data; Each tree votes, and the prediction with the most votes would be selected as the final prediction.

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Image/example-of-random-forest-classifier.png)

Please read [Ensemble_Learning_and_Random_Forest.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_and_Random_Forest.ipynb) to learn more details.

---

### Datasets

There are two datasets used to implement Ensemble Learning algorithms (including Random Forest):

* make_moons Dataset:

The [make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset is an artificial data loaded from [sklearn.dataset](https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets). It can make two interleaving half circles. Adjusting the parameters (e.g., `noise`) can change the data distribution.

* Penguins Dataset:

The Penguins Dataset contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. These data were collected from 2007 - 2009 by Dr. Kristen Gorman's team. It consists of 344 rows and 7 columns. The three different species of penguins are Chinstrap, Adélie, and Gentoo penguins.
