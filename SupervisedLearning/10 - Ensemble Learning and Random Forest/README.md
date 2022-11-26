# Ensemble Learning and Random Forest

This sub-repository demonstrates the implementation of Ensemble Learning algorithms (including Random Forest) to solve classification problems.

Content of **Ensemble Learning and Random Forest**

* [Image](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/10%20-%20Ensemble%20Learning%20and%20Random%20Forest/Image): contains images used in README
* [Data](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/10%20-%20Ensemble%20Learning%20and%20Random%20Forest/Data): contains datasets used in this module
* [Ensemble_Learning_makemoons.ipynb]([b](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/10%20-%20Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_makemoons.ipynb)): Jupyter notebook file performing Random Forest and two other different Ensemble Learning algorithms using the make_moons Dataset from sklearn (artificial dataset)
* [Ensemble_Learning_and_Random_Forest.ipynb](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/10%20-%20Ensemble%20Learning%20and%20Random%20Forest/Ensemble_Learning_and_Random_Forest.ipynb): Jupyter notebook file containing
  * a. Introduction of the Ensemble Learning algorithm and the Random Forest algorithm
  * b. Performing Random Forest algorithm and three other different Ensemble Learning algorithms using penguins dataset to classify penguins species

![image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Ensemble%20Learning%20and%20Random%20Forest/Image/10image001.png)

# Ensemble Learning

Ensemble learning model makes predictions by combining multiple individual models together. The combination of multiple models improve the overall performance of the algorithm and makes ensemble models more flexible (less bias) and less data-sensitive (less variance). Ensemble algorithm assumes all the predictions are completely independent.


![image](https://miro.medium.com/max/2000/1*bUySDOFp1SdzJXWmWJsXRQ.png)

### Hard Voting Classifier

Hard Voting Classifier is one of simple ensemble learning techniques. Multiple models are used to make predictions for each data point, and the predictions by each model are considered as a separate vote. The prediction which got from majority of the models would be selected as the final prediction.

### Bagging (Bootstrap Aggregation)

In this method, a bunch of individual models are trained in a sequential way (or parallel way) using a random subset of the data. In this approach individual models learns from mistakes made by the previous model. Similar to hard voting classifier, the bagged algorithm counts the votes and assigns the class with the most votes as the final prediction.

# Random Forest

In a single decision tree model typically has high variance and has performance similar to many other models. However, one way to improve performance is to produce many variants of a single decision tree by selecting every time a different subset of the same training set in the context of randomization-based ensemble methods.

![image](https://miro.medium.com/max/2000/1*jXkT3mj1mCqMaX5SqU1wNw.png)

Random forest method is an extension of decision trees, which belongs to the a class of machine learning algorithms called **ensemble methods**. Random forest draws multiple subsets from the training data and trained by a group of decision tree classifiers to make individual predictions according to the different subsets of the training data.  Random forest works with both a mix of categorical and numerical variables, and is less sensitive to scaling and is computationally less intensive than SVM. Random forest works well even with the missing data and is less sensitive to overfitting without needing hyperparameter tuning. Each tree votes, and the prediction with the most votes would be selected as the final prediction.

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/10%20-%20Ensemble%20Learning%20and%20Random%20Forest/Image/random-forest-classifier.png)


Generalized Algorithm
1. Select n (e.g. 100) random subsets from the training data
2. Train n (e.g. 100) decision trees
   - one random subset is used to train one decisions tree
   - the optimal splits for each decision tree are based on a random subset of features (e.g. 10 features in total, randomly select 5 out of 10 features to split)
 1. Each individual tree predicts the records/candidates in the test set, independently
 2. Make the final prediction
    - For each candidate in the test set, random forest uses the class (e.g. cat or dog) with the majority vote as this candidate's final prediction

The fundamental principle of ensemble methods is based on randomization. There are three main decisions to make when constructing a tree:
- method for splitting the leaves
- type of predictor used in each leaf
- method for injecting randomness into the tree (bagging or bootstrapping)

There are a few stopping criteria:
- minimum number of samples in a terminal node to allow it to split
- minimum number of samples in a leaf node when the terminal node is split
- maximum tree depth, i.e. the maximum number of levels a tree can grow
- Tree accuracy (defined by the Gini Index) is less than a fixed threshold


### Datasets

There are two datasets used to implement Ensemble Learning algorithms (including Random Forest):

* make_moons Dataset:

The [make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset is an artificial data loaded from [sklearn.dataset](https://scikit-learn.org/stable/modules/classes.html?highlight=dataset#module-sklearn.datasets). It can make two interleaving half circles. Adjusting the parameters (e.g., `noise`) can change the data distribution.

* Penguins Dataset:

The Penguins Dataset contains size measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica. These data were collected from 2007 - 2009 by Dr. Kristen Gorman's team. It consists of 344 rows and 7 columns. The three different species of penguins are Chinstrap, Adélie, and Gentoo penguins.
