# Supervised Learning

Supervised learning methods are widely used in practice. The intuition of Supervised learning is that algorithm learns from the **labeled data** and helps the user predict outcomes for unforeseen data. In other words the algorithms are designed to "learn by example". Labeled training data means each example consists of a set of input objects (features) and a desired output value (labels), and the algorithm learns the patterns between the input and the output. The learning precess is just like to be taught by a teacher that the correct answers (labels) are known and the algorithm iteratively makes predictions which would be corrected by the teacher. 

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Image/supervised_learning.png)

There are nine supervised learning algorithms performed in this sub-repository. Most of the algorithms are coded from scratch. Each algorithm is performed in two different datasets, one of which is from the class example. 

List of **Supervised Learning** Algorithms in this repository:

* [K Nearest Neighbors (KNN)]()
* [Gradient Descent](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/2%20-%20Gradient%20Descent)
* [Linear Regression](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/Linear%20Regression)
* [Logistic Regression](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/4%20-%20Logistic%20Regression)
* [Perceptron](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/Perceptron)
* [Neuron Network]()
* [Decision Tree]()
* [Ensemble Learning and Random Forest]()
* [Support Vector Machines (SVMs)]()

---

In general supervised learning methods are used for two types task. Primarily supervised learning methods attempt to fit a line to that labeled data to either split it into categories (classification) or represent a trend (regression), and this line allows us to make predictions. Successfully building, scaling, and deploying accurate supervised machine learning data science models takes time and technical expertise from a team of highly skilled data scientists. Moreover, data scientists must rebuild models to make sure the insights remain true until the data changes. 

There are two types of supervised learning methods: 
- **classification**
  - goal: assign each input value and assign it to a class or category that it fits best based on the training data
  - predicted output is a list of tags or values
  - e.g. fake news, with tags "fake" or "not fake"
  - popular classification algorithms: linear classifiers, logistic regression, K nearest neighbor classifier, support vector machines, decision tree classifier, random forest classifier
- **regression**
  - goal: find a relationship between dependent and independent variables
  - predicted output is a target numeric value, given a set of features, and is represented by a line of best fit (for a model with more than two features, this is often a hyperplane)
  - e.g. impact of social media advertising on company's sales
  - popular regression algorithms: linear regression, logistic regression, polynomial regression, K nearest neighbor regression, decision trees regression, support vector regression


![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Image/types_of_supervised_learning.png)

