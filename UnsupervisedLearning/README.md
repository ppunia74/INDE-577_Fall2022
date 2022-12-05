# Unsupervised Learning

There are two unsupervised learning algorithm performed in this sub-repository. The K Means Clustering algorithm is coded from scratch. 

Contents of **Unsupervised Learning**:
* [K Means Clustering](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/UnsupervisedLearning/K%20Means%20Clustering)

---

**Unsupervised Learning**

Unsupervised learning is used to find underlying patterns in data, and involves finding structures and relationships from inputs. This is helpful when we have unlabeled data or aren't sure which outputs are meaningful. There is a set of data that is **unlabeled** to learn from, with the goal of identifying patterns in that data. In contrast to supervised learning, which focuses on labels, unsupervised learning focuses on **features** of the data. 

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/Image/unsup_header.png)


The result of an unsupervised learning model is to place observations into specific clusters (clustering), or to create rules to identify associations between variables (association). With large datasets, it is important to keep the datasets used for training as small and efficient as possible. Unsupervised learning algorithms are also able to utilize "dimensionality reduction" to represent the information in a dataset with only a portion of the actual content, and these are sometimes implemented as part of pre-processing. 

Unsupervised learning allows data scientists to perform more complex processing tasks compared to supervised learning. In this sense, it is more powerful because it can identify patterns within a dataset not visible to a human observer, however unsupervised learning can also be more unpredictable compared with other deep learning and reinforcement learning methods. 

Variables:
- features

Types of Algorithms:
- **Dimensionality Reduction**
  - goal: combine parts of data in unique ways to convey meaning in less space
  - e.g. analyzing high resolution images by reducing the resolution an appropriate amount
  - common algorithms: principal component analysis (PCA), singular-value decomposition (SVD)
- **Clustering**
  - goal: find similarities/patterns in observed data and put them into "clusters" or subgroups hat are as similar to others within the group and as different from data in other clusters as possible
  - e.g. detecting groups of similar visitors to a blog (40% male comic book lovers who read in the evening vs 20% young sci-fi lovers who visit on weekends)
  - common algorithms: K-means clustering, hierarchical clustering, probabilistic clustering
- **Association**
  - goal: identify sequences and new and interesting insights between different objects in a set. In this case, the pattern identified is a rule (like if *this* then *that*)
  - e.g. analyzing supermarket data and finding that people who buy bbq sauce and potato chips tend to also buy steak
  - common algorithms: A priori algorithm, frequent pattern (FP) growth algorithm, rapid association rule mining (RARM), ECLAT algorithm


![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/Image/unsup_cat.png)
