# K Means Clustering

This sub-repository implements K-Means Clustering to solve classification problems. The **Principal Component Analysis (PCA)** is used to reduced the dimension of the dataset.

Contents of **K Means Clustering**

* [Image](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/UnsupervisedLearning/K%20Means%20Clustering/Image): contains images used in subrepository
* [Data](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/UnsupervisedLearning/K%20Means%20Clustering/Data): contains data files used in this module

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/k%20means.png)

# K Means Clustering
K-means clustering is one of the most popular unsupervised machine learning algorithms. The objective of the algorithm is to partition a given data with similar features together into **clusters** to find underlying patterns. This achieved by using a pre-defined number (k) of clusters for the data. Points are only able to be part of one cluster, and each cluster has a "centroid" that roughly averages the data and represents the center of the cluster. Data points are then allocated to each of the clusters by assigning them to clusters in such a way that minimizes each cluster's sum of squares (often using Euclidean distance).

In general, the algorithm starts with a set of k randomly selected centroids, then iteratively optimizes the locations of those centroids until there is effectively no change in the centroids between iterations or data points stop changing clusters (meaning the clustering was successful) or the algorithm has completed the defined number of iterations.

The algorithm outputs 1) the centroids of the k clusters, which can be used to label new data, and 2) labels for the training data
 
## How does K-Means Clustering work?
1. Input a dataset and choose the number of clusters, k
  There are two primary ways of selecting the number of clusters:
    - **Elbow criterion**
      Run k-means clustering on the dataset for a range of values of k and calculate the sum of squared errors (SSE) for each k, then calculate the mean distance between data points and their cluster's centroid. As the number of clusters increeases, the number of datapoints per cluster will decrease and will decrease the SSE until the SSE equals zero when k equals the number of datapoints. The goal is to select a small value of k with a low SSE.
    
      In short: run the algorithm for different values of k and plot the k values against SSE, then select the value of k for the "elbow point" where the graph dips.
    - **Silhouette coefficient:**
      The silhouette coefficient is about finding a model with well-defined clusters. This value is calculated by taking the mean distance between a sample and all other points in the same cluster as well as its distance from all other points in the nearest cluster. The equation for a single sample is below:
    
      ![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/146045653-2898b2fa-6f54-4a50-a617-f8efb3f07ef5.png)

        Using the equation above, calculate the silhouette coefficient for all the clusters. A higher value indicates the sample is well matched to its own cluster and poorly matched to neighboring clusters. 
    - Other ways to choose k: cross-validation, information criteria, information theoretic jump method, G-means algorithm
3. Select k random points from the data as centroids. 
4. Calculate the Eucledean Distance from each feature vector to each centroid and assign each datapoint to the closest cluster centroid.

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/132998845-37a6f436-47b4-4337-a030-72bd9212d59f.png)

   The algorithm in the attached notebook uses the Euclidean Distance, but there are two other ways of calculating diatance that are used in machine learning algorithms: 
     
   - **Cosine distance:** determines the cosine of the angle between the point vectors of two points in n dimensional space. The closer the point vectors are by angle, the higher the Cosine Similarity. The equation is below:
    
   ![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/146046597-0f8d9449-30d1-4bc7-9560-1b271cff737b.png)

   - **Manhattan distance** total sum of the difference between the x-coordinates and the y-coordinates. The equation is below: 
    
      ![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/146046963-1f7a89a4-2a11-4466-b756-5bb1960d4c44.png)

   The Manhattan distance is so-called because it measures the distance between two points in a city if you could only travel along orthogonal city blocks.
   
4. Update the centroid for each cluster by taking the mean of all the datapoints assigned to that centroid's cluster.

5. Report previous steps 3-4 until the centroids converge (no change in cetnroids), datapoints stop moving between clusters, or the algorithm reaches the maximum number of iterations. Note that the algorithm may converge on a _local_ optimum, so it is important to run the algorithm several times. 

## Common Challenges:
The algorithm ensures the distribution of points among clusters is roughly even, when in reality they could be different sizes. 

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/08/Screenshot-from-2019-08-09-13-15-26.png)

Or the algorithm also struggles when the real clusters are of different densities.

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/08/Screenshot-from-2019-08-09-13-19-05.png)

One potential solution is to use a larger number of clusters, but there is also a risk of being so overly specific that the data loses any meaning. Alternatively, one could generalize k-means clustering to allow for differing cluster widths and densities.

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/UnsupervisedLearning/K%20Means%20Clustering/Image/146047921-d690478c-1ca2-4a7b-bf81-1a4d9b73806e.png)

The plot on the left is normal k-means, without generalization. This results in a non-intuitive cluster boundary. The plot in the middle allows for different cluster widths, which results in more intuitive clusters of different sizes. Finally, the right plot further allows for different widths per dimension, which results in and elliptical rather than circular shape. Depending on the dataset, these could improve the results.

Other common issues with k-means clustering include that it is dependent on the initial values (so it must be run several times with different initial centroid values), and the centroids can be dragged by outliers in the data.

# Principal Component Analysis (PCA)

PCA is a technique to reduce the dimensionality of the large dataset. It transforms a large set of variables into a smaller one that still contains most of the information. 

To calculate PCA:

* Calculate the mean values of each variables
* Center the values in each variables by subtracting the corresponding mean values
* Calculate the covariance matrix of the centered matrix
* Calculate the eigen decomposition of the covariance matrix

---