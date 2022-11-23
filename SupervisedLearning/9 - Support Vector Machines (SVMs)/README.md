# Support Vector Machines (SVMs)

Contents of **Support Vector Machines (SVMs)**

* [Image](https://github.com/ppunia74/INDE-577_Fall2022/tree/main/SupervisedLearning/9%20-%20Support%20Vector%20Machines%20(SVMs)/Image): contains images used in README
* [SVM.ipynb](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/9%20-%20Support%20Vector%20Machines%20(SVMs)/SVM_single_cell.ipynb): Jupyter notebook file performs the SVM algorithm to train a classifier to automatically annotate single-cell RNA-seq data.

![iamge](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/9%20-%20Support%20Vector%20Machines%20(SVMs)/Image/SVM.png)

### A Short Summary

# Support Vector Machines (SVMs)

Support Vector Machine (SVM) algorithm is a supervised machine learning method, which can be used in both regression and classification tasks. SVM finds a hyper-plane in N-dimensional space (N = number of features) that separates different types of data. To separates the boundaries, it estimates the maximum distance from the nearest points of two classes, and this optimal decision boundary is called support vectors. The region between the decision boundary defined by support vectors is margin. Thus, it is used for binary classification tasks. It works very well for linearly separable data. We use **Kernalized SVM** for non-linearly separable data. SVM is more often used for classification, but it can also be used for regression. The advantage of SVM is its significant accuracy with less computational power.

The working flow of a simple SVM algorithm can be simply summarized in two steps:

* Find boundaries (or hyperplane) that correctly separate the classes for the training data
* Picks the one that has the maximum distance from the closest data points

---

### Dataset

* Processed 3k PBMCs Dataset:

The [Processed 3k PBMCs](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k_processed.html) Dataset is loaded from [scanpy.datasets](https://scanpy.readthedocs.io/en/stable/api.html#module-scanpy.datasets). The Processed 3k Peripheral Blood Mononuclear Cells (PBMCs) Dataset consists of 3k PBMCs from a Healthy Donor and are freely available from [10x Genomics](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k). There are 2,700 single cells that were sequenced on the Illumina NextSeq 500.
