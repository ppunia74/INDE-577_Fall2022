# Support Vector Machines (SVMs)

Contents of **Support Vector Machines (SVMs)**

* [Image](https://github.com/cissyyang1014/DataScience_and_MachineLearning/tree/main/SupervisedLearning/Support%20Vector%20Machines%20(SVMs)/Image): folder containing images used in README
* [SVM_single_cell.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Support%20Vector%20Machines%20(SVMs)/SVM_single_cell.ipynb): Jupyter notebook file performs the SVM algorithm to train a classifier to automatically annotate single-cell RNA-seq data

![iamge](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Support%20Vector%20Machines%20(SVMs)/Image/Support-Vector-Machines-1.jpg)

### A Short Summary

# Support Vector Machines (SVMs)

Support Vector Machine (SVM) algorithm is a powerful supervised machine learning algorithm, which can be used in both regression and classification problems.

There are two types of SVM, linear SVM (simple SVM) and non-linear SVM (kernel SVM). Linear SVM is used for linearly separable data, while the non-linear SVM is used for non-linearly separated data.

The goal of the SVM algorithm is to create the best decision boundary to separate the classes. To achieve the goal, it finds the maximum distance from the nearest points of two classes, and this optimal decision boundary is called support vectors. The region between the decision boundary defined by support vectors is margin.

The working flow of a simple SVM algorithm can be simply summarized in two steps:

* Find boudaries (or hyperplane) that correctly separate the classes for the trainig data
* Picks the one that has the maximum distance from the closest data points

Please read  [SVM_single_cell.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Support%20Vector%20Machines%20(SVMs)/SVM_single_cell.ipynb) to learn more details.

---

### Dataset

* Processed 3k PBMCs Dataset:

The [Processed 3k PBMCs](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k_processed.html) Dataset is loaded from [scanpy.datasets](https://scanpy.readthedocs.io/en/stable/api.html#module-scanpy.datasets). The Processed 3k Peripheral Blood Mononuclear Cells (PBMCs) Dataset consists of 3k PBMCs from a Healthy Donor and are freely available from [10x Genomics](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k). There are 2,700 single cells that were sequenced on the Illumina NextSeq 500. 
