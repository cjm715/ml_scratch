# ML-Scratch

This is a python library with numpy implementations of common machine learning algorithms. This was built for educational purposes as a means to learn the details of machine learning algorithms for myself. I also hope that others who are learning machine learning will benefit from viewing these numpy implementations.

This repository takes inspiration from a similar repository linked below that I recommend checking out
for alternative implementations: https://github.com/eriklindernoren/ML-From-Scratch

# Installing package `ml_scratch`

1. Clone this repo locally.
2. In Terminal, run `pip install -e .` within root directory of repository.

# The algorithms
List of algorithms implemented:
- [Linear Regression](#linear-regression)
- [Logistic Regression](#logistic-regression)
- [Gaussian Discriminant Analysis](#gaussian-discriminant-analysis)
- [Naive Bayes Classifier](#naive-bayes-classifier)
- [Support Vector Machine](#support-vector-machine-svm)
- [CART Decision Tree](#decision-tree)
- [Random Forest](#random-forest)
- [K-means clustering](#k-means-clustering)
- [Neural Network](#neural-network)

Each algorithm listed below will have all or subset of the following links:
1. **Code** : This link directs you to the numpy implementation code of the algorithm within the `ml_scratch` library.
2. **Notebook** : This link directs you to a jupyter notebook demonstrating the algorithm with a dataset. It also includes the code for generating the matplotlib visualizations shown in this readme.


## Linear Regression
Two solvers are implemented to fit data: (1) normal equations and (2) gradient descent.
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/LinearRegression.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/LinearRegression.ipynb)


## Logistic Regression
Two solvers are implemented to fit data: (1) gradient descent and (2) Newton's method.
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/LogisticRegression.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/LogisticRegression.ipynb)

## Gaussian Discriminant Analysis
![Alt text](./notebooks/images/GDA.svg)
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/GDA.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/GDA.ipynb)


## Naive Bayes Classifier
- Code (Coming soon. Numpy implementation is in notebook link below)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/NaiveBayes.ipynb)

## Support Vector Machine (SVM)
![Alt text](./notebooks/images/SVM.svg)

This implementation is a simplified version of the full Sequential Minimal Optimization (SMO) algorithm.
- Code (Coming soon. Numpy implementation is in notebook link below)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/SVM.ipynb)


## Decision Tree
![Alt text](./notebooks/images/decision_tree.svg)
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/TreeMethods.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/DecisionTree.ipynb)

## Random Forest
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/TreeMethods.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/RandomForest.ipynb)

## K-means clustering
![Alt text](./notebooks/images/kmeans.png)
- Code (Coming soon. Numpy implementation is in notebook link below)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/kMeans.ipynb)


## Neural Network
- [Code](https://github.com/cjm715/ml_scratch/blob/master/ml_scratch/NeuralNetworks.py)
- [Notebook](https://github.com/cjm715/ml_scratch/blob/master/notebooks/NeuralNetworks.ipynb)
