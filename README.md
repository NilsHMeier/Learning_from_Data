# Learning from Data

This repository contains the code for the lecture/seminar *"Learning from Data"* at the Leuphana University Lüneburg
taking place in the winter term 2022/23. The course is part of the master program Management & Data Science and is
taught by Prof. Burkhardt Funk and Jonas Scharfenberger. This repository will be updated with the course material during
the semester. The material will be available in the form of .py files and Jupyter notebooks.

## Course Description

The course will cover

- Theoretical foundation of statistical learning
- Learning settings and frameworks
- Linear models
- Regularization and feature selection
- Model evaluation
- Neuronal nets, SVMs and their application

### Lecture Content

1. Introduction, Confusion Matrix & ROC Curve
2. Learning Problem, Types of Machine Learning & Perceptron Learning Algorithm
3. Linear Regression, Gradient Descent & Stochastic Gradient Descent
4. Logistic Regression, Error Functions & Gaussian Discriminant Analysis
5. Decision Trees, Entropy Measurement & Gain Ratio
6. Learnability, Concept Learning & Hoeffding's Inequality
7. Generalization Theory, Breakpoints of Hypothesis Sets & VC Dimension
8. Bias and Variance, Model Complexity & Choice of Hypothesis Set
9. Support-Vector-Machines, Lagrange Formulation & Quadratic Programming
10. Neural Networks, Impact of Layers & Backpropagation
11. Feature Space, Transformations & Radial-Basis Functions
12. Overfitting, Regularization & Cross-Validation
13. Bagging, Boosting & Random Forests
14. Clustering, K-Means & Expectation-Maximization

#### Remark on Lecture 12

Within the file *lecture_12_overfitting.py* some experiments from the slides are replicated, namely the overfitting
depending on the noise level, the target complexity and the number of data points is studied. The code provides the
option to use multiprocessing in order to speed up the calculations. If you want to use the multithreading version,
please choose a proper number of threads - otherwise your computer might be running out of capacity. If you want to use
the single-threaded version, please set the variable *USE_MULTIPROCESSING* to *False*.

### Assignment Content

1. Perceptron Learning Algorithm
2. Linear Regression
3. Logistic Regression
4. Decision Trees
5. Hoeffding's Inequality
6. VC Dimension & Growth Functions
7. Bias-Variance-Tradeoff
8. Lagrange-Formulation & SVM
9. MLPs & Backpropagation
10. Non-Linear Transformations & RBFs
11. Regularization & Overfitting

#### Remark on Assignment 8

Within the implementation of the Support-Vector-Machine in assignment_08.py the *cvxopt* package is used to solve the
Lagrange formulation in order to receive the alphas. Since the standard solver does not always yield the correct
solution based on the input features, the Mosek solver is used instead (the *mosek* package is included in the
requirements file). However, a licence is required to make use of the Mosek solver. To get your own licence file, please
request an academic-licence on the [Mosek-Page](https://www.mosek.com/products/academic-licenses/).

## Contributing to this Repository

**General remark:** When committing your changes, please make use of the [guidelines](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53) on how to write commit messages!

If you want to add your solutions, remarks or anything else to this repository, choose one of the following ways:

1. Create a fork of this repository, commit your changes and open a pull request to the master branch.
2. Send me a message so that I can add you as a contributor to the repository. Then create a new branch from the master named *"dev/your_name"*, commit your changes and open a pull request to the master branch. **Please do not commit to the master branch directly!**

If you want to see any specific content in the repo, feel free to open up a new issue on GitHub and briefly describe what you want included in the repository. In case of any questions, please send me a short message.
