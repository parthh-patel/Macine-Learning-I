# Machine Learning I @ Georgia Tech

## [Report 1 - K-Means and Spectral Clustering Algorithms](https://github.com/parthh-patel/Macine-Learning-I/blob/main/1%20-%20K-Means%20and%20Spectral%20Clustering%20Algorithms.pdf)
- Building blocks of a fundamental machine learning algorithm: K-Means clustering (unsupervised learning)
- Derivations of the optimizing function in K-means
- Image compression using K-means, built from scratch without any packages (includes images of original vs compressed images w/ # of iterations and run time)
- Evaluation of the clusters created by K-means
- Spectral clustering which creates clusters based on the geometry or connectivity of data 

## [Report 2 - PCA, Kernel Density Estimation, and Facial Recognition](https://github.com/parthh-patel/Macine-Learning-I/blob/main/2%20-%20PCA%2C%20Kernel%20Density%20Estimation%2C%20and%20Facial%20Recognition.pdf)
- Mainly covers PCA which is a popular linear dimensionality reduction technique 
- Derivations and math behind the PCA algorithm
- PCA algorithm built from scratch without any packages and tested on an example problem of food consumptions in European countries
- PCA algorithm on a dataset of faces using the ISOMAP technique to group images together based on the composition of images
- Density estimation which captures the distributional information of the data (plots of 2D and 3D histograms, KDE plots, heat maps, contour plots, etc.)
- Applying kernel density estimation on a psychological experiment that tested the relationship between different volumes of the brain and political orientation
- Facial recognition by training a PCA algorithm with images of two different subjects and evaluating results using test images and projection residual scores

## [Report 3 - EM, GMM, Optimization, Classification Models, Spam Filter](https://github.com/parthh-patel/Macine-Learning-I/blob/main/3%20-%20EM%2C%20GMM%2C%20Optimization%2C%20Classification%20Models%2C%20Spam%20Filter.pdf)
- Implementation of the EM algorithm from scratch to fit a Gaussian mixture model for the MNIST hand-written digits dataset
- Includes reconstruction of the mean images of the GMM components, heatmaps of covariance matrices, and confusion matrices used to calculate accuracies and misclassification rates
- Basics of optimization theory and the math behind derivations of logistic regression, cost functions, and gradient descents 
- Bayes classifier algorithm to fit a spam filter by hand and the logical math/steps behind classifying emails as spam/not spam
- Comparison of various classification models (Naive Bayes, Logistic Regression, and KNN) on a dataset about marriage dataset to predict divorce rates
- Evaluating each classification model's training and testing accuracies as well as the various decision boundaries inherent to the classification models

## [Report 4 - Comparing Classification Models, Neural Net, KNN, SVM, Feature Selection, CUSUM, Image Reconstruction Using Lasso and Ridge](https://github.com/parthh-patel/Macine-Learning-I/blob/main/4%20-%20Comparing%20Classification%20Models%2C%20Neural%20Net%2C%20KNN%2C%20SVM%2C%20Feature%20Selection%2C%20CUSUM%2C%20Image%20Reconstruction%20Using%20Lasso%20and%20Ridge.pdf)
- Comparing multi-class classifiers for handwritten digit classification
- Calculated precision, recall, f1-score, confusion matrix, training accuracy, and testing accuracy for KNN, Logistic Regression, Linear SVM, Kernel SVM, and Neural Net
- Derivations and math behind SVM including an SVM exercise by hand
- Derivations and math behind Neural networks and backpropagation
- Feature selection and change-point detection using mutual information value
- CUSUM exercise to detect change in randomly generated i.i.d. samples
- Medical image reconstruction using cross-validation errors for both Lasso and Ridge regression algorithms
 
## [Report 5 - AdaBoost, Random Forest, One-class SVM for Email Spam Classifier, Locally Weighted Linear Regression, Bias-variance Tradeoff](https://github.com/parthh-patel/Macine-Learning-I/blob/main/5%20-%20AdaBoost%2C%20Random%20Forest%2C%20One-class%20SVM%20for%20Email%20Spam%20Classifier%2C%20Locally%20Weighted%20Linear%20Regression%2C%20Bias-variance%20Tradeoff.pdf)
- Difference between boosting and bagging, ways to prevent overfitting in CART, explanation of OOB errors, and bias-variance tradeoff in linear regression setting
- AdaBoost algorithm by hand on sample data points including decision stumps, calculations of weights for each iteration, and final classifier results
- Fit and visualized a CART model for a spam classifier using the UCR email spam dataset
- Fit several Random Forest models on the same email spam dataset to compare test errors and plotted the test error vs number of trees for comparison
- Used a One-class SVM approach for spam filtering and plotted the misclassification error vs gamma parameter for the OneClassSVM algorithm 
- Created a local linear weighted linear regression function and used 5-fold CV to tune the bandwidth parameter $h$
- Used the tuned model to make predictions for given input values and plotted a graph showing training data and prediction curve