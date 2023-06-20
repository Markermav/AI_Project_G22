# AI_Project_G22
COMP-6721-AI , A comparative analysis of image classification using CIFAR10 Dataset
Description:
This repository contains source code for Naive Bayes for supervised learning classification, Decision Tree for both supervised and semi supervised learning classification, CNN for supervised learning classification. The Cifar 10 Dataset was used for the procedure.
##Naive Bayes:
Data Loading: The CIFAR-10 dataset is loaded using scikit-learn's fetch_openml function. The dataset is split into input features (X) and target labels (y).
Feature Extraction: The Histogram of Oriented Gradients (HOG) features are extracted from the images. HOG is a widely used technique for capturing shape and gradient information in an image.
Data Split: The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The testing set is kept as 20% of the total data.
Preprocessing and Classifier Pipeline: A pipeline is created using scikit-learn's make_pipeline function, which consists of a StandardScaler for feature scaling and a GaussianNB classifier, which is a Naive Bayes classifier assuming Gaussian distribution.
Hyperparameter Tuning: Grid search with cross-validation (GridSearchCV) is performed to find the best hyperparameters for the classifier. In this case, no hyperparameters are specified to tune.
Model Training and Prediction: The best classifier obtained from hyperparameter tuning is trained on the training set and used to predict the labels for the test set.
Performance Evaluation: Various performance metrics such as precision, recall, F1 score, and accuracy are calculated using scikit-learn's metrics module.
Bar Plot of Predicted Class Labels: A bar plot is created to visualize the predicted class labels and their counts using matplotlib.
t-SNE Visualization: t-SNE (t-Distributed Stochastic Neighbor Embedding) is applied to reduce the dimensionality of the test set to 2 dimensions. The reduced features are then plotted in a scatter plot, with each point colored according to its predicted class label.

