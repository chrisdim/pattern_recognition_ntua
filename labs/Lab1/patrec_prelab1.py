"""
ECE NTUA - Flow S: Signals, Control, Robotics
Pattern Recognition - Fall 2021
1st Lab Exercise: Visual Digit Recognition

Author: Christos Dimopoulos - 03117037
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from lib import *

plt.close('all')

# Step 1: Read train and test datafiles
train = np.loadtxt('data/train.txt')
test = np.loadtxt('data/test.txt')

X_train = train[:, 1:]
X_test = test[:, 1:]
y_train = train[:, 0]
y_test = test[:, 0]

# Step 2: Plot digit number 131 of Train Set
show_sample(X_train, 131)

# Step 3: Plot random samples of train dataset from digits 0-9
plot_digits_samples(X_train, y_train)

# Step 4: Calculate mean value of pixel (10,10) for digit 0 of Train Set.
mean1 = digit_mean_at_pixel(X_train, y_train, digit = 0, pixel=(10, 10))
print("Mean Value of pixel (10,10) for Digits 0: "+str(mean1))

# Step 5: Calculate variance of pixel (10,10) for digit 0 of Train Set.
var1 = digit_variance_at_pixel(X_train, y_train, digit = 0, pixel=(10, 10))
print("Variance of pixel (10,10) for Digits 0: "+str(var1))

# Step 6: Calculate Mean Value & Variance of all 256 Features for Digit 0
mean_zero = digit_mean(X_train, y_train, digit = 0)
var_zero = digit_variance(X_train, y_train, digit = 0)

# Step 7: Plot digit 0 based on Mean Value of its 256 Features
fig, ax = plt.subplots()
ax.set_title('Mean Value of Features for Digit 0')
ax.axis('off')
ax.imshow(np.reshape(mean_zero,(16,16)))

# Step 8: Plot digit 0 based on Variance of its 256 Features
fig, ax = plt.subplots()
ax.axis('off')
ax.set_title('Variance of Features for Digit 0')
ax.imshow(np.reshape(var_zero,(16,16)))

# Step 9:
# (9a) Calculate Mean Values and Variances for all digits 0-9
digits = [0,1,2,3,4,5,6,7,8,9]
mean_digits = []
var_digits = []
for d in digits:
    mean_digits.append(digit_mean(X_train, y_train, d))
    var_digits.append(digit_variance(X_train, y_train, d))

# (9b) Plot Mean Values of all digits
fig, axs = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        toplot = np.reshape(mean_digits[5*i+j], (16,16))
        axs[i, j].imshow(toplot)
        axs[i,j].axis('off')
        axs[i,j].set_title(str(digits[5*i+j]))
        
# Step 10: Classify Digit 101 Based on Euclidean Distance

# Plot Digit Sample 101
digit_101 = X_test[101,:]
show_sample(X_test, 101)

distances = []
for mean in mean_digits:
    distances.append(euclidean_distance(digit_101, mean))

y_pred = np.argmin(distances) # Label as minimum distance digit
print('\nGold Label of Test Digit 101: '+ str(int(y_test[101]))+'\nPredicted Label of Test Digit 101: '+str(y_pred))

# Step 11: Classify Test Set using Euclidean Distance
# (11a) Classify each Digit of Test Set
y_preds = euclidean_distance_classifier(X_test, mean_digits)

# (11b) Evaluate Accuracy of Classification
print('\nAccuracy with Euclidean Distance: '+str(accuracy_score(y_test, y_preds)))

# Step 12: Implement Euclidean Classifier as sklearn Estimator
euclidean_clf = EuclideanDistanceClassifier()
euclidean_clf.fit(X_train, y_train)

# Step 13:
# (13a) Calculate 5-fold-cross-validation score
cv_scores = evaluate_classifier(euclidean_clf, X_train, y_train)
cv_mean_score = np.mean(cv_scores)
print('\nMean Cross-Validation Accuracy Score: '+str(cv_mean_score))
print('Cross-Validation Error: %f +-%f' % (1. - np.mean(cv_scores), np.std(cv_scores))+'\n')

# (13b) Function from Python Lab Course - Plot Decision Surface
def plot_clf(clf, X, y, labels):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision Surfaces of Euclidean Classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    colors = ['blue', 'red', 'green', 'yellow', 'magenta', 'orange', 
              'black', 'white', 'gray', 'brown']
    for i in range(10):
        ax.scatter(
        X0[y == i], X1[y == i],
        c=colors[i], label=labels[i],
        s=60, alpha=0.9, edgecolors='k')
    
    ax.set_xlabel('1st Principal Component')
    ax.set_ylabel('2nd Principal Component')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
    
# Apply PCA to downgrade 256-features to only 2-dimensions
pca = PCA(n_components=2)
pca.fit(X_train)

# Apply PCA on train and test set
X_train_2d = pca.transform(X_train)
X_test_2d = pca.transform(X_test)

# Train Classifier on 2D-Features & Plot Decision Hyperplanes
clf_2d = EuclideanDistanceClassifier()
clf_2d.fit(X_train_2d, y_train)
plot_clf(clf_2d, X_test_2d, y_test, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# (13c) Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    EuclideanDistanceClassifier(), X_train, y_train, cv=5, 
    train_sizes=np.linspace(.25, 1.0, 5))

# Function from Python Lab Course - Plot Learning Curve
def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.8, 1))