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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from lib import *

plt.close('all')

# Read train and test data files
train = np.loadtxt('data/train.txt')
test = np.loadtxt('data/test.txt')

X_train = train[:, 1:]
X_test = test[:, 1:]
y_train = train[:, 0]
y_test = test[:, 0]

# Step 14: Calculate a-priori Probabilities of train data set
apriori = calculate_priors(X_train,y_train)

# Step 15: Implement Naive Bayes Classifier

# (15a) NB Classifier as scikit-learn Estimator
myNB1 = CustomNBClassifier(use_unit_variance=False)
myNB1.fit(X_train, y_train)

# (15b) Evaluate my Naive Bayes Classifier
myscore1 = myNB1.score(X_test, y_test)
print('My Gaussian NB Classifier Test Score: '+str(np.round(myscore1,4)))

# (15c) Compare to scikit-learn method GaussianNB
sk_NB = GaussianNB()
sk_NB.fit(X_train, y_train)
sk_score = sk_NB.score(X_test, y_test)
print('scikit-learn GaussianNB Classifier Test Score: '+str(np.round(sk_score,4)))

# Step 16: Implement Naive Bayes Classifier with Unit Variance

# (16a) NB Classifier as scikit-learn Estimator
myNB2 = CustomNBClassifier(use_unit_variance=True)
myNB2.fit(X_train, y_train)

# (16b) Evaluate my Naive Bayes Classifier
myscore2 = myNB2.score(X_test, y_test)
print('My Gaussian NB Classifier Test Score (Unit Variance): '+str(np.round(myscore2,4)))


# Step 17: Compare scores of different classifiers
print('\n========== Compare Classifiers by Score ==========')

print('1. Custom NB Classifier Score (No Unit Variance): '+str(np.round(evaluate_custom_nb_classifier(X_train, y_train, folds=5,use_unit_variance=False),4)))
print('2. Custom NB Classifier Score (Unit Variance): '+str(np.round(evaluate_custom_nb_classifier(X_train, y_train, folds=5,use_unit_variance=True),4)))
print('3. sklearn Gaussian NB Classifier Score: '+str(np.round(evaluate_sklearn_nb_classifier(X_train, y_train, folds=5),4)))
print('4. Euclidean Distance Classifier Score: '+str(np.round(evaluate_euclidean_classifier(X_train, y_train, folds=5),4)))
print('5. SVM Classifier Score (Linear Kernel): '+str(np.round(evaluate_linear_svm_classifier(X_train, y_train, folds=5),4)))
print('6. SVM Classifier Score (rbf kernel): '+str(np.round(evaluate_rbf_svm_classifier(X_train, y_train, folds=5),4)))
print('7. KNN Classifier (3 neighbors) Score: '+str(np.round(evaluate_knn_classifier(X_train, y_train, folds=5, neighbors = 3),4)))
print('8. KNN Classifier (5 neighbors) Score: '+str(np.round(evaluate_knn_classifier(X_train, y_train, folds=5, neighbors = 5),4)))

print('==================================================\n')

# Step 18: Ensemble/Bagging/Boosting

# At first, we find the top mispredicted digits for each classifier
def top_mispredict(clf, Xtrain, ytrain, Xtest, ytest):
    clf.fit(Xtrain, ytrain) # train model
    y_preds = clf.predict(Xtest) # evaluate on test set
    
    # Count Mispredictions by digit
    n_mispredicts = np.zeros(10)
    for i in range(len(y_test)):
        if y_test[i] != y_preds[i]:
            n_mispredicts[int(y_test[i])] += 1
    
    # return most mispredicted digits
    top_mispredicted_digits = np.argsort(n_mispredicts)[::-1]
    
    return top_mispredicted_digits

classifiers = [CustomNBClassifier(use_unit_variance=False),
               CustomNBClassifier(use_unit_variance=True), GaussianNB(),
               EuclideanDistanceClassifier(), SVC(kernel='linear'), SVC(kernel='rbf'),
               KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5)]
        
print('\n========== Top Mispredicted Digits by Classifier ==========')
print('1. Custom NB Classifier (Not Unit Variance):')
print(top_mispredict(classifiers[0], X_train, y_train, X_test, y_test))
print('2. Custom NB Classifier (Unit Variance):')
print(top_mispredict(classifiers[1], X_train, y_train, X_test, y_test))
print('3. sklearn NB Classifier:')
print(top_mispredict(classifiers[2], X_train, y_train, X_test, y_test))
print('4. Euclidean Distance Classifier:')
print(top_mispredict(classifiers[3], X_train, y_train, X_test, y_test))
print('5. SVM Classifier (Linear Kernel):')
print(top_mispredict(classifiers[4], X_train, y_train, X_test, y_test))
print('6. SVM Classifier (rbf Kernel):')
print(top_mispredict(classifiers[5], X_train, y_train, X_test, y_test))
print('7. KNN Classifier (3 Neighbors):')
print(top_mispredict(classifiers[6], X_train, y_train, X_test, y_test))
print('8. KNN Classifier (5 Neighbors):')
print(top_mispredict(classifiers[7], X_train, y_train, X_test, y_test))
print('============================================================\n')


# (18a) Experiment with VotingClassifier
estimators1 = [('svm_lin',SVC(kernel='linear',probability=True)),
               ('knn_3',KNeighborsClassifier(n_neighbors=3)),
               ('knn_5',KNeighborsClassifier(n_neighbors=5))]

estimators2 = [('svm_lin',SVC(kernel='linear',probability=True)),
               ('svm_rbf',SVC(kernel='rbf',probability=True)),
               ('knn_3',KNeighborsClassifier(n_neighbors=3))]

ensemb1 = evaluate_voting_classifier(estimators1, X_train, y_train,5,'hard')
ensemb2 = evaluate_voting_classifier(estimators1, X_train, y_train,5,'soft')

ensemb3 = evaluate_voting_classifier(estimators2, X_train, y_train,5,'hard')
ensemb4 = evaluate_voting_classifier(estimators2, X_train, y_train,5,'soft')

print('\n========== Voting Classifier Experiments ==============')
print('1st Experiment - Score: '+str(np.round(ensemb1,4)))
print('2nd Experiment - Score: '+str(np.round(ensemb2,4)))
print('3rd Experiment - Score: '+str(np.round(ensemb3,4)))
print('4th Experiment - Score: '+str(np.round(ensemb4,4)))
print('=======================================================\n')


# (18b) Experiment with Bagging

# Use Classifier with highest score: SVM with rbf kernel
base_clf = SVC(kernel='rbf', probability=True)
bag1 = evaluate_bagging_classifier(base_clf, X_train, y_train, estimators=5, folds=5 )
bag2 = evaluate_bagging_classifier(base_clf, X_train, y_train, estimators=10, folds=5 )
bag3 = evaluate_bagging_classifier(base_clf, X_train, y_train, estimators=15, folds=5 )

print('\n========== Bagging Classifier Experiments ==============')
print('1st Experiment - Score: '+str(np.round(bag1,4)))
print('2nd Experiment - Score: '+str(np.round(bag2,4)))
print('3rd Experiment - Score: '+str(np.round(bag3,4)))
print('=======================================================\n')


# Step 19: (BONUS) PyTorch Neural Networks
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

EPOCHS = 50
BATCH_SIZE = 64


layers = [[100,100], [50,50], [100], [50]]
activations = ['sigmoid', 'tanh', 'ReLU']

accuracies = []
for layer in layers:
    for func in activations:
        # Define NN Model
        model = PytorchNNModel(layers = layer, n_features=X_train.shape[1], 
                   n_classes = len(set(y_train)), activation = func)
    
        # Train Model on Train Set and Evaluate on Validation Set
        model.fit(X_train, y_train, EPOCHS, BATCH_SIZE)

        # Evaluate Accuracy of model on Test Set
        acc = model.score(X_test, y_test)
        accuracies.append(acc)

# Print Results of Experiment
i = 0
print('')
for layer in layers:
    for func in activations:
        print('# Experiment Number '+str(i+1))
        print('Hidden Layers Dimensions: ', layer)
        print('Activation Function: ', func)
        print('Test Accuracy: '+ str(np.round(accuracies[i], 5))+'\n')
        i+=1
