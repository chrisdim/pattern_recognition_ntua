from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    fig, ax = plt.subplots()
    digit = np.reshape(X[index, :], (16,16))
    ax.axis('off')
    ax.imshow(digit)    


def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    digits = [0,1,2,3,4,5,6,7,8,9]
    indexes = []
    # Pick random samples from digits 0 - 9
    for digit in digits:
        index = np.random.choice(np.where(y == digit)[0])
        indexes.append(index)
    
    # Create Subplot
    fig, axs = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            toplot = np.reshape(X[indexes[5*i+j],:],(16,16))
            axs[i, j].imshow(toplot)
            axs[i,j].axis('off')
            axs[i,j].set_title(str(digits[5*i+j]))
    

def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    indexes = np.where(y==digit)[0]
    samples = []
    
    for i in indexes:
        samples.append(np.reshape(X[i,:], (16,16))[pixel])

    return np.expand_dims(np.mean(samples),0)


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    indexes = np.where(y==digit)[0]
    samples = []
    
    for i in indexes:
        samples.append(np.reshape(X[i,:], (16,16))[pixel])
    
    return np.var(samples)



def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    indexes = np.where(y==digit)[0]
    samples = []
    for i in indexes:
        samples.append(X[i,:])
    
    return np.mean(np.array(samples),axis = 0)
    

def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    indexes = np.where(y==digit)[0]
    samples = []
    for i in indexes:
        samples.append(X[i,:])
    
    return np.var(np.array(samples),axis = 0)


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    return euclidean_distances(s.reshape(1, -1),m.reshape(1, -1))


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    predictions = []
    for digit in X:
        distances = [] # 10 distances - one for each class
        for mean in X_mean:
            distances.append(euclidean_distance(digit, mean))
        predictions.append(np.argmin(distances))
        
    return np.array(predictions)



class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        digits = [0,1,2,3,4,5,6,7,8,9]
        mean_digits = []
        for d in digits:
            mean_digits.append(digit_mean(X, y, d))
        self.X_mean_ = np.array(mean_digits, dtype='object')
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        y_preds = euclidean_distance_classifier(X, self.X_mean_)
        return y_preds

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        y_preds = self.predict(X)
        accuracy = accuracy_score(y, y_preds)
        return accuracy


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, 
                         cv=KFold(n_splits=folds), 
                         scoring="accuracy")
    return scores

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    apriori = np.zeros(10)
    samples = np.shape(X)[0]
    
    for i in range(samples):
        apriori[int(y[i])] = 1+apriori[int(y[i])]
    
    apriori = apriori/samples #probability N_occ/N_samples
    return apriori
        
def Gaussian_prob(x, mean, var):
    """
    Calculate Normal Distribution ~N(mean,var)
    of feature x
    """
    return np.exp(-(x-mean)**2/(2*var))/np.sqrt(2*np.pi*var)


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.X_mean_ = None
        self.X_var_ = None
        self.apriori = None


    def fit(self, X, y, smooth = 1e-5):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_, self.X_var_ based on the mean
        feature values and variance correspondingly in X for each class.
        
        Calculates self.apriori, thw a-priori
        probability of each class

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        
        self.X_var_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        
        self.apriori becomes a numpy.array of shape
        (n_classes)

        fit always returns self.
        """
        digits = [i for i in range(1,10)]
        mean_digits = []
        var_digits = []
        for d in digits:
            mean_digits.append(digit_mean(X, y, d))
            var_digits.append(digit_variance(X,y,d))
            
        self.X_mean_ = np.array(mean_digits, dtype='object')
        
        if self.use_unit_variance == True:
            self.X_var_ = np.ones(self.X_mean_.shape)
        else:
            self.X_var_ = np.array(var_digits, dtype = 'object') + smooth
            
        self.apriori = calculate_priors(X,y)
        return self


    def predict(self, X):
        """
        Make predictions for X based on Maximum a-posteriori (MAP) Rule
        P(C|x1x2...xn) = argmax(P(x1x2...xn|C)P(C))
        """
        n_features = np.shape(X)[1]
        n_classes = 9
        predictions = []

        for sample in X:
            aposteriors = []
            
            for i in range(n_classes):   
                """
                Calculate Likelihood Probability P(x|C)
                Naive Bayes assumes i.i.d features:
                P(x1...xn|C) = P(x1|C)...P(xn|C)
                """
                obsrv = 1
                for j in range(n_features):
                    # Normal Distributions for each feature
                    obsrv = obsrv * Gaussian_prob(sample[j], self.X_mean_[i,j], self.X_var_[i,j])
                    
                # Multiply with a-priori probability P(C)
                aposteriors.append(obsrv*self.apriori[i+1])
            
            # Keep argmax of a-posteriori Probabilities as prediction
            predictions.append(np.argmax(aposteriors)+1)
        
        return np.array(predictions)
                  

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        y_preds = self.predict(X)
        accuracy = accuracy_score(y, y_preds)
        return accuracy

##############################################################################

# Define Dataset Class of Digit Samples
class DigitDataset(Dataset):
    def __init__(self, X, y, trans=None):
        # all the available data are stored in a list
        self.data = list(zip(X, y))
        # we optionally may add an augmentation
        self.trans = trans
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]

class LinearWActivation(nn.Module): 
    def __init__(self, in_features, out_features, activation='sigmoid'):
        super(LinearWActivation, self).__init__()
        # nn.Linear is just a matrix of [in_features, out_features] randomly initialized
        self.f = nn.Linear(in_features, out_features)
        if activation == 'sigmoid':
            self.a = nn.Sigmoid()
        elif activation == 'tanh':
            self.a = nn.Tanh()
        else:
            self.a = nn.ReLU()
            
    # the forward pass of info through the net
    def forward(self, x): 
        return self.a(self.f(x.float()))

# Class of a Neural Net Model of given layers
class NeuralNet(nn.Module): 
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
      '''
      Args:
        layers (list): a list of the number of consecutive layers
        n_features (int):  the number of input features
        n_classes (int): the number of output classes
        activation (str): type of non-linearity to be used
      '''
      super(NeuralNet, self).__init__()
      layers_in = [n_features] + layers # list concatenation
      layers_out = layers + [n_classes]
      # loop through layers_in and layers_out lists
      self.f = nn.Sequential(*[
          LinearWActivation(in_feats, out_feats, activation=activation)
          for in_feats, out_feats in zip(layers_in, layers_out)
      ])
      # final classification layer is always a linear mapping
      self.clf = nn.Linear(n_classes, n_classes)
                
    def forward(self, x): # again the forwrad pass
      # apply non-linear composition of layers/functions
      y = self.f(x)
      # return an affine transformation of y <-> classification layer
      return self.clf(y)
    
class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, n_features, n_classes, activation = 'sigmoid'):
        '''
        Args:
          layers (list): a list of the number of consecutive layers
          n_features (int):  the number of input features
          n_classes (int): the number of output classes
          activation (str): type of non-linearity to be used
        '''
        # Initialize model, criterion and optimizer
        self.model = NeuralNet(layers=layers, n_features = n_features, n_classes = n_classes,
                               activation = activation)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)

    def fit(self, X, y, EPOCHS, BATCH_SIZE):
        # Split Train Dataset to Train and Validation Sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
        
        # Define Datasets
        train_data = DigitDataset(X_train, y_train)
        val_data = DigitDataset(X_val, y_val)
        
        # Define DataLoaders
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
        
        loss_train = []
        loss_val = []
        
        # Train model on Train Dataset
        self.model.train()                                  # gradients "on"
        correct_train = 0
        total_train = 0
        for epoch in range(EPOCHS):                         # loop through dataset
            running_average_loss = 0
            for i, data in enumerate(train_loader):         # loop through batches
                X_batch, y_batch = data                     # get the features and labels
                self.optimizer.zero_grad() 
                out = self.model(X_batch)                   # forward pass
            
                _, predicted = torch.max(out.data, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()
                loss = self.criterion(out, y_batch.long())  # compute per batch loss 
                loss.backward()                             # compute gradients based 
                                                            # on the loss function
                self.optimizer.step()                       # update weights 
                
                running_average_loss += loss.detach().item()
                
                if i % 50 == 0:
                    print("Epoch: {} \t Batch: {} \t Training Loss {}".format(epoch+1, i, float(np.round(running_average_loss / (i + 1),5))))
            
            
            # Evaluate on Validation Dataset
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                running_average_val_loss = 0
                for i, data in enumerate(val_loader):
                    X_batch, y_batch = data
                    out = self.model(X_batch)
                    val_loss = self.criterion(out, y_batch.long())
                    val, y_pred = out.max(1)
                    total_val += y_batch.size(0)
                    correct_val += (y_pred == y_batch).sum().item()
                    
                    running_average_val_loss += val_loss.detach().item()
            
            loss_train.append(running_average_loss)
            loss_val.append(running_average_val_loss)
            
            print('Accuracy in Train Set: %f %%' % ( 100 * correct_train / total_train))
            print('Accuracy in Validation Set: %f %%' % ( 100 * correct_val / total_val))
        
        # Plot Train and Validation Losses of Training Process
        plt.figure()
        x = np.arange(1,EPOCHS+1,1)
        markerline1, stemlines, _ = plt.stem(x, np.array(loss_train),label="Training Loss")
        plt.setp(markerline1, 'markerfacecolor', 'blue')
        markerline2, stemlines, _ = plt.stem(x, np.array(loss_val),label="Validation Loss")
        plt.setp(markerline2, 'markerfacecolor', 'orange')
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.show()
        return self

    def predict(self, X):
        # Wrap X in a Test Loader of 1 batch
        test_loader = DataLoader(X, batch_size=X.shape[0])
        
        self.model.eval() # turns off batchnorm/dropout
        n_samples = 0
        predictions = []
        with torch.no_grad():                       # no gradients required
            for i, data in enumerate(test_loader):
                X_batch = data                      # test data
                out = self.model(X_batch)           # get net's predictions
                val, y_pred = out.max(1)            # argmax since output is a prob distribution
                predictions.append(y_pred)          # collect batch predictions
                n_samples += X_batch.size(0)
                
        predictions = torch.cat(predictions)        # concatenate all batches
        return np.array(predictions)

    def score(self, X, y):
        # Return accuracy score.
        y_preds = self.predict(X)
        accuracy = accuracy_score(y, y_preds)
        return accuracy

###############################################################################
      
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel='rbf')
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)


def evaluate_knn_classifier(X, y, folds=5, neighbors = 5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = GaussianNB()
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
    
    
def evaluate_custom_nb_classifier(X, y, folds=5, use_unit_variance=False):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = CustomNBClassifier(use_unit_variance)
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
    
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = EuclideanDistanceClassifier()
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
    
def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError    

    

def evaluate_voting_classifier(classifiers, X, y, folds=5, vote_method = 'hard'):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = VotingClassifier(estimators = classifiers, voting = vote_method)
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
    
    

def evaluate_bagging_classifier(base_clf, X, y, estimators, folds=5 ):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = BaggingClassifier(base_clf, n_estimators=estimators)
    clf.fit(X, y)
    
    scores = evaluate_classifier(clf, X, y, folds)
    
    return np.mean(scores)
