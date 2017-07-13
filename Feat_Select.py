__author__ = 'Sumit'
import csv
import os
import zipfile
import numpy
from sklearn import model_selection
from sklearn import feature_selection as fs
from sklearn import feature_extraction as fe
from scipy import stats
from matplotlib.mlab import PCA as mlabPCA
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import tree
from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from skfeature.function.similarity_based import fisher_score

import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
class Feat_Select:
    num_fea = 5
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    def Fisher_Score(self):
        score = fisher_score.fisher_score(X_train, y_train)
        idx = fisher_score.feature_ranking(score)
