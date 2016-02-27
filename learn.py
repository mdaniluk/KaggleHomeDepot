# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:52:39 2016

@author: dudu
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def load_data():
    df_x_train = pd.read_csv('data/train_features.csv')
    df_y_train = pd.read_csv('data/train_relevance.csv')
    df_x_test = pd.read_csv('data/test_features.csv')
    X = df_x_train.as_matrix()
    Y = df_y_train.as_matrix()
    Xt = df_x_test.as_matrix()
    return X[:,1:], Y[:,1:], Xt[:,1:]
    
def clamp_1_3(x):
    if x < 1.0:
        return 1.0
    elif x > 3.0:
        return 3.0
    else:
        return x
    
def save_submission(Yp):
    df = pd.read_csv('data/sample_submission.csv')
    df['relevance'] = Yp
    df.to_csv('data/my_submission.csv', index=False)
    
def learn_svr(X,Y,Xt):
    clf = SVR()
    clf.fit(X,Y)
    Yp = clf.predict(Xt)
    Yp_clamped = np.array([clamp_1_3(x) for x in Yp])
    return Yp_clamped
    
if __name__ == '__main__':
    X,Y,Xt = load_data()
    Yp = learn_svr(X,Y,Xt)
    save_submission(Yp)