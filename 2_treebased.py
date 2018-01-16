## modified from https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt


df_all_data = pd.read_csv("../df_master_all.csv",sep=',',header=0)
df_all_data = df_all_data.iloc[0:1000,:]
len(df_all_data)
df_all_data.shape
df_all_data.head(2)
df_all_data.isnull().sum()

xtrain = df_all_data.loc[:,'87_mean':'insurance_Self Pay']
ytrain = df_all_data.IsReadmitted

#ytrain[ytrain['IsReadmitted']>0].count() / ytrain[ytrain['IsReadmitted']<1].count() * 100



seed =123


#### bagged decision trees

kfold = model_selection.KFold(n_splits=3, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold)
print results.mean()





################## random forest
'''
There are 4 hyperparameters required for a Random Forest classifier;

    the number of trees in the forest (n_estimators).
    the number of features to consider at each split. By default: square root of total number of features (max_features).
    the maximum depth of a tree i.e. number of nodes (max_depth).
    the minimum number of samples required to be at a leaf node / bottom of a tree (min_samples_leaf).

'''




# RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
rfc = RandomForestClassifier(n_jobs=-1, oob_score=True, max_depth=10, max_features='sqrt', min_samples_leaf=10)

# Range of `n_estimators` values to explore.
n_estim = filter(lambda x: x % 2 == 0, list(range(10, 16)))

cv_scores = []

for i in n_estim:
    rfc.set_params(n_estimators=i)
    kfold = model_selection.KFold(n_splits=3, random_state=seed)
    scores = model_selection.cross_val_score(rfc, xtrain, ytrain, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean() * 100)

print cv_scores


############## optimize all three parameters

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score=True)

# Use a grid over parameters of interest
param_grid = {
    "n_estimators": [9, 18, 27, 36, 45, 54, 63],
    "max_depth": [1, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10]}

CV_rfc = model_selection.GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
CV_rfc.fit(xtrain, ytrain)
print CV_rfc.best_params_


########## another bagging approach is ExtraTreesClassifier

############ boosting


###adaboost


num_trees = 10
kfold = model_selection.KFold(n_splits=3, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold)
print results.mean()


## gradient boosting

num_trees = 10
kfold = model_selection.KFold(n_splits=3, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold)
print results.mean()



## voting

kfold = model_selection.KFold(n_splits=3, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, xtrain, ytrain, cv=kfold)
print results.mean()
