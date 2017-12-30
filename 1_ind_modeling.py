import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


df_all_data = pd.read_csv("../df_master_all.csv",sep=',',header=0)
len(df_all_data)
df_all_data.shape
df_all_data.head(2)
df_all_data.isnull().sum()

df_data = df_all_data.loc[:,'87_mean':'IsReadmitted']


xtrain = df_all_data.loc[:,'87_mean':'insurance_Self Pay']
ytrain = pd.DataFrame(df_all_data.loc[:,'IsReadmitted'])

ytrain[ytrain['IsReadmitted']>0].count() / ytrain[ytrain['IsReadmitted']<1].count() * 100



clf = GaussianNB()
clf.fit(xtrain, ytrain.values)
yhat = pd.DataFrame(clf.predict(xtrain), columns=['predict'])
np.sum([1 if x==y else 0 for x,y in zip(ytrain.values,yhat.values)])/float(len(yhat))



# data is an array with our already pre-processed dataset examples
kf = KFold(n_splits=3, random_state=0)
result = []
for train, test in kf.split(df_data):
    train_data = df_data.iloc[train,:]
    test_data =  df_data.iloc[test,:]

    trainx = train_data.loc[:,'87_mean':'insurance_Self Pay']
    trainy =   train_data.loc[:,'IsReadmitted']

    testx = test_data.loc[:, '87_mean':'insurance_Self Pay']
    testy = test_data.loc[:, 'IsReadmitted']

    clf = GaussianNB()
    clf.fit(trainx, trainy.values)

    yhat = pd.DataFrame(clf.predict(testx), columns=['predict'])
    result.append(np.sum([1 if x == y else 0 for x, y in zip(testy.values, yhat.values)]) / float(len(yhat)))
    print result


print np.sum(result)/len(result)





################## random forest
'''
There are 4 hyperparameters required for a Random Forest classifier;

    the number of trees in the forest (n_estimators).
    the number of features to consider at each split. By default: square root of total number of features (max_features).
    the maximum depth of a tree i.e. number of nodes (max_depth).
    the minimum number of samples required to be at a leaf node / bottom of a tree (min_samples_leaf).

'''


xtrain = df_all_data.loc[:,'87_mean':'insurance_Self Pay']
ytrain = pd.DataFrame(df_all_data.loc[:,'IsReadmitted'])

seed =123

# RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
rfc = RandomForestClassifier(n_jobs=-1, oob_score=True, max_depth=10, max_features='sqrt', min_samples_leaf=10)

# Range of `n_estimators` values to explore.
n_estim = filter(lambda x: x % 2 == 0, list(range(10, 20)))

cv_scores = []

for i in n_estim:
    rfc.set_params(n_estimators=i)
    kfold = KFold(n_splits=3, random_state=seed)
    scores = cross_val_score(rfc, xtrain, ytrain.values, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean() * 100)

optimal_n_estim = n_estim[cv_scores.index(max(cv_scores))]
print "The optimal number of estimators is %d with %0.1f%%" % (optimal_n_estim, cv_scores[optimal_n_estim])

plt.plot(n_estim, cv_scores)
plt.xlabel('Number of Estimators')
plt.ylabel('Train Accuracy')
plt.show()

############## optimize all three parameters

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score=True)

# Use a grid over parameters of interest
param_grid = {
    "n_estimators": [9, 18, 27, 36, 45, 54, 63],
    "max_depth": [1, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10]}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
CV_rfc.fit(xtrain, df_all_data.IsReadmitted)
print CV_rfc.best_params_