import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
seed=123



df_all_data = pd.read_csv("../df_master_all.csv",sep=',',header=0)
df_all_data = df_all_data.iloc[0:1000,:]
df_data = df_all_data.loc[:,'87_mean':'IsReadmitted']
xtrain = df_all_data.loc[:,'87_mean':'insurance_Self Pay']
ytrain = df_data.IsReadmitted


# with cross validation k =3
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
    #print result


print "Naive bayes 3-fold cv result", np.sum(result)/len(result)





############## Random Forest : optimize all three parameters

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score=True)

#Use a grid over parameters of interest
param_grid = {
    "n_estimators": [9, 18, 27, 36, 45, 54, 63],
    "max_depth": [1, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10]}


CV_rfc = model_selection.GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
CV_rfc.fit(xtrain, ytrain)
print "random forest best parameters", CV_rfc.best_params_
print "random forest best score", CV_rfc.score(xtrain, ytrain)


