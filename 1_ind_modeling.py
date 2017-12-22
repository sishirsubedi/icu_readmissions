import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


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

