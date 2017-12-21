import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import math
import collections
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


df_all_data = pd.read_csv("../df_master_all.csv",sep=',',header=0)
len(df_all_data)
df_all_data.shape
df_all_data.head(2)
df_all_data.isnull().sum()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(df_all_data.loc[:,'age':'51301_var'], df_all_data.loc[:,'IsReadmitted'])
yhat = pd.DataFrame(clf.predict(df_all_data.loc[:,'age':'51301_var']), columns=['predict'])


y_train = pd.DataFrame(df_all_data.loc[:,'IsReadmitted'])

match = 0.0
for m in range(0, len(yhat)):
    if yhat.iloc[m].values == y_train.iloc[m].values: match += 1

match = match / len(yhat)
print match

