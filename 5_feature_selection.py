import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


def correlation_filter(datamatrix):
    df_all_data = datamatrix
    corr_matrix = df_all_data.iloc[:,0:(df_all_data.shape[1]-1)].corr()
    #sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)
    cormat_melted = []
    for i in range(len(corr_matrix)):
        f1 = corr_matrix.columns[i]
        for j in range(i,len(corr_matrix)):
            f2 = corr_matrix.columns[j]
            cormat_melted.append([f1, f2, corr_matrix.iloc[i,j]])
    cormat_melted = pd.DataFrame(cormat_melted,columns=['f1','f2','values'])
    cormat_melted.head(5)
    cormat_melted_filt = cormat_melted.loc[(cormat_melted['values']>=0.80) & (cormat_melted['values'] !=1.0)]
    todrop = set(cormat_melted_filt['f2'])
    df_all_data.drop(todrop, axis=1, inplace=True)

    print ("Applied Correlation filter >0.8: Removed " + str(len(todrop)) + " features from the dataset")
    return df_all_data


def get_feature_ranking(X_train,y_train):
    print ("feature ranking running....-> logistic regression")
    model1 = LogisticRegression()
    rfe = RFECV(estimator=model1, step=1, cv=StratifiedKFold(3),scoring='accuracy')
    rfe = rfe.fit(X_train,y_train )
    logr_ranking =[]
    for x,d in zip(rfe.ranking_,X_train.columns):
        logr_ranking.append([d,x])
    logr_ranking = pd.DataFrame(logr_ranking,columns=['features1','logr'])
    logr_ranking.sort_values('features1',inplace=True)

    print ("feature ranking running....-> XGBClassifier")
    model2 = XGBClassifier()
    rfe = RFECV(estimator=model2, step=1, cv=StratifiedKFold(3),scoring='accuracy')
    rfe = rfe.fit(X_train,y_train )
    xgboost_ranking =[]
    for x,d in zip(rfe.ranking_,X_train.columns):
        xgboost_ranking.append([d,x])
    xgboost_ranking = pd.DataFrame(xgboost_ranking,columns=['features2','xgboost'])
    xgboost_ranking.sort_values('features2',inplace=True)

    print ("feature ranking running....-> LinearSVC")
    model3 = LinearSVC()
    rfe = RFECV(estimator=model3, step=1, cv=StratifiedKFold(3),scoring='accuracy')
    rfe = rfe.fit(X_train,y_train )
    lsvc_ranking =[]
    for x,d in zip(rfe.ranking_,X_train.columns):
        lsvc_ranking.append([d,x])
    lsvc_ranking = pd.DataFrame(lsvc_ranking,columns=['features3','lsvc'])
    lsvc_ranking.sort_values('features3',inplace=True)

    feature_sum = logr_ranking['logr']+ xgboost_ranking['xgboost']+lsvc_ranking['lsvc']
    df_ranked =  pd.concat([logr_ranking['features1'],feature_sum],axis=1)
    df_ranked.sort_values(0,inplace=True)

    return df_ranked


def get_best_features(df_all_data):

    print ("correlation filtering....")
    df_all_data_s1 = correlation_filter(df_all_data)

    X_train = df_all_data_s1.iloc[:, 0:(df_all_data.shape[1] - 1)]
    y_train = df_all_data_s1.iloc[:, (df_all_data.shape[1] - 1)]

    print ("feature ranking started....")
    df_ranked = get_feature_ranking(X_train,y_train)
    ranked_list = list(df_ranked['features1'])
    X_train_sorted = X_train[ranked_list]

    print ("getting best k features....")
    accuracy = []
    model = LogisticRegression()
    for k in range(1,len(ranked_list)):
        x_train_filt = X_train_sorted.iloc[:,0:k]
        predicted = cross_val_predict(model, x_train_filt, y_train, cv=3)
        accuracy.append(metrics.accuracy_score(y_train, predicted))
    optimal_features_index = accuracy.index(max(accuracy)) + 1
    optimal_features = df_ranked.features1[0:optimal_features_index]
    optimal_features.to_csv("best_features.csv",index=False)
    print ("program completed..check best_features.csv ....")


# df_all_data = pd.read_csv("df_master_all_small.csv")
# get_best_features(df_all_data)
