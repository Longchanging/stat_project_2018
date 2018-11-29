# coding:utf-8
'''
@time:    Created on  2018-11-28 17:13:18
@author:  Lanqing
@Func:    Project of stat
'''
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from collections import Counter
import sklearn.metrics
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
import keras.preprocessing

saved_dimension_after_pca = 20
NB_CLASS = 12
data_path = 'E:/DATA/stat_project_2018/'

def PCA(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=saved_dimension_after_pca)
    X = pca.fit_transform(X)
    return X, pca

def min_max_scaler(train_data):
    from sklearn import preprocessing
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    return train_data, XX

def train_test_evalation_split(data, label): 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, shuffle=True)
    return X_train, X_test, y_train, y_test

def random_forest_classifier(trainX, trainY):  
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300, criterion='entropy')  # , max_depth=10)
    model.fit(trainX, trainY)
    return model

# read explore
train_org = pd.read_csv(data_path + 'train.csv')
test_org = pd.read_csv(data_path + 'test.csv')
data = train_org.iloc[:, 1:-1]
label = train_org.iloc[:, -1]
test = test_org.iloc[:, 1:]
print(Counter(list(label.values)))

# preprocess
data, pca = PCA(data)
test = pca.transform(test)
data, XX = min_max_scaler(data)
test = XX.transform(test)
print(data.shape, test.shape)

# train test
X_train, X_test, y_train, y_test = train_test_evalation_split(data, label)
X, y = data, label 

# use RF, 5-fold
kf = StratifiedShuffleSplit(n_splits=5, random_state=1)
auc_list1 = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf = random_forest_classifier(X_train, y_train)
    pred_train = rf.predict(X_test)
    auc1 = sklearn.metrics.accuracy_score(y_test, pred_train)
    auc_list1.append(auc1)
    cfr = sklearn.metrics.classification_report(y_test, pred_train)
print(auc_list1, np.mean(auc_list1))
   
# use XgBoost, 5-fold
kf = StratifiedShuffleSplit(n_splits=5, random_state=1)
auc_list2 = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    param = {
             'n_estimators':200,
              'bst:max_depth':7,
              'bst:eta':0.01,
              'silent':1,
              # 'learning_rate' : 0.1,
              # 'max_depth' : 6,
              'num_feature':10,
              # 'gamma': 0.1,
             'alpha':0.1,
             # 'seed': 1,
             # 'subsample':0.7,
             'objective':'multi:softmax',
             'num_class':12}
    import xgboost as xgb
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    bst = xgb.train(param, data_dmatrix, 30)
    test_dmatrix = xgb.DMatrix(data=X_test)
    preds = bst.predict(test_dmatrix)
    preds = np.array(preds)
    auc2 = sklearn.metrics.accuracy_score(y_test, preds)
    cfr = sklearn.metrics.classification_report(y_test, preds)
    auc_list2.append(auc2)
print(auc_list2, np.mean(auc_list2))

# save result
best_model = rf
pred = best_model.predict(test)
pred = pred.reshape(-1, 1)
result = np.hstack([test_org.iloc[:, 0].values.reshape(-1, 1), pred])
result = pd.DataFrame(result, columns=['id', 'categories'])
result.to_csv(data_path + '017033910034_Yanglanqing.csv', index=False)
