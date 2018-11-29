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

def min_max_scaler(train_data):
    from sklearn import preprocessing
    XX = preprocessing.MinMaxScaler().fit(train_data)  
    train_data = XX.transform(train_data) 
    return train_data, XX

def train_test_evalation_split(data, label): 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0, shuffle=True)
    return X_train, X_test, y_train, y_test

def new_model():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.constraints import maxnorm
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
    from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU, MaxPooling1D, Conv2D
    model = Sequential()
    model.add(Reshape(target_shape=(64, 64, 1), input_shape=(64, 64)))
    model.add(Conv2D(32, (20, 20), input_shape=(64, 64, 1), data_format='channels_last')) 
    model.add(Activation(activation='relu'))  # 去掉batch normalization 和relu效果反而好很多，对于浅层模型
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(NB_CLASS, activation='softmax'))
    # One layer
    model.summary()
    return model

# read explore
train_org = pd.read_csv(data_path + 'train.csv')
test_org = pd.read_csv(data_path + 'test.csv')
data = train_org.iloc[:, 1:-1]
label = train_org.iloc[:, -1]
test = test_org.iloc[:, 1:]

# preprocess
cnn_data, XX = min_max_scaler(data)
cnn_test = XX.transform(test)
X, y = cnn_data, label
X = X.reshape(-1, 64, 64)
test = cnn_test.reshape(-1, 64, 64)

# compile model
cnn_model = new_model()
cnn_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])   

# cross validation
kf = StratifiedShuffleSplit(n_splits=3, random_state=1)
auc_list3 = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    np.random.seed(1)
    cnn_model.fit(X_train, y_train, nb_epoch=3, batch_size=256, validation_split=0.1)
    scores = cnn_model.evaluate(X_test, y_test, verbose=2)
    auc_list3.append(scores[1]) 
print(auc_list3)

cnn_model.save(data_path + 'cnn_model')
from keras.models import load_model
cnn_model = load_model(data_path + 'cnn_model')

# save result
pre = cnn_model.predict(test)
pred = []
for item in pre:
    pred.append(np.argmax(item))
pred = np.array(pred)

pred = pred.reshape(-1, 1)
result = np.hstack([test_org.iloc[:, 0].values.reshape(-1, 1), pred])
result = pd.DataFrame(result, columns=['id', 'categories'])
result.to_csv(data_path + '017033910034_Yanglanqing.csv', index=False)