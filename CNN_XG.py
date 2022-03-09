import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Model
from keras.layers import Input,BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D,MaxPooling1D,AveragePooling1D
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd
import keras
from sklearn.svm import SVR
import scipy.stats as stats
from keras import layers
from keras.models import Sequential
from matplotlib import  pyplot as plt
from keras.utils import  to_categorical
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from xgboost import plot_importance
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

# 对sgRNA进行编码
def grna_preprocess(lines):
    length = 23
    data_n = len(lines)
    seq = np.zeros((data_n, length, 4), dtype=int)  # data_n层 length*4 的0矩阵

    for l in range(data_n):
        data = lines[l]
        seq_temp = data
        for i in range(length):
            if seq_temp[i] in "Aa":
                seq[l, i, 0] = 1
            elif seq_temp[i] in "Cc":
                seq[l, i, 1] = 1
            elif seq_temp[i] in "Gg":
                seq[l, i, 2] = 1
            elif seq_temp[i] in "Tt":
                seq[l, i, 3] = 1
    return seq


def epi_preprocess(lines):
    length = 23
    data_n = len(lines)
    epi = np.zeros((data_n, length), dtype=int)
    for l in range(data_n):
        data = lines[l]
        epi_temp = data
        for i in range(length):
            if epi_temp[i] in "A":
                epi[l, i] = 1
            elif epi_temp[i] in "N":
                epi[l, i] = 0
    return epi


def preprocess(file_path, usecols):
    data = pd.read_csv(file_path, usecols=usecols)
    data = np.array(data)
    ctcf, dnase, h3k4me3, rrbs = epi_preprocess(data[:, 0]), epi_preprocess(data[:, 1]), epi_preprocess(
        data[:, 2]), epi_preprocess(data[:, 3])
    epi = []
    for i in range(len(data)):
        ctcf_t, dnase_t, h3k4me3_t, rrbs_t = pd.DataFrame(ctcf[i]), pd.DataFrame(dnase[i]), pd.DataFrame(
            h3k4me3[i]), pd.DataFrame(rrbs[i])
        epi_t = pd.concat([ctcf_t, dnase_t, h3k4me3_t, rrbs_t], axis=1)  # 当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
        epi_t = np.array(epi_t)
        epi.append(epi_t)
    epi = np.array(epi)
    return epi


def load_data(train_file):
    train_data = pd.read_csv(train_file, usecols=[4, 9])
    train_data = np.array(train_data)
    train_seq, train_y = train_data[:, 0], train_data[:, 1]
    train_seq = grna_preprocess(train_seq)

    train_epi = preprocess(train_file, [5, 6, 7, 8])
    train_y = train_y.reshape(len(train_y), -1)

    return train_seq, train_epi,train_y


def build_model():
    seq_input = Input(shape=(23, 4))
    seq_conv1 = Convolution1D(64, 3,  name='seq_conv_1')(seq_input)
    seq_bat1 = BatchNormalization(name='seq_batchNormalization1')(seq_conv1)
    seq_act1 = Activation('relu', name='seq_activation1')(seq_bat1)
    seq_pool1 = MaxPooling1D(2,padding='same')(seq_act1)
    seq_drop1 = Dropout(0.2)(seq_pool1)

    seq_conv2 = Convolution1D(128, 3, name='seq_conv_2')(seq_drop1)
    seq_bat2 = BatchNormalization(name='seq_batchNormalization2')(seq_conv2)
    seq_act2 = Activation('relu', name='seq_activation2')(seq_bat2)
    seq_pool2 = MaxPooling1D(2, padding='same')(seq_act2)
    seq_drop2 = Dropout(0.2)(seq_pool2)

    seq_conv3 = Convolution1D(256, 3, name='seq_conv_3')(seq_drop2)
    seq_bat3 = BatchNormalization(name='seq_batchNormalization3')(seq_conv3)
    seq_act3 = Activation('relu', name='seq_activation3')(seq_bat3)
    seq_pool3 = MaxPooling1D(2, padding='same')(seq_act3)
    seq_drop3 = Dropout(0.2)(seq_pool3)
    seq_flat = Flatten()(seq_drop3)

    seq_dense1 = Dense(256, activation='relu', name='seq_dense_1')(seq_flat)
    seq_drop4 = Dropout(0.2)(seq_dense1)
    seq_dense2 = Dense(128, activation='relu', name='seq_dense_2')(seq_drop4)
    seq_drop5 = Dropout(0.2)(seq_dense2)
    seq_dense3 = Dense(64, activation='relu', name='seq_dense_3')(seq_drop5)
    seq_drop6 = Dropout(0.2)(seq_dense3)
    seq_out = Dense(32, activation='relu', name='seq_dense_4')(seq_drop6)

    epi_input = Input(shape=(23, 4))
    epi_conv1 = Convolution1D(64, 3, name='epi_conv_1')(epi_input)
    epi_bat1 = BatchNormalization(name='epi_batchNormalization1')(epi_conv1)
    epi_act1 = Activation('relu', name='epi_activation1')(epi_bat1)
    epi_pool1 = MaxPooling1D(2, padding='same')(epi_act1)
    epi_drop1 = Dropout(0.2)(epi_pool1)

    epi_conv2 = Convolution1D(128, 3, name='epi_conv_2')(epi_drop1)
    epi_bat2 = BatchNormalization(name='epi_batchNormalization2')(epi_conv2)
    epi_act2 = Activation('relu', name='epi_activation2')(epi_bat2)
    epi_pool2 = MaxPooling1D(2, padding='same')(epi_act2)
    epi_drop2 = Dropout(0.2)(epi_pool2)

    epi_conv3 = Convolution1D(256, 3, name='epi_conv_3')(epi_drop2)
    epi_bat3 = BatchNormalization(name='epi_batchNormalization3')(epi_conv3)
    epi_act3 = Activation('relu', name='epi_activation3')(epi_bat3)
    epi_pool3 = MaxPooling1D(2, padding='same')(epi_act3)
    epi_drop3 = Dropout(0.2)(epi_pool3)
    epi_flat = Flatten()(epi_drop3)

    epi_dense1 = Dense(256, activation='relu', name='epi_dense_1')(epi_flat)
    epi_drop4 = Dropout(0.2)(epi_dense1)
    epi_dense2 = Dense(128, activation='relu', name='epi_dense_2')(epi_drop4)
    epi_drop5 = Dropout(0.2)(epi_dense2)
    epi_dense3 = Dense(64, activation='relu', name='epi_dense_3')(epi_drop5)
    epi_drop6 = Dropout(0.2)(epi_dense3)
    epi_out = Dense(32, activation='relu', name='epi_dense_4')(epi_drop6)

    merged = concatenate([seq_out, epi_out], axis=-1)

    pretrain_model = Model(inputs=[seq_input, epi_input], outputs=[merged])
    pretrain_model.load_weights("weights/weights.h5", by_name=True)

    prediction =Dense(1, activation='sigmoid', name='prediction')(merged)
    model = Model([seq_input, epi_input], prediction)
    return merged,model

if __name__ == '__main__':
    train_path = "data/HL60.csv"
    seq_train,  epi_train, y_train = load_data(train_path)

    merged, model = build_model()
    new_model = Model(model.inputs, outputs=[merged])
    x_train = new_model.predict([seq_train, epi_train])
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train0, test_size=0.2)
    selected_cnn_fea_cols = [0, 2, 4, 5, 8, 13, 14, 15, 21, 22, 24, 26, 29, 30, 32]
    x_train = x_train[:, selected_cnn_fea_cols]

    xgmodel = XGBRegressor(learning_rate=0.1,
                               n_estimators=1000, 
                               max_depth=6, 
                               min_child_weight=1,
                               gamma=0.,
                               subsample=0.8,
                               colsample_btree=0.8,
                               objective='reg:logistic',
                               scale_pos_weight=1,
                               random_state=27)
    # xgmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    xgmodel.fit(x_train, y_train, eval_set=[(x_train, y_train)],
                    eval_metric=['rmse','mae'], early_stopping_rounds=15, verbose=True)
    y_pred = xgmodel.predict(x_train)
    y_test = np.array(y_train).ravel()
    y_test = list(y_test)

