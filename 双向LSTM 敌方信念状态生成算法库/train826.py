# -*- coding: utf-8 -*-
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Dense, LSTM, Dropout, Flatten, Bidirectional, Attention
from keras.models import Sequential, Model, load_model
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def create_dataset(data, n_predictions, n_next):
    '''
    对数据进行处理
    n_predictions:学习步数
    n_next：预测步数1
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    test_X, test_Y = [], []
    i = data.shape[0] - n_predictions - n_next - 1
    a = data[i:(i + n_predictions), :]
    test_X.append(a)
    tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
    b = []
    for j in range(len(tempb)):
        for k in range(dim):
            b.append(tempb[j, k])
    test_Y.append(b)
    test_X = np.array(test_X, dtype='float64')
    test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y


def NormalizeMult(data, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)
    listlow = 0
    listhigh = 99
    for i in range(0, data.shape[1]):

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = float(listhigh) - float(listlow)
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data, normalize


if __name__ == "__main__":
    train_num = 2  # 根据两步预测一步
    per_num = 1  # 预测步数
    # set_range = False
    set_range = True

    # 读入时间序列的文件数据
    new_data = pd.read_csv(r"C:\Users\87451\Desktop\JK\RTS\lstm\data_some.txt", sep=r' ', engine='python',
                           header=None).iloc[:, 0:3].values
    time_step = new_data[:, 0:1].flatten()  # 取第一列，形状为 (10, 1)
    new_data = new_data[:, 1:3]
    # 画样本数据库
    plt.scatter(new_data[:, 1], new_data[:, 0], c='r', marker='o', label='result of recognition')
    # plt.scatter(new_data[:, 2], new_data[:, 1], c='r', marker='o', label='result of recognition')
    plt.legend(loc='upper right')
    plt.grid()

    # 训练模型
    # model = trainModel(train_X, train_Y)
    model = load_model("./traj_model_120.h5")
    new_data = new_data.astype('float64')
    new_data, normalize = NormalizeMult(new_data, set_range=True)

    # 生成新的训练数据
    train_X, train_Y, test_X, test_Y = create_dataset(new_data, train_num, per_num)
    print("x\n", train_X.shape)
    print("y\n", train_Y.shape)

    # 用新的数据集进行迁移学习
    # model = trainModel(train_X, train_Y, base_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])

    # 学习率调度器和早停
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(train_X, train_Y, epochs=50, batch_size=16, verbose=1,
              callbacks=[lr_scheduler, early_stopping])
    model.summary()
    loss, acc = model.evaluate(train_X, train_Y, verbose=2)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    np.save("traj_model_trueNorm_some.npy", normalize)
    model.save("./traj_model_120_some.h5")
