# -*- coding: utf-8 -*-s
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
gpus = tf.config.list_physical_devices('GPU')


def save_trajectory_plot(id_number, normalized_original, predicted_data):
    # 绘制实际轨迹和预测结果的散点图
    plt.figure(figsize=(8, 6))  # 可以设置图形大小
    plt.scatter(normalized_original[:, 0], normalized_original[:, 1], c='r', marker='o', label='实际轨迹')
    plt.scatter(predicted_data[:, 0], predicted_data[:, 1], c='b', marker='o', label='预测结果')
    plt.gca().invert_yaxis()  # Y轴向下增大
    plt.legend(loc='upper right')
    plt.grid()

    # 在图像下方添加ID
    plt.figtext(0.5, 0.04, f"ID: {id_number}", ha='center', va='center', fontsize=12)

    # 文件保存路径，保存图像
    filename = f"./result_some/trajectory_ID{id_number}.png"
    plt.savefig(filename)

    # 关闭当前图形，防止与后续的图形发生冲突
    plt.close()



def reshape_y_hat(y_hat, dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i + j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y


# 数据切分
def data_set(dataset, test_num):  # 创建时间序列数据样本
    dataX, dataY = [], []
    time_step = dataset[:, 0:1].flatten()
    dataset = dataset[:, 1:3]
    for i in range(len(dataset) - test_num):  # 滑动窗口选择test_num步的数据组合。最后一组预测没有下一步，故使用len(dataset) - test_num
        a = dataset[i:(i + test_num)]
        dataX.append(a)
        dataY.append(dataset[i + test_num])
    return np.array(dataX), np.array(dataY), time_step


# 使用训练数据的归一化
def NormalizeMultUseData(data, normalize):
    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data


def calculate_normalized_errors(original_data, predicted_data, id_number):
    # 归一化原始数据
    scaler_value = 99
    normalized_original = original_data / scaler_value

    # 计算归一化后的MSE
    mse_normalized = mean_squared_error(normalized_original, predicted_data)
    # print('归一化后mse:', mse_normalized)

    # 计算归一化后的平均位移误差
    errors = np.sqrt(np.sum((normalized_original - predicted_data) ** 2, axis=1))
    average_displacement_error_normalized = np.mean(errors)
    # print('归一化后Average Displacement Error:', average_displacement_error_normalized)

    # 绘制实际轨迹和预测结果的散点图
    save_trajectory_plot(id_number, normalized_original, predicted_data)
    # 返回计算结果
    return mse_normalized, average_displacement_error_normalized


def predict_in_batches(model, data, batch_size=4096):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=0)
        predictions.extend(batch_predictions)
    return np.array(predictions)


def read_and_process_trajectory_data(id_number, normalize, model):
    # 根据ID拼接文件路径
    file_path = "D:/unit/some_train/data_" + str(id_number) + ".txt"

    # 从文件中读取原始数据和实验数据
    yuanshi = pd.read_csv(file_path, sep=r' ', engine='python', header=None).iloc[:, 0:3].values
    ex_data = pd.read_csv(file_path, sep=r' ', engine='python', header=None).iloc[:, 0:3].values

    # 创建数据集
    data, dataY, time_step = data_set(ex_data, test_num)
    data = data.astype('float64')
    # 转换数据类型
    data_guiyi = []
    for i in range(len(data)):
        data[i] = list(NormalizeMultUseData(data[i], normalize))
        data_guiyi.append(data[i])

    y_hat1 = []  # 已经反归一化的预测值
    predict_time_step = []
    for i in range(len(data_guiyi)):
        test_X = data_guiyi[i].reshape(1, data_guiyi[i].shape[0], data_guiyi[i].shape[1])
        dd = model.predict(test_X, verbose=0)
        predict_time_step.append(time_step[i + test_num])
        # 在进行任何处理之前，将未反归一化的预测值添加到y_hat1中
        dd_unnormalized = dd.tolist()  # 转换为一维列表
        y_hat1.append(dd_unnormalized[0])

    # 将y_hat和y_hat1转换为NumPy数组，以便后续处理
    y_hat1 = np.array(y_hat1)
    plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
    # 计算归一化后的误差
    yuanshi_trimmed = yuanshi[test_num:, 1:3]
    mse, average_displacement_error = calculate_normalized_errors(yuanshi_trimmed, y_hat1, id_number)
    with open(filename, "a") as file:
        file.write(f"ID: {id_number}, MSE: {mse}, ADE: {average_displacement_error}\n")
    print(f"ID: {id_number}, MSE: {mse}, ADE: {average_displacement_error}")
    return y_hat1, predict_time_step


def add_noise(predictions, noise_level=1e-4):
    """
    向预测结果添加随机噪声
    """
    noise = np.random.normal(0, noise_level, predictions.shape)
    predictions_with_noise = predictions + noise
    return predictions_with_noise


def ensure_unique_y_hat(predict_time_step_all, y_hat):
    for i, time_steps in enumerate(predict_time_step_all):
        for j, time_step in enumerate(time_steps):
            # 检查这个时间步是否已经在其他地方出现过
            if time_step not in used_y_hat_values:
                used_y_hat_values[time_step] = set()

            # 当前的y_hat值
            current_value = tuple(y_hat[i][j])

            # 如果这个y_hat值已经被使用过了，寻找一个新的值
            while current_value in used_y_hat_values[time_step]:
                y_hat[i][j] = add_noise(y_hat[i][j])  # 修改y_hat值
                current_value = tuple(y_hat[i][j])

            # 标记这个y_hat值已经被使用过了
            used_y_hat_values[time_step].add(current_value)

    return y_hat


if __name__ == '__main__':
    test_num = 2  # 根据两步预测一步
    per_num = 1  # 预测步数
    # normalize = np.load("traj_model_trueNorm.npy")
    # model = load_model("traj_model_120.h5")
    normalize = np.load("traj_model_trueNorm_some.npy")
    model = load_model("traj_model_120_some.h5")
    # normalize = np.load("traj_model_trueNorm_some_new.npy")
    # model = load_model("traj_model_120_some_new.h5")

    with open(r'D:\\unit\\some_train\\unitIDList.txt', 'r') as f:
        ids = f.readlines()

    # 获取当前时间，用于命名文件
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./result_some/results_{current_time}.txt"
    # 打印模型结构
    model.summary()
    # 对每个ID进行处理
    y_hat = []  # 预测结果
    predict_time_step_all = []  # 所有预测结果对应的时间步
    for id_number in ids:
        id_number = id_number.strip()  # 去除可能的换行符
        y_hatn, predict_time_step = read_and_process_trajectory_data(id_number, normalize, model)
        predict_time_step_all.append(predict_time_step)
        y_hat.append(y_hatn)
    used_y_hat_values = {}
    y_hat = ensure_unique_y_hat
