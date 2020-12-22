# @File    :   minist_cnn.py
# @Version :   1.0
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/11/28 20:22:08
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import os


# 加载数据集
# 获取数据集合并归一化处理
def load_data():
    (X_tarin, y_train), (X_test, y_test) = mnist.load_data()
    X_train4D = X_tarin.reshape(X_tarin.shape[0], 28, 28, 1).astype('float32')
    X_test4D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_train4D_Normalize = X_train4D / 255  # 归一化
    X_test4D_Normalize = X_test4D / 255
    # 独热编码
    y_trainOnehot = to_categorical(y_train)
    y_testOnehot = to_categorical(y_test)
    return X_train4D_Normalize, y_trainOnehot, X_test4D_Normalize, y_testOnehot


# 创建模型
def create_model():
    model = Sequential()  # 一层卷积
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding='same',  # 保证卷积核大小，不够补零
            input_shape=(28, 28, 1),
            activation='relu'))
    # 池化层1
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 二层卷积
    model.add(
        Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    # 池化层2
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(
        Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(
        Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 平坦层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    # 输出神经网络模型
    print(model.summary())
    return model


# 训练模型
def model_train(epoch, X_train4D_Normalize, y_trainOnehot, X_test4D_Normalize, y_testOnehot):
    if os.path.exists('data&model/minist_cnn_model.h5'):
        restored_model = tf.keras.models.load_model(
            'data&model/minist_cnn_model.h5')
        history = restored_model.fit(X_train4D_Normalize, y_trainOnehot, validation_data=(
            X_test4D_Normalize, y_testOnehot), epochs=epoch)
        # 保存训练模型的权重和偏置
        restored_model.save('data&model/minist_cnn_model.h5')
        # 删除模型
        del restored_model
    else:
        log_dir = os.path.join('logs')
        # print(log_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # 定义TensorBoard对象.histogram_freq 如果设置为0，则不会计算直方图。
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        model = create_model()
        # lr为学习率，上次设置为0.01
        model.compile(optimizer=tf.keras.optimizers.Adam(
            lr=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
        history = model.fit(X_train4D_Normalize, y_trainOnehot, validation_data=(
            X_test4D_Normalize, y_testOnehot),batch_size = 300, epochs=epoch,callbacks=[tensorboard_callback])
        # 保存训练模型的权重和偏置
        model.save('data&model/minist_cnn_model.h5')
        # 删除模型
        del model
    return history


# 定义训练结果可视化
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 定义图片可视化
def plot_labels_prediction(images, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap='binary')
        if len(prediction) > 0:
            title = 'labels' + \
                str(list(prediction[i]).index(max(prediction[i])))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

if __name__=='__main__':
    X_train4D_Normalize, y_trainOnehot, X_test4D_Normalize, y_testOnehot = load_data()
    history = model_train(10, X_train4D_Normalize, y_trainOnehot, X_test4D_Normalize, y_testOnehot)
    # 准确率
    show_train_history(history, 'accuracy', 'val_accuracy')
    # 损失率
    show_train_history(history, 'loss', 'val_loss')

    model = tf.keras.models.load_model('data&model/minist_cnn_model.h5')
    # 预测值,预测结果的shape为(10000,10)，10分别表示对10个数字的可能性
    prediction = model.predict(X_test4D_Normalize)
    # 可视化结果,起始位置与index数值相同
    plot_labels_prediction(X_test4D_Normalize, prediction[8:18], idx=8)
