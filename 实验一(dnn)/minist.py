# @File    :   minist.py
# @Version :   1.0
# @Author  :   Wang Huzhen
# @Email   :   2327253081@qq.com
# @Time    :   2020/11/07 18:37:20
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.python.keras import layers
import os


# 可视化训练结果
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 获取数据集合并归一化处理
def load_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train_scaled = (x_train)/255.0
    X_test_scaled = (x_test)/255.0
    return X_train_scaled, y_train, X_test_scaled, y_test


# 创建模型
def create_model():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation="softmax"))
    print(model.summary())
    return model


# 优化器选择 Adam 优化器。
# 损失函数使用 sparse_categorical_crossentropy，
# 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
# sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
def model_train(epoch, X_train_scaled, y_train, X_test_scaled, y_test):
    if os.path.exists('data&model/minist_model.h5'):
        restored_model = tf.keras.models.load_model(
            'data&model/minist_model.h5')
        history = restored_model.fit(X_train_scaled, y_train, validation_data=(
            X_test_scaled, y_test), epochs=epoch)
        # 保存训练模型的权重和偏置
        restored_model.save('data&model/minist_model.h5')
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
            lr=0.001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        history = model.fit(X_train_scaled, y_train, validation_data=(
            X_test_scaled, y_test), epochs=epoch, callbacks=[tensorboard_callback])
        # 保存训练模型的权重和偏置
        model.save('data&model/minist_model.h5')
        # 删除模型
        del model
    return history


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


if __name__ == '__main__':
    X_train_scaled, y_train, X_test_scaled, y_test = load_data()

    # # 实测多次运行（6、7次左右）会是准确率降低，运行前删除保存的模型
    # history = model_train(6, X_train_scaled, y_train, X_test_scaled, y_test)
    # # # 准确率
    # show_train_history(history, 'accuracy', 'val_accuracy')
    # # # 损失率
    # show_train_history(history, 'loss', 'val_loss')

    model = tf.keras.models.load_model('data&model/minist_model.h5')
    # 预测值,预测结果的shape为(10000,10)，10分别表示对10个数字的可能性
    prediction = model.predict(X_test_scaled)
    # 可视化结果,起始位置与index数值相同
    plot_labels_prediction(X_test_scaled, prediction[12:22], idx=12)
