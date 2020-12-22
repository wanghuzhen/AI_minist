'''
Author: Wang Huzhen
Version: 1.0
FilePath: \homework\minist_autoencoder.py
Email: 2327253081@qq.com
Date: 2020-12-13 15:37:44
'''
from json import decoder, encoder
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np


# 获取数据集合并归一化处理
def load_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train.reshape((-1, 28*28)) / 255.0
    X_test = x_test.reshape((-1, 28*28)) / 255.0
    return X_train, y_train, X_test, y_test


# 创建模型
def create_model():
    n_hidden_1 = 256
    n_hidden_2 = 128
    n_hidden_3 = 32
    inputs = layers.Input(shape=(784,), name='inputs')
    inputs_1 = layers.Dense(
        n_hidden_1, activation='relu', name='inputs_1')(inputs)
    inputs_2 = layers.Dense(
        n_hidden_2, activation='relu', name='inputs_2')(inputs_1)
    inputs_3 = layers.Dense(
        n_hidden_3, activation='relu', name='inputs_3')(inputs_2)
    outputs_2 = layers.Dense(
        n_hidden_2, activation='relu', name='outputs_2')(inputs_3)
    outputs_1 = layers.Dense(
        n_hidden_1, activation='relu', name='outputs_1')(outputs_2)
    outputs = layers.Dense(784, activation='softmax',
                           name='outputs')(outputs_1)
    auto_encoder = keras.Model(inputs, outputs)
    print(auto_encoder.summary())
    # keras.utils.plot_model(auto_encoder, show_shapes=True)
    encoder = keras.Model(inputs, inputs_3)
    # keras.utils.plot_model(encoder, show_shapes=True)
    decoder_input = keras.Input((n_hidden_3,))
    decoder_out_1 = auto_encoder.layers[-3](decoder_input)
    decoder_out_2 = auto_encoder.layers[-2](decoder_out_1)
    decoder_out_3 = auto_encoder.layers[-1](decoder_out_2)
    decoder = keras.Model(decoder_input, decoder_out_3)
    # keras.utils.plot_model(decoder, show_shapes=True)
    return auto_encoder, encoder, decoder


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    auto_encoder, encoder, decoder = create_model()
    auto_encoder.compile(optimizer='adam',
                         loss='binary_crossentropy')
    # add noisy
    noise_factor = 0.3
    x_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1, size=X_train.shape)
    x_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1, size=X_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    history = auto_encoder.fit(
        x_train_noisy, X_train, batch_size=64, epochs=20, validation_split=0.1)
    encoded = encoder.predict(x_test_noisy)
    decoded = decoder.predict(encoded)
    plt.figure(figsize=(10, 4))
    n = 5
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, n+i+1)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
