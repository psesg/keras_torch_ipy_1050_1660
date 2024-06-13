import os

# for Tensorflow Suppressing Warning messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy
# from keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
from keras import utils


def setup_tf_gpu(str_tf_ver):
    major_tf_ver = int(str(str_tf_ver).split('.', 3)[0])
    if major_tf_ver >= 2:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        for i in range(0, len(gpu)):
            print(gpu[i])
            details = tf.config.experimental.get_device_details(gpu[i])
            device_name = details.get('device_name', 'Unknown GPU')
            print(f'device name: {device_name}')
            compute_capability = details.get('compute_capability', 'Unknown')
            print(f'compute capability: {compute_capability[0]}.{compute_capability[1]}')
            tf.config.experimental.set_memory_growth(gpu[i], True)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        tf.keras.backend.set_session(tf.Session(config=config))
    print('tensorflow v.=', str_tf_ver, ' configurated for major v.=', major_tf_ver)


setup_tf_gpu(tf.__version__)

# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)

# Размер изображения
img_rows, img_cols = 28, 28

# Загружаем данные
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

import gzip
import sys
import pickle

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(X_train, y_train), (X_test, y_test) = data

# Преобразование размерности изображений
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = utils.to_categorical(y_train, 10)
Y_test = utils.to_categorical(y_test, 10)

# Создаем последовательную модель
model = Sequential()

model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
with tf.device('/GPU:0'):
    model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

#print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
# Work1 Win10 Python 3.7.3:
# tensorflow v.  1.14.0
# name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.468
# device:GPU:0 with 3001 MB memory

# Epoch 9/10
#  - 6s - loss: 0.2081 - acc: 0.9381 - val_loss: 0.1233 - val_acc: 0.9634
# Epoch 10/10
# - 6s - loss: 0.1917 - acc: 0.9414 - val_loss: 0.1150 - val_acc: 0.9644
# Точность работы на тестовых данных: 96.73%
#------------------------------------------------------------------------
# Work2 Win10 Python 3.6.5:
# tensorflow v.  1.8.0
# name: GeForce GTX 660 major: 3 minor: 0 memoryClockRate(GHz): 1.0975
# device:GPU:0 with 1348 MB memory

# Epoch 9/10
#  - 10s - loss: 0.2081 - acc: 0.9380 - val_loss: 0.1232 - val_acc: 0.9636
# Epoch 10/10
#  - 10s - loss: 0.1917 - acc: 0.9414 - val_loss: 0.1150 - val_acc: 0.9643
# Точность работы на тестовых данных: 96.75%
#------------------------------------------------------------------------
# Work2 Ub16.4 Python 3.6.5:
# tensorflow v.  1.8.0
# name: GeForce GTX 660 major: 3 minor: 0 memoryClockRate(GHz): 1.0975
# device:GPU:0 with 1487 MB memory

# Epoch 9/10
# - 10s - loss: 0.2082 - acc: 0.9380 - val_loss: 0.1233 - val_acc: 0.9635
# Epoch 10/10
#  - 10s - loss: 0.1918 - acc: 0.9413 - val_loss: 0.1150 - val_acc: 0.9645
# Точность работы на тестовых данных: 96.74%
#------------------------------------------------------------------------
# Home1 MSI Win10 Python 3.7.3:
# tensorflow v.  2.0.0
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
# device:GPU:0 with 1343 MB memory

#Epoch 9/10
# - 8s - loss: 0.2082 - accuracy: 0.9382 - val_loss: 0.1233 - val_accuracy: 0.9635
#Epoch 10/10
# - 8s - loss: 0.1918 - accuracy: 0.9413 - val_loss: 0.1150 - val_accuracy: 0.9647
# Точность работы на тестовых данных: 96.73%
#------------------------------------------------------------------------
# Home1 MSI Ub19.10 Python 3.7.3:
# tensorflow v.  2.0.0
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
# device:GPU:0 with 1412 MB memory

# Epoch 9/10
#  - 6s - loss: 0.2081 - accuracy: 0.9379 - val_loss: 0.1233 - val_accuracy: 0.9635
# Epoch 10/10
#  - 6s - loss: 0.1917 - accuracy: 0.9415 - val_loss: 0.1150 - val_accuracy: 0.9645
# Точность работы на тестовых данных: 96.76%
#------------------------------------------------------------------------
# Home1 MBA W10 Python 3.7.4:
# tensorflow v.  2.0.0
# no device:GPU CPU DualCore Intel Core i5-3427U, 2600 MHz (26 x 100) RAM 4Gb

# Epoch 1/10
#  - 201s - loss: 1.9407 - accuracy: 0.3662 - val_loss: 0.8939 - val_accuracy: 0.8205
# Epoch 2/10
#  - 193s - loss: 0.7646 - accuracy: 0.7588 - val_loss: 0.3671 - val_accuracy: 0.8953
#------------------------------------------------------------------------
# Work2 W10/Ub20.04.1 LTS Python 3.7.3:
# tensorflow v.  2.0.0
# name: GeForce GTX 1660, Compute Capability 7.5 memoryClockRate(GHz): 1.785
# device:GPU:0 with 4587 MB memory

#Epoch 9/10
# - 3s - loss: 0.2082 - accuracy: 0.9380 - val_loss: 0.1232 - val_accuracy: 0.9635
#Epoch 10/10
# - 3s - loss: 0.1918 - accuracy: 0.9414 - val_loss: 0.1150 - val_accuracy: 0.9644
#Точность работы на тестовых данных: 96.74%
