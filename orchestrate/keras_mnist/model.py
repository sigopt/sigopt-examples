import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers

import orchestrate.io

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

OPTIMIZERS = {
    'adam': optimizers.Adam,
    'rmsprop': optimizers.RMSprop,
    'gradient_descent': optimizers.SGD,
}

def create_model():
    model = Sequential()
    model.add(Conv2D(orchestrate.io.assignment('conv1_size', default=32), kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(orchestrate.io.assignment('conv2_size', default=64), kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(orchestrate.io.assignment('dropout', default=0.8)))
    model.add(Flatten())
    model.add(Dense(orchestrate.io.assignment('hidden1_size', default=500), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=OPTIMIZERS[orchestrate.io.assignment('optimizer', default='adam')](
            lr=10**orchestrate.io.assignment('log_learning_rate', default=-3)
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=orchestrate.io.assignment('epochs', default=10), batch_size=100)

    return model

def evaluate_model():
    model = create_model()
    accuracy = model.evaluate(x_test, y_test)[1]
    orchestrate.io.log_metric('accuracy', accuracy)

if __name__ == '__main__':
  evaluate_model()
