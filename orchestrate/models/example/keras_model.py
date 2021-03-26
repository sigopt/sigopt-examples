import sigopt
from data_and_model_setup import LoadTransformData, log_inference_metrics, CheckpointCB
import time
import platform
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf


class KerasModel:

    def create_model(self, trainX):
        # model architecture
        model_keras = Sequential()

        model_keras.add(Dense(
            trainX.shape[1] * 2,
            activation='relu',
            kernel_initializer='random_normal',
            bias_initializer='zeros',
            input_dim=trainX.shape[1]
        ))
        model_keras.add(Dense(
            trainX.shape[1] * 2,
            activation='relu',
            kernel_initializer='random_normal',
            bias_initializer='zeros'
        ))
        model_keras.add(Dense(
            1,
            activation='sigmoid',
            kernel_initializer='random_normal',
            bias_initializer='zeros'
        ))

        return model_keras


def train_keras_model(dataset):
    tf.debugging.set_log_device_placement(True)

    # set tf seed
    seed_value = sigopt.get_parameter('random_seed', default=1)
    tf.compat.v1.set_random_seed(seed_value)

    print("loading and transforming data")
    load_transform_data = LoadTransformData()
    trainX, testX, trainY, testY = load_transform_data.load_split_dataset(dataset)
    scaled_trainX, scaled_testX = load_transform_data.scale_dataset(trainX, testX)

    # logging to sigopt Run
    sigopt.log_model("keras.Sequential")  # model_keras.__class__
    sigopt.log_dataset('Scaled Anomaly detection')
    sigopt.log_metadata('Training Records', len(scaled_trainX))
    sigopt.log_metadata('Testing Reccords', len(scaled_testX))
    sigopt.log_metadata("Platform", platform.uname())

    learning_rate = sigopt.get_parameter('learning_rate', default=0.01)
    loss_fn = sigopt.get_parameter('loss_function', default='binary_crossentropy')
    batch_size = sigopt.get_parameter('batch_size', default=4096)
    sigopt.get_parameter('layers', 3) # tracking number of layers to SigOpt Run
    num_epochs = sigopt.get_parameter('epochs', default=6)

    keras_model = KerasModel()
    model_keras = keras_model.create_model(trainX)
    model_keras.compile(
        optimizer=Adam(lr=learning_rate),
        loss=loss_fn,
        metrics=[tf.keras.metrics.AUC()]
    )

    model_keras.fit(
        scaled_trainX,
        trainY,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[CheckpointCB()],
        validation_data=(scaled_testX, testY),
    )

    # Collect model metrics
    start = time.perf_counter()
    probability = model_keras.predict(scaled_testX).flatten()
    prediction = probability > 0.5

    sigopt.log_metric('Inference Time', time.perf_counter() - start)
    log_inference_metrics(prediction, probability, testY, testX)


if __name__ == "__main__":
    dataset = 'https://www.dropbox.com/s/437qdt4yjj64sxd/Fraud_Detection_SigOpt_dataset.csv?dl=1'
    from tensorflow.python.client import device_lib
    print("Printing available devices")
    print(device_lib.list_local_devices())
    train_keras_model(dataset)
