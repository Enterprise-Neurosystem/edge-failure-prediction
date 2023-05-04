import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from dataprep.data_preparation import DataPreparation
from model.lstm_model import LSTMModel

# import joblib


class TrainManager:
    model = None  # Make model a class variable for easy retrieval

    def __init__(
        self,
        hidden_layer1_nodes,
        hidden_layer2_nodes,
        hidden_layer3_nodes,
        learning_rate,
    ):
        self.train_input_shape = (
            DataPreparation.X_train.shape[1],
            DataPreparation.X_train.shape[2],
        )
        TrainManager.model = LSTMModel(
            self.train_input_shape,
            hidden_layer1_nodes,
            hidden_layer2_nodes,
            hidden_layer3_nodes,
        )
        TrainManager.model.compile(learning_rate)

    # This is a learning rate callback.  Its purpose is to reduce the learning rate by a factor of 10 after 10 epochs
    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * (0.1)

    def fit_model(self, X, y, epochs=10, batch_size=64):
        # Must reshape labels, y to match 3-d shape of output
        reshaped_labels = tf.expand_dims(
            tf.expand_dims(DataPreparation.y_train, axis=1), axis=1
        )
        print(
            "y_train original shape: {} y_train reshaped: {}".format(
                y, tf.expand_dims(tf.expand_dims(y, axis=1), axis=1).shape
            )
        )

        # TODO:  make callbacks configurable
        callback = LearningRateScheduler(TrainManager.scheduler)
        history = TrainManager.model.fit(
            X, reshaped_labels, epochs, batch_size, callbacks=[callback]
        )
        TrainManager.model.save("static/cache/trained_model")
        # joblib.dump(TrainManager.model, 'static/cache/trained_model')

        return history

    @staticmethod
    def calculate_job_size(epochs, batch_size):
        return epochs * batch_size


# def save_model(self):
# joblib.dump(self.model, 'static/cache/trained_model.gz')
#     tf.keras.models.save_model(self.model, 'static/cache/trained_model.gz')
