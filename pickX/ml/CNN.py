from abc import ABC

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

from pickX.ml.Classifier import Classifier


class CNN(Classifier, ABC):
    model = None
    input_dimension = None
    feature_field = 'noise_ratios'

    def initialize(self, conv_params: [int, int], dense_layers: int, dense_nodes: int,
                   input_shape: (int, int)):
        """
        builds a CNN model
        :param conv_params: [[number of neurons in layer, window size in layer]...]
        :param dense_layers: number of dense layers
        :param dense_nodes: number of neurons in dense layers
        :param input_shape: shape of input vector e.g. (1024,1)
        """
        self.model = Sequential()

        for nodes, window_size in conv_params:
            self.model.add(Conv1D(nodes, window_size, input_shape=input_shape))
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())

        for _ in range(dense_layers):
            self.model.add(Dense(dense_nodes))
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.5))

        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))

        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def __train__(self, data_vector: {}, epochs: int = 15, log_dir=None):
        """
        Trains a CNN on a given data vector
        :param data_vector: training data vector
        :param epochs: maximal number of epochs to train
        :param log_dir: directory to save the log files, usually userdir/pickX/logs
        :return: self
        """
        callbacks = []
        if log_dir is not None:
            callbacks.append(TensorBoard(log_dir=log_dir))
            callbacks.append(EarlyStopping("val_loss", patience=2))

        self.model.fit(x=data_vector[self.feature_field],
                       y=data_vector['labels'],
                       validation_split=0.1,
                       batch_size=32,
                       epochs=epochs,
                       callbacks=callbacks)
        self.model.summary()
        self.input_dimension = self.model.input_shape[1:]

        return self

    def __evaluate__(self, data_vector):
        """
        Makes a prediction on a given data vector
        :param data_vector: the data vector to be used for the prediction
        pick probabilities will be overwritten after operation
        :return: the data vector with pick probabilities inserted
        """
        values = self.model.predict(data_vector[self.feature_field])
        data_vector['pick_probability'] = values

        return data_vector

    def __test__(self, data_vector):
        """
        Validates a CNN on a given validation set
        :param data_vector: validation set as data vectors
        :return: evaluation results [val_loss, val_accuracy]
        """
        evaluation = self.model.evaluate(data_vector[self.feature_field], data_vector['labels'])

        return evaluation

    def __import__(self, path):
        """
        Loads a CNN from .model file
        :param path: path to the .model file
        :return: void
        """
        self.model = load_model(path)
        self.input_dimension = self.model.input_shape[1:]

    def __export__(self, path):
        """
        Saves a CNN to a .model file
        :param path: path to the .model file to be saved
        :return: void
        """
        save_model(self.model, path)
