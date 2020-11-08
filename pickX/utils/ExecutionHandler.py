from pickX.data.DataHandler import DataHandler
from pickX.data.DataListQueryBuilder import DataListQueryBuilder
from pickX.data.DataSet import DataSet
from pickX.ml.CNN import CNN
from os.path import expanduser
import time

USERSPACE_PATH = expanduser("~") + "\\pickX"
MODEL_FILE_EXTENSION = '.model'
MODEL_PATH = USERSPACE_PATH + '\\models\\'
LOGS_PATH = USERSPACE_PATH + '\\logs\\'


class ExecutionHandler:

    @staticmethod
    def predict(model_name: str, path: str) -> object:
        """
        High-level method for retrieving predictions
        :param model_name: name of the mode in userdir to use
        :param path: path of the data set to be validated
        :return: a summary of the prediction -> true / false ratio
        """
        dataset = DataHandler.import_dataset_binary(path)
        model_name = model_name[:-6] if model_name.endswith(MODEL_FILE_EXTENSION) else model_name

        model = CNN()
        model.__import__(MODEL_PATH + model_name + MODEL_FILE_EXTENSION)

        dlqb = DataListQueryBuilder().init_with_dataset(dataset)
        data_vector = dlqb.normalize_length(pos=model.input_dimension[0]) \
            .normalize_ampl_by_trace() \
            .get_data_vectors()

        result = model.__evaluate__(data_vector)

        summary = DataHandler.export_pck_from_datalist(result, path)
        DataHandler.export_summary(summary, path)
        return summary

    @staticmethod
    def train(model_name: str, path: str, model_conf: (([[int]], int, int), int)):
        """
        High-level method to train a model
        :param model_name: unique name for the model to be saved in userdir
        :param path: path of training data set
        :param model_conf: model configuration, a tuple with (list of two-element list that contain
        [number of node, window size] elements, number of dense layers, number of dense nodes, training epochs
        :return: void
        """
        dataset = DataHandler.import_dataset(path)
        model_name = model_name[:-6] if model_name.endswith(MODEL_FILE_EXTENSION) else model_name

        if isinstance(dataset, DataSet):
            dlqb = DataListQueryBuilder().init_with_dataset(dataset)
            data_vector = dlqb.shuffle() \
                .balance() \
                .normalize_length() \
                .normalize_ampl_by_trace() \
                .shuffle() \
                .generate_noise_ratios(12) \
                .get_data_vectors()
            DataHandler.pickle(data_vector, str(time.time()), path=path)
            input_shape = (dlqb.length, 1)
        else:
            data_vector = dataset
            input_shape = (len(data_vector['features'][0]), 1)

        model = CNN()
        model_params, epochs = model_conf
        model.initialize(*model_params, input_shape)

        model.__train__(data_vector, epochs=epochs, log_dir='{}{}'.format(LOGS_PATH, model_name))
        model.__export__(MODEL_PATH + model_name + MODEL_FILE_EXTENSION)

    @staticmethod
    def test(model_name: str, path: str) -> object:
        """
        High-level method to validate a model with given validation
        :param model_name: name of the model in userdir
        :param path: path to the validation data set
        :return: returns a tuple [val_los, val_accuracy]
        """
        dataset = DataHandler.import_dataset(path)
        model_name = model_name[:-6] if model_name.endswith(MODEL_FILE_EXTENSION) else model_name

        model = CNN()
        model.__import__(MODEL_PATH + model_name + MODEL_FILE_EXTENSION)

        if isinstance(dataset, DataSet):
            dlqb = DataListQueryBuilder().init_with_dataset(dataset)
            data_vector = dlqb.normalize_length(pos=model.input_dimension[0]) \
                .normalize_ampl_by_trace() \
                .generate_noise_ratios(12) \
                .get_data_vectors()
            DataHandler.pickle(data_vector, str(time.time()), path=path)
        else:
            data_vector = dataset

        dlqb = DataListQueryBuilder().init_with_data_vectors(data_vector)
        data_vector = dlqb.normalize_length(pos=model.input_dimension[0]) \
            .normalize_ampl_by_trace() \
            .get_data_vectors()

        return model.__test__(data_vector)

    @staticmethod
    def prepare_dataset_for_training(path: str) -> bool:
        """
        High-level method to create a pickled training data set from project folder
        :param path: path to the data set to be converted
        :return: true or false whether the conversion succeeded
        """
        dataset = DataHandler.import_dataset(path)
        if isinstance(dataset, DataSet):
            dlqb = DataListQueryBuilder().init_with_dataset(dataset)
            data_vector = dlqb.shuffle() \
                .balance() \
                .normalize_length() \
                .normalize_ampl_by_trace() \
                .shuffle() \
                .generate_noise_ratios(12) \
                .get_data_vectors()
            DataHandler.pickle(data_vector, str(time.time()), path=path)
            return True
        return False
