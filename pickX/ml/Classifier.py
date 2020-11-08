import abc


class Classifier(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __train__(self, dataset):
        pass

    @abc.abstractmethod
    def __evaluate__(self, data_vector):
        pass

    @abc.abstractmethod
    def __test__(self, data_vector):
        pass

    @abc.abstractmethod
    def __import__(self, path):
        pass

    @abc.abstractmethod
    def __export__(self, path):
        pass
