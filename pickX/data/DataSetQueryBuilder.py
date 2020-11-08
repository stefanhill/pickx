import copy

from pickX.data.DataSet import DataSet


class DataSetQueryBuilder:
    dataset = DataSet()

    def __init__(self, dataset):
        """
        Initializes the query builder with data set, e.g. a project
        :param dataset: the dataset for the queries
        """
        self.dataset = copy.deepcopy(dataset)

    # getter functions

    def get_dataset(self):
        """
        Getter
        :return: the dataset as DataSet
        """
        return self.dataset

    # filter and action functions

    def filter_by_project_names(self, project_names):
        """
        Filters the data set by given project name
        :param project_names: project name to be filtered
        :return: self
        """
        self.dataset.projects = list(filter(lambda x: x.name in project_names, self.dataset.projects))
        return self
