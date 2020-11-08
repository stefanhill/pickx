import time

from pickX.data.DataHandler import DataHandler
from pickX.data.DataListQueryBuilder import DataListQueryBuilder

path = "C:\git\pickx\data\Training"
vector = DataHandler.import_dataset_binary(path)

dlqb = DataListQueryBuilder().init_with_dataset(vector)
vector = dlqb.generate_noise_ratios(12).datalist
DataHandler.pickle(vector, str(time.time()), path=path)
