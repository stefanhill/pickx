from pickX.data.DataHandler import DataHandler
from pickX.data.DataListQueryBuilder import DataListQueryBuilder

#dataset = DataHandler.import_dataset("D:\Richard's Laptop\git\pickx\data\Testing", "P021 2018 UniWork Tomo Siegen-Eisern")
dataset = DataHandler.import_dataset("C:\git\pickx\data\Testing", "P021 2018 UniWork Tomo Siegen-Eisern")

dlqb = DataListQueryBuilder().init_with_dataset(dataset)
dlqb.generate_noise_ratios(12)
