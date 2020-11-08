from pickX.data.DataHandler import DataHandler
from pickX.ml.SNR import SNR
"""
path = "C:\git\pickx\data\Training"
training_datalist = DataHandler.import_dataset(path)
snr = SNR()
snr.__train__(training_datalist)
snr.__export__("C:/git/pickx/pickX/test/models/snr.model")

"""

path = "C:\git\pickx\data\Testing"
testing_datalist = DataHandler.import_dataset(path)
snr = SNR()
snr.__import__("C:/git/pickx/pickX/test/models/snr.model")
evaluation = snr.__test__(testing_datalist)
print(evaluation)
