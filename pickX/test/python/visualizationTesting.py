import pandas as pd
import matplotlib.pyplot as plt
import time
from pickX.data.DataHandler import DataHandler
from pickX.data.DataListQueryBuilder import DataListQueryBuilder
from pickX.utils.Visualizer import Visualiser
import pandas as pd

path = "C:\git\pickx\data\Testing"
#path = "D:\Richard's Laptop\git\pickx\data\Testing"
#ds = DataHandler.import_dataset(path)
#dlqb = DataListQueryBuilder().init_with_dataset(ds)
#dl = dlqb.generate_noise_ratios(12).datalist
#DataHandler.pickle(dl, "datalist" + str(time.time()), path=path)

datalist = DataHandler.import_dataset(path)
picked = list(filter(lambda x: x["picked"], datalist))
not_picked = list(filter(lambda x: not x["picked"], datalist))
#vis = Visualiser.visualize(datalist[101])
def analysis(not_picked, picked, minlength=200):
    dataframes = []
    for i in not_picked[:minlength]:
        df = pd.DataFrame(i["noise_ratios"])
        dataframes.append((df,i["picked"],df.std()[0],df.mean()[0]))
    for i in picked[:minlength]:
        df = pd.DataFrame(i["noise_ratios"])
        dataframes.append((df,i["picked"],df.std()[0],df.mean()[0]))

    picked = list(filter(lambda x : x[1], dataframes))
    not_picked = list(filter(lambda x : not x[1], dataframes))

    stds_picked = list(map(lambda x : x[2], picked))
    stds_not_picked = list(map(lambda x : x[2], not_picked))

    print("avg picked std: "+str(sum(stds_picked)/len(stds_picked)))
    print("avg not picked std: "+str(sum(stds_not_picked)/len(stds_not_picked)) if len(stds_not_picked) != 0 else 0)


def predict(datalist) -> list:
    """ if std of datalist_elem > 2 predict pickable

    :param datalist:
    :return:
    """
    threshhold = 0.9
    return [1 if x>threshhold else 0 for x in [pd.DataFrame(z["noise_ratios"]).std()[0] for z in datalist]]

print("accuracy on picked: "+str((predict(picked[:1000]).count(1) / 1000)))
print("accuracy on not_picked: "+str((predict(not_picked[:1000]).count(0) / 1000)))












"""
dataset = DataHandler.import_dataset("D:\Richard's Laptop\git\pickx\data\Testing")
dlqb = DataListQueryBuilder().init_with_dataset(dataset)

vector = dlqb.get_data_vectors()
picked = dlqb.is_picked(True).get_data_vectors()
not_picked = dlqb.is_picked(False).get_data_vectors()

number_of_plots = 10
fig, axes = plt.subplots(nrows=1, ncols=number_of_plots)
df_list = []

for j in range(number_of_plots):
    df_list.append((pd.DataFrame(vector["noise_ratios"].reshape(25215, 1000)[j], range(len((vector["noise_ratios"][j])))),
                    pd.DataFrame(vector["features"].reshape(25215, 1024)[j],range(len((vector["features"][j]))))
    ))

counter = 0

for i in range(len(vector["pick"])):
    if vector["pick"][i] is None:
        df_list.append((
            pd.DataFrame(vector["noise_ratios"].reshape(25215, 1000)[i], range(len((vector["noise_ratios"][i]))))
            , pd.DataFrame(vector["features"].reshape(25215, 1024)[i], range(len((vector["features"][i]))))))
        counter += 1
        if counter >= number_of_plots:
            break

for i, (d1, d2) in list(zip(range(0, len(df_list), 2), df_list)):
    d1.plot(ax=axes[i])
    d2.plot(ax=axes[i + 1])
    std = d1.std()
    mean = d1.mean()
    th = mean[0]+std[0]
    print(len(list(filter(lambda x: x[0]>th, d1.values))))
    print("std" + str(d1.std()) + "\nmean: " + str(d1.mean()))
plt.show()
"""