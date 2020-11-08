import pandas as pd
import matplotlib.pyplot as plt
import time
from pickX.data.DataHandler import DataHandler
from pickX.data.DataListQueryBuilder import DataListQueryBuilder

path = "C:\git\pickx\data\Hessigheim"
vector = DataHandler.import_dataset(path)

#dlqb = DataListQueryBuilder().init_with_dataset(vector)
#vector = dlqb.generate_noise_ratios(12).get_data_vectors()
#DataHandler.pickle(vector, str(time.time()), path=path)


"""
dataset = DataHandler.import_dataset("D:\Richard's Laptop\git\pickx\data\Testing")
dlqb = DataListQueryBuilder().init_with_dataset(dataset)

vector = dlqb.get_vectors()
picked = dlqb.is_picked(True).get_vectors()
not_picked = dlqb.is_picked(False).get_vectors()

number_of_plots = 10
fig, axes = plt.subplots(nrows=1, ncols=number_of_plots)
df_list = []
vlen = len(vector["noise_ratios"][0])
size = len(vector['features'])

for j in range(number_of_plots):
    df_list.append((pd.DataFrame(vector["noise_ratios"].reshape(size, vlen)[j], range(vlen)),
                    pd.DataFrame(vector["features"].reshape(size, vlen)[j],range(vlen))
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

vlen = len(vector["noise_ratios"][0])
size = len(vector['features'])


positive = []
negative = []

for i in range(size):
    vlen = len(vector["noise_ratios"][0])
    snrs = pd.DataFrame(vector["noise_ratios"].reshape(size, vlen)[i], range(vlen))
    std = snrs.std()
    mean = snrs.mean()
    th = mean[0] + std[0]
    num_outliers = len(list(filter(lambda x: x[0] > th, snrs.values)))
    if vector["pick"][i] is not None:
        positive.append(num_outliers / vlen)
    else:
        negative.append(num_outliers / vlen)


kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

plt.hist(positive, **kwargs, color='g', label='Positive')
plt.hist(negative, **kwargs, color='b', label='Negative')
plt.gca().set(title='Histogram', ylabel='Frequency')
plt.xlim(0, 0.2)
plt.legend()
plt.show()
