import pandas as pd
import matplotlib.pyplot as plt
import copy

from pickX.data.DataHandler import DataHandler

path = "C:\git\pickx\data\Hessigheim"
vector = DataHandler.import_dataset(path)

vlen = len(vector["noise_ratios"][0])
size = len(vector['features'])

k = 5


# split into train test set
def get_fold(split_set, pos):
    copy_set = copy.deepcopy(split_set)
    vs = copy_set.pop(pos)
    ts = []
    for elem in copy_set:
        ts += elem
    return ts, vs


# get the estimations in pos neg
def predict(ds):
    s = len(ds)
    positive_pred = []
    negative_pred = []
    vlen = len(ds[0]["noise_ratios"])
    for h in range(s):
        snrs = pd.DataFrame(ds[h]["noise_ratios"].reshape(vlen, 1), range(vlen))
        std = snrs.std()
        mean = snrs.mean()
        th = mean[0] + std[0]
        num_outliers = len(list(filter(lambda x: x[0] > th, snrs.values)))
        if ds[h]["pick"] is not None:
            positive_pred.append(num_outliers / vlen)
        else:
            negative_pred.append(num_outliers / vlen)
    return positive_pred, negative_pred


def get_acc_for_cut(positives, negatives, cut):
    plow = list(filter(lambda x: x < cut, positives))
    phigh = list(filter(lambda x: x >= cut, positives))
    nlow = list(filter(lambda x: x < cut, negatives))
    nhigh = list(filter(lambda x: x >= cut, negatives))
    return len(nhigh + plow) / len(nlow + nhigh + plow + phigh)


# generate a split set
split_set = [[] for _ in range(k)]
for i in range(size):
    split_set[i % k].append({
        'features': vector['features'][i],
        'noise_ratios': vector['noise_ratios'][i],
        'pick': vector['pick'][i]
    })

positive = []
negative = []

opts = []

for i in range(k):
    training_set, validation_set = get_fold(split_set, i)
    training_pos, training_neg, = predict(training_set)
    validation_pos, validation_neg = predict(validation_set)
    # grid search over range to find the best threshold
    r = [x / 1000 for x in range(50, 150)]
    accs = [get_acc_for_cut(training_pos, training_neg, j) for j in r]
    max_acc_pos = accs.index(max(accs))
    print(max(accs))
    val_acc = get_acc_for_cut(validation_pos, validation_neg, max_acc_pos)
    print(val_acc)
    opts.append(max_acc_pos)

optimal_threshold = sum(opts) / len(opts) + 50

"""
kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

plt.hist(positive, **kwargs, color='g', label='Positive')
plt.hist(negative, **kwargs, color='b', label='Negative')
plt.gca().set(title='Histogram', ylabel='Frequency')
plt.xlim(0, 0.2)
plt.legend()
plt.show()
"""
