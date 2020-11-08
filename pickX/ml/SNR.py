import pickle
from abc import ABC

import pandas as pd

from pickX.data.DataHandler import DataHandler
from pickX.ml.Classifier import Classifier
from pickX.utils.DataListUtils import DataListUtils
from pickX.utils.Utils import Utils


class SNR(Classifier, ABC):
    pred_snr = {'pos': [], 'neg': []}
    pred_snr_sum = {'pos': [], 'neg': []}
    granularity = 0.01

    pred_n = {'pos': [], 'neg': []}
    pred_n_fn_points = {'0_pos': 0, '0_neg': 1,
                        '05_pos': 0.5, '05_neg': 0.5,
                        '1_pos': 1, '1_neg': 0}

    def __train__(self, datalist):
        """
        Derives the probability dense and mass functions for a given datalist
        Shots MUST be contained completely otherwise neighbourhood consideration will not be correct
        :param datalist: data list to train wtih
        :return: void, changes in self variables
        """
        shot_datalists = DataListUtils.get_shot_split(datalist)
        for shot in shot_datalists.values():
            neighbourhood_avg_picked = DataListUtils.get_neighbourhood_avg_values(shot, 'picked')
            for elem in shot:
                noise_len = len(elem['noise_ratios'])
                num_outliers = DataListUtils.get_num_outliers(elem)
                if elem['pick'] is not None:
                    self.pred_snr['pos'].append(num_outliers / noise_len)
                    self.pred_n['pos'].append(neighbourhood_avg_picked[elem['trace_no']])
                else:
                    self.pred_snr['neg'].append(num_outliers / noise_len)
                    self.pred_n['neg'].append(neighbourhood_avg_picked[elem['trace_no']])
        self.generate_pred_snr_sum()
        self.generate_n_fn_points()
        print("passed")

    def __evaluate__(self, datalist):
        pass

    def __test__(self, datalist):
        true_classifications = 0
        false_classifications = 0
        total = len(datalist)
        shot_datalists = DataListUtils.get_shot_split(datalist)
        for shot in shot_datalists.values():
            for elem in shot:
                noise_len = len(elem['noise_ratios'])
                num_outliers = DataListUtils.get_num_outliers(elem)
                pred_sum_index = round(num_outliers / noise_len / self.granularity * 100)
                elem["pick_pos_prob"], elem["pick_neg_prob"] = self.get_pred_snr(pred_sum_index)
                elem['snr'] = pred_sum_index * self.granularity
            # n_avg_pick_pos = DataListUtils.get_neighbourhood_avg_values(shot, 'pick_pos_prob')
            # n_avg_pick_neg = DataListUtils.get_neighbourhood_avg_values(shot, 'pick_neg_prob')
            for elem in shot:
                # TODO: find out which method delivers the best accuracy values
                is_picked = elem['picked']
                """
                is_pos = elem['pick_pos_prob']
                is_neg = elem['pick_neg_prob']
                if is_pos > is_neg and is_picked == 1 or is_pos < is_neg and is_picked == 0:
                    true_classifications += 1
                else:
                    false_classifications += 1               
                """
                snr = elem['snr']
                if snr < 11.5 and is_picked == 1 or snr > 11.5 and is_picked == 0:
                    true_classifications += 1
                else:
                    false_classifications += 1

        return true_classifications / total

    def __import__(self, path):
        """

        :param path:
        :return:
        """
        try:
            with open("{}".format(path), "rb") as f:
                obj = pickle.load(f)
                self.pred_snr = obj['pred_snr']
                self.pred_snr_sum = obj['pred_snr_sum']
                self.granularity = obj['granularity']
                self.pred_n = obj['pred_n']
                self.pred_n_fn_points = obj['pred_n_fn_points']
                f.close()

        except IOError:
            print("pickle error, couldn't retrieve object")

    def __export__(self, path):
        """

        :param path:
        :return:
        """
        obj = {
            'pred_snr': self.pred_snr,
            'pred_snr_sum': self.pred_snr_sum,
            'granularity': self.granularity,
            'pred_n': self.pred_n,
            'pred_n_fn_points': self.pred_n_fn_points
        }
        try:
            with open("{}".format(path), "wb") as f:
                pickle.dump(obj, f)
                f.close()
        except IOError:
            print("failed while saving classifier")

    def generate_pred_snr_sum(self):
        for i in range(int(100 / self.granularity)):
            th = i * self.granularity / 100
            self.pred_snr_sum['pos'].append(1 - Utils.get_num_lt(self.pred_snr['pos'], th) / len(self.pred_snr['pos']))
            self.pred_snr_sum['neg'].append(Utils.get_num_lt(self.pred_snr['neg'], th) / len(self.pred_snr['neg']))

    def generate_n_fn_points(self):
        pos_total = len(self.pred_n['pos'])
        neg_total = len(self.pred_n['neg'])
        self.pred_n_fn_points = {
            '0_pos': self.pred_n['pos'].count(0) / pos_total,
            '0_neg': self.pred_n['neg'].count(0) / neg_total,
            '05_pos': self.pred_n['pos'].count(0.5) / pos_total,
            '05_neg': self.pred_n['neg'].count(0.5) / neg_total,
            '1_pos': self.pred_n['pos'].count(1) / pos_total,
            '1_neg': self.pred_n['neg'].count(1) / neg_total
        }

    def get_n_fn_value(self, x):
        if x < 0.5:
            fn_value_pos = abs(self.pred_n_fn_points['05_pos'] - self.pred_n_fn_points['0_pos']) * 0.5 * x \
                           + self.pred_n_fn_points['0_pos']
            fn_value_neg = abs(self.pred_n_fn_points['05_neg'] - self.pred_n_fn_points['0_neg']) * 0.5 * x \
                           + self.pred_n_fn_points['0_neg']
            return {
                'pos': fn_value_pos,
                'neg': fn_value_neg
            }
        else:
            fn_value_pos = abs(self.pred_n_fn_points['1_pos'] - self.pred_n_fn_points['05_pos']) * 0.5 * (x - 0.5) \
                           + self.pred_n_fn_points['05_pos']
            fn_value_neg = abs(self.pred_n_fn_points['1_neg'] - self.pred_n_fn_points['05_neg']) * 0.5 * (x - 0.5) \
                           + self.pred_n_fn_points['05_neg']
            return {
                'pos': fn_value_pos,
                'neg': fn_value_neg
            }

    def get_pred_snr(self, index):
        return self.pred_snr_sum['pos'][index], self.pred_snr_sum['neg'][index]

    def get_pred_n_fn(self, index):
        return self.get_n_fn_value(index)['pos'], self.get_n_fn_value(index)['neg']
