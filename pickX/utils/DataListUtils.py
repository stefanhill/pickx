import pandas as pd


class DataListUtils:

    @staticmethod
    def get_shot_split(datalist: []) -> []:
        split_set = {}
        for instance in datalist:
            key_name = instance['path_id'] + str(instance['source_depth'])
            if key_name not in split_set.keys():
                split_set[key_name] = []
            split_set[key_name].append(instance)
        return split_set

    @staticmethod
    def get_list_elem_by_key_value(shot_datalist: [], key: str, value):
        for elem in shot_datalist:
            if elem[key] == value:
                return elem

    @staticmethod
    def get_neighbourhood_avg_values(shot_datalist: [], key: str) -> []:
        values = {}
        for elem in shot_datalist:
            trace_no = elem["trace_no"]
            value_sum = 0
            i = 0
            if trace_no > 1:
                val= DataListUtils.get_list_elem_by_key_value(shot_datalist, 'trace_no', trace_no - 1)[key]
                value_sum = value_sum + val if val is not None else value_sum
                i += 1
            if trace_no < len(shot_datalist):
                val = DataListUtils.get_list_elem_by_key_value(shot_datalist, 'trace_no', trace_no + 1)[key]
                value_sum = value_sum + val if val is not None else value_sum
                i += 1
            values[trace_no] = value_sum / i if i != 0 else 0
        return values

    @staticmethod
    def get_num_outliers(elem):
        snrs = pd.DataFrame(elem["noise_ratios"])
        std = snrs.std()
        mean = snrs.mean()
        th = mean[0] + std[0]
        return len(list(filter(lambda x: x[0] > th, snrs.values)))