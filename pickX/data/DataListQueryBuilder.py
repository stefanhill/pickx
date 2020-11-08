import random
import numpy as np
import copy
import multiprocessing

from pickX.utils.Utils import Utils
from multiprocessing.pool import ThreadPool


class DataListQueryBuilder:
    """
    builder pattern for easy operations on data sets
    sometimes multi-threaded
    """

    datalist = []
    lengths = set()
    length = 0
    noise_window_size = 12
    normalized_by_datalist = False
    normalized_by_trace = False
    is_normalized_length = False
    cpu_count = multiprocessing.cpu_count()

    def init_with_dataset(self, dataset):
        """
        initializes the query builder with a dataset
        :param dataset: DataSet from project directory
        :return: self
        """
        """self.datalist = []

        def get_trace_info(trace):

            return {
                'data': trace.data,
                'noise_ratios': [],
                'picked': trace.is_picked(),
                'pick_probability': None,
                'pick': trace.pick,
                'max_abs': trace.get_max_abs_ampl(),
                'sample_interval': shot.sample_interval,
                'delay': trace.delay,
                'path_id': project.name + '/' + measurement.name + '/' + shot.name,
                'source_depth': trace.source_depth,
                'receiver_depth': trace.receiver_depth,
                'trace_no': trace.trace_no
            }

        def get_trace_length(trace):
            return len(trace.data)

        traces = []
        for project in dataset.projects:  # preprocessing to enable multithreading
            for measurement in project.measurements:
                for shot in measurement.shots:
                    for trace in shot.traces:
                        traces.append(trace)

        pool = ThreadPool(self.cpu_count)

        self.lengths.update(pool.map(get_trace_length, traces))
        self.datalist = pool.map(get_trace_info, traces)
        pool.close()
        pool.join()"""

        self.datalist = []
        for project in dataset.projects:
            for measurement in project.measurements:
                for shot in measurement.shots:
                    for trace in shot.traces:
                        self.lengths.add(len(trace.data))
                        self.datalist.append({
                            'data': trace.data,
                            'noise_ratios': [],
                            'picked': trace.is_picked(),
                            'pick_probability': None,
                            'pick': trace.pick,
                            'max_abs': trace.get_max_abs_ampl(),
                            'sample_interval': shot.sample_interval,
                            'delay': trace.delay,
                            'path_id': project.name + '/' + measurement.name + '/' + shot.name,
                            'source_depth': trace.source_depth,
                            'receiver_depth': trace.receiver_depth,
                            'trace_no': trace.trace_no
                        })
        return self

    def init_with_data_vectors(self, data_vectors):
        """
        initializes the query builder with data vectors
        :param data_vectors: object with same sized data vectors
        :return: self
        """
        self.datalist = []

        def thread_helper(i):
            self.lengths.add(len(data_vectors['features'][i]))
            prob = data_vectors['pick_probability'][i]
            return {
                'data': data_vectors['features'][i].reshape(len(data_vectors['features'][i])).tolist(),
                'noise_ratios': data_vectors['noise_ratios'][i].reshape(len(data_vectors['noise_ratios'][i])).tolist(),
                'picked': int(data_vectors['labels'][i]),
                'pick_probability': float(prob) if prob is not None else None,
                'pick': data_vectors['pick'][i],
                'max_abs': data_vectors['max_abs'][i],
                'sample_interval': data_vectors['sample_interval'][i],
                'delay': data_vectors['delay'][i],
                'path_id': data_vectors['path_id'][i],
                'source_depth': data_vectors['source_depth'][i],
                'receiver_depth': data_vectors['receiver_depth'][i],
                'trace_no': data_vectors['trace_no'][i]
            }

        pool = ThreadPool(self.cpu_count)
        self.datalist = pool.map(thread_helper, range(len(data_vectors['features'])))
        pool.close()
        pool.join()

        return self

    def init_with_datalist(self, datalist):
        """
        simple initialization with data list for (e.g. deepcopy)
        :param datalist: datalist to be
        :return: self
        """
        self.datalist = datalist
        return self

    # getter functions

    def get_data_vectors(self, auto_normalize: bool = True):
        """
        should not be threaded, we have no guarantee the order remains
        :param auto_normalize: parameter for length normalization
        :return: an object of same sized column vectors
        """
        if not self.is_normalized_length and auto_normalize:
            self.normalize_length()
            print('INFO: The datalist is normalized by length automatically to generate consistent vectors.')
        elif not auto_normalize and self.length == 0:
            self.length = min(self.lengths)

        features = []
        noise_ratios = []
        labels = []
        picks = []
        pick_probs = []
        max_abs = []
        sample_intervals = []
        delays = []
        path_ids = []
        source_depths = []
        receiver_depths = []
        trace_nos = []

        for entry in self.datalist:
            features.append(entry['data'])
            noise_ratios.append(entry['noise_ratios'])
            labels.append(entry['picked'])
            picks.append(entry['pick'])
            pick_probs.append(entry['pick_probability'])
            max_abs.append(entry['max_abs'])
            sample_intervals.append(entry['sample_interval'])
            delays.append(entry['delay'])
            path_ids.append(entry['path_id'])
            source_depths.append(entry['source_depth'])
            receiver_depths.append(entry['receiver_depth'])
            trace_nos.append(entry['trace_no'])

        return {
            'features': np.array(features).reshape(-1, self.length, 1),
            'noise_ratios': np.array(noise_ratios).reshape(-1, self.length, 1),
            'labels': np.array(labels),
            'pick': picks,
            'pick_probability': pick_probs,
            'max_abs': max_abs,
            'sample_interval': sample_intervals,
            'delay': delays,
            'path_id': path_ids,
            'source_depth': source_depths,
            'receiver_depth': receiver_depths,
            'trace_no': trace_nos
        }

    def get_shot_datalists(self):
        dlqbs = self.group_by_column('path_id')
        return [x.datalist for x in dlqbs]

    def group_by_column(self, col: str):
        """
        group by operation for the data list
        :param col: column identifier
        :return: a list of query builders with subsets split by column identifier
        """
        split_set = {}
        for instance in self.datalist:
            if instance[col] not in split_set.keys():
                split_set[instance[col]] = []
            split_set[instance[col]].append(instance)  # ?

        return list(map(lambda x: DataListQueryBuilder().init_with_datalist(x), split_set.values()))  # ???

    # filter and action functions

    def shuffle(self):
        """
        shuffles the whole data list
        :return: self
        """
        random.shuffle(self.datalist)
        return self

    def normalize_ampl_by_datalist(self):
        """
        normalizes the amplitude of features by the maximum of the whole data list
        :return: self
        """
        max_ampl = max(list(map(lambda x: x['max_abs'], self.datalist)))

        def lfn(x):
            x['data'] = list(map(lambda y: y / max_ampl, x['data'])) if max_ampl != 0 \
                else [0 for _ in x['data']]
            return x

        pool = ThreadPool(self.cpu_count)
        self.datalist = pool.map(lambda x: lfn(x), self.datalist)

        pool.close()
        pool.join()
        self.normalized_by_trace = False
        self.normalized_by_datalist = True
        return self

    def normalize_ampl_by_trace(self):
        """
        normalizes the amplitude for each trace by its own maximum
        :return: self
        """
        for entry in self.datalist:
            entry['data'] = [x / entry['max_abs'] for x in entry['data']] if entry['max_abs'] != 0 \
                else [0 for _ in entry['data']]
        self.normalized_by_trace = True
        self.normalized_by_datalist = False
        return self

    def normalize_length(self, pos=-1, hard_cut=False, puffer=1.5):
        """
        normalizes the lengths of all traces in the data list
        :param pos: position where to cut
        :param hard_cut: if set to true puffer will be overwritten by the minimal trace length
        :param puffer: factor of trace which will be used from the last pick ongoing
        :return: self
        """
        if pos == -1:
            pos = self.find_optimal_cut_pos(hard_cut=hard_cut, puffer=puffer)

        def lfn(x):
            if len(x['data']) >= pos:
                x['data'] = x['data'][:pos]
                x['noise_ratios'] = x['noise_ratios'][:pos]
            else:
                x['data'], scaling_factor = Utils.interpolate(x['data'], pos)
                x['pick'] = x['pick'] * scaling_factor
                x['noise_ratios'], _ = Utils.interpolate(x['noise_ratios'], pos)
            return x

        self.datalist = list(map(lambda x: lfn(x), self.datalist))
        self.lengths = {pos}
        self.length = pos
        self.is_normalized_length = True
        return self

    def balance(self):
        """
        removes instances from the datalist to balance 50:50 positive:negative samples
        :return: self
        """
        picked = copy.deepcopy(self) \
            .is_picked(True)
        not_picked = copy.deepcopy(self) \
            .is_picked(False)
        minsize = min(picked.get_size(), not_picked.get_size())
        self.datalist = picked.datalist[:minsize] + not_picked.datalist[:minsize]
        return self

    def generate_noise_ratios(self, noise_window_size: int):
        self.noise_window_size = noise_window_size
        padding = [1.0 for _ in range(self.noise_window_size)]

        def lfn(x):
            x['noise_ratios'] = padding + Utils.generate_noise_ratios(x['data'], self.noise_window_size) + padding
            return x

        self.datalist = list(map(lambda x: lfn(x), self.datalist))
        return self

    def is_picked(self, picked):
        """
        filters by pick
        :param picked: either picked or not
        :return: self
        """
        self.datalist = list(filter(lambda x: x['picked'] == picked, self.datalist))
        return self

    def filter_continuous_by(self, col, fn, val, ignore_none=True):
        """
        generic filter function to filter data lists
        :param col: column to apply filter
        :param fn: filter function ['<', '==', '>'] as string
        :param val: threshold value for the filter function
        :param ignore_none: should None values be included in the result
        :return: self
        """

        def lfn(x):
            if x[col] is not None:
                if fn == '<':
                    return x[col] < val
                elif fn == '>':
                    return x[col] > val
                elif fn == '==':
                    return x[col] == val
                else:
                    return False
            else:
                if ignore_none:
                    return False
                else:
                    return True

        self.datalist = list(filter(lambda x: lfn(x), self.datalist))
        return self

    # helper functions

    def find_optimal_cut_pos(self, hard_cut=False, puffer=1.5):
        """
        finds the optimal position for a cut by multiplying the last pick with the puffer
        :param hard_cut: if set to true puffer will be overwritten by the minimal trace length
        :param puffer: factor of trace which will be used from the last pick ongoing
        :return: optimal position as integer 2^x
        """
        latest_pick = max(
            list(map(lambda x: (x['pick'] / 1000 - x['delay']) / x['sample_interval']if x['pick'] is not None \
                else 0, self.datalist)))
        max_puffer = min(self.lengths)
        min_puffer = round(latest_pick * puffer)
        if max_puffer < min_puffer and hard_cut:
            print('WARNING: Hard cut performed. This may cause unexpected behaviour!')
            return max_puffer
        else:
            return int(min(max_puffer, Utils.next_e2_of(min_puffer)))

    def get_size(self):
        """
        Get the size of the data list in the query builder
        :return: the size of the data set
        """
        return len(self.datalist)
