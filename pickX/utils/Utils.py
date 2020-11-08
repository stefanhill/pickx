import math
from struct import pack
import os
import numpy as np


class Utils:

    @staticmethod
    def next_e2_of(x):
        """
        Returns the next higher 2^x value from number
        :param x: number
        :return: a number e.g. 2048
        """
        if x > 0:
            for i in range(1, 40):
                if math.pow(2, i) > x:
                    return math.pow(2, i)
        return 0

    @staticmethod
    def pack_string(string: str):
        """
        Packs a string into binary format
        :param string: string to be packed
        :return: binary string as char set
        """
        p = pack('')
        for s in string:
            p += pack('c', str.encode(s))
        return p

    @staticmethod
    def mkdir_rec(folders: [str], path_prefix):
        """
        build a path of directories that does not exist
        :param folders: an ordered list of directories
        :param path_prefix: the path where to build this stuff
        :return: true if finished
        """
        if len(folders) == 0:
            return True
        folder = folders[:1][0]
        if folder == '':
            return True
        if not os.path.isdir(path_prefix + '/' + folder):
            os.mkdir(path_prefix + '/' + folder)
        return Utils.mkdir_rec(folders[1:], path_prefix + '/' + folder)

    @staticmethod
    def interpolate(y: [int], sample_interval: int):
        """
        interpolates a data list for on a given interval
        :param y: a list of data values
        :param sample_interval: decimal sampling rate
        :return: an interpolated data list
        """
        x = np.linspace(0, 1, len(y))
        x_interp = np.linspace(0, 1, sample_interval)
        return list(np.interp(x_interp, x, np.array(y))), len(y) / sample_interval

    @staticmethod
    def generate_noise_ratios(data: [float], window_size: int) -> [float]:
        noise_ratios = []
        for i in range(len(data)):
            if window_size <= i < len(data) - window_size:
                p = data[i - window_size:i]
                n = data[i + 1:i + window_size + 1]
                pv = abs(max(p) - min(p))
                nv = abs(max(n) - min(n))
                noise_ratio = nv / pv if pv != 0 else 0
                noise_ratios.append(noise_ratio)
        return noise_ratios

    @staticmethod
    def get_num_lt(l: [float], th: float):
        return len(list(filter(lambda x: x < th, l)))