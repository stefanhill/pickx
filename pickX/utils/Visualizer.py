import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Visualiser:

    def __init__(self):
        pass

    @staticmethod
    def visualize(datalist_element, verbose=False) -> ():
        length = len(datalist_element["data"])
        """ TODO: pick visualization not working...
        :param datalist_element: element from datalist containing keys data, noise_ratios ...
        :return:
        """
        if datalist_element["noise_ratios"] == []:
            raise ValueError("noise ratios were not computed yet")

        df1 = pd.DataFrame(datalist_element["data"])
        df2 = pd.DataFrame(datalist_element["noise_ratios"])


        std = df2.std()
        mean = df2.mean()
        th = mean[0]+std[0]

        if datalist_element["pick"] is not None:
            pick_index = int((datalist_element["pick"]/1000-datalist_element["delay"])\
                             /datalist_element["sample_interval"])
        else:
            pick_index = 0

        fig, axes = plt.subplots(nrows=1, ncols=2)

        # draw data and noise_ratios
        df2.plot(ax=axes[0], label="noise_ratios")
        df1.plot(ax=axes[1], label="data")

        # draw pick at index pick_index
        axes[1].scatter([pick_index], [datalist_element["data"][pick_index]], marker='o', color='r')

        # draw std, mean and threshhold
        std_line = Line2D(list(range(length)), [std for _ in range(length)], color='r', label="std")
        mean_line = Line2D(list(range(length)), [mean for _ in range(length)], color='g', label='mean')
        threshhold_line = Line2D(list(range(length)), [th for _ in range(length)], color = 'b', label="threshhold=mean+std")

        axes[0].add_line(std_line)
        axes[0].add_line(mean_line)
        axes[0].add_line(threshhold_line)
        axes[0].legend()

        if verbose:
            print("threshhold: {}\nmean : {}\nstd: {}\n".format(th, mean[0], std[0]))
            print("{} indeces exceeded the threshhold".format(len(list(filter(lambda x: x>th, datalist_element["data"])))))
            print("pick was made at: {}".format(pick_index))

        plt.show()





