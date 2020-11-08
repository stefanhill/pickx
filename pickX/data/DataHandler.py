from struct import unpack, pack
import os
import pickle
import time
import csv

from pickX.data.DataListQueryBuilder import DataListQueryBuilder
from pickX.utils.Utils import Utils
from pickX.data.DataSet import DataSet
from pickX.data.reflexw.Measurement import Measurement
from pickX.data.reflexw.Shot import Shot
from pickX.data.reflexw.Trace import Trace
from pickX.data.reflexw.Project import Project


class DataHandler:

    @staticmethod
    def import_dataset(directory: str, project_names_filter=None):
        """
        General function to import a project from directory
        :param directory: directory with typical project structure
        Project name -> Measurement -> Some shots that contain ROHDATA and maybe LINEDATA
        :param project_names_filter: filter to import only a subset of projects
        :return: a DataSet
        """
        if DataHandler.path_contains_pickle(directory):
            filename = DataHandler.get_id_of_all_pickles(directory + "\\pickled")[0]
            return DataHandler.unpickle(filename, path=directory)
        else:
            return DataHandler.import_dataset_binary(directory, project_names_filter=project_names_filter)

    @staticmethod
    def import_binary(filename_par: str, filename_dat: str, filename_pck: str) -> Shot:
        """
        Imports a single shot from binary reflexw data files
        :param filename_par: file path to the header file
        :param filename_dat: file path to the data file
        :param filename_pck: file path to the pick file
        :return: information of the binary files as shot object
        """

        shot = Shot()
        shot.name = filename_dat[:-4]

        fid = open(filename_par, 'rb')

        fid.read(420)
        shot.measure_samples = unpack('i', fid.read(4))[0]
        fid.read(28)
        shot.format_code = unpack('i', fid.read(4))[0]
        fid.read(28)
        shot.scans_measured = unpack('l', fid.read(4))[0]
        fid.read(12)
        shot.sample_interval = unpack('d', fid.read(8))[0] / 1000

        fid.close()

        if not (shot.format_code == 2 | shot.format_code == 3):
            return shot

        fid = open(filename_dat, 'rb')

        traces = []
        for _ in range(shot.scans_measured):
            trace = Trace()
            pos_src = 0
            pos_rec = 0
            single_trace_data = []

            trace.trace_no = unpack('i', fid.read(4))[0]
            fid.read(30)
            time_del = unpack('f', fid.read(4))[0]
            trace.delay = time_del / 1000
            fid.read(16)

            if shot.format_code == 2 | shot.format_code == 3:
                pos_src = [unpack('d', fid.read(8))[0], unpack('d', fid.read(8))[0]]
                pos_rec = [unpack('d', fid.read(8))[0], unpack('d', fid.read(8))[0]]
                fid.read(72)

            for _ in range(shot.measure_samples):
                single_trace_data.append(unpack('f', fid.read(4))[0])

            trace.receiver_depth = pos_rec[0]
            trace.source_depth = pos_src[0]
            trace.data = single_trace_data
            traces.append(trace)

        shot.traces = traces
        fid.close()

        if not os.path.isfile(filename_pck):
            filename_pck = filename_pck[:-4] + '.PCK'

        if os.path.isfile(filename_pck):
            fid = open(filename_pck, 'rb')

            fid.read(105)
            no_of_points = unpack('i', fid.read(4))[0]
            fid.read(36)

            for _ in range(no_of_points):
                pos = unpack('d' * 6, fid.read(48))
                fid.read(8)
                pick_index = list(map(lambda x: x.receiver_depth, shot.traces)).index(pos[3])
                shot.traces.__getitem__(pick_index).pick = unpack('d', fid.read(8))[0]
                fid.read(20)

            fid.close()

        return shot

    @staticmethod
    def import_dataset_binary(directory: str, project_names_filter=None) -> DataSet:
        """
        Imports a project from binary reflexw data files
        :param directory: path to pickled on machine where data is stored in binary format
        :param project_names_filter:  None by default, list of project names to filter
        :return: DataSet filled with measurements etc..
        """
        dataset = DataSet()
        projects = []
        for project_folder in os.listdir(directory):

            project = Project()
            project.name = project_folder
            measurements = []
            if project_names_filter is not None and project.name not in project_names_filter:
                del project
                continue
            print("preparing: {}".format(project.name))
            project_path = directory + "\\" + project_folder
            for measurement_folder in os.listdir(project_path):
                measurement = Measurement()
                measurement.name = measurement_folder
                shots = []

                measurement_path = project_path + "\\" + measurement_folder
                for file in os.listdir(measurement_path + "\\ROHDATA"):
                    if file.endswith(".DAT"):
                        shot_name = file[:-4]
                        filename_par = measurement_path + "\\ROHDATA\\" + str(shot_name) + ".PAR"
                        filename_dat = measurement_path + "\\ROHDATA\\" + str(shot_name) + ".DAT"
                        filename_pck = measurement_path + "\\LINEDATA\\" + str(shot_name) + ".pck"

                        if os.path.isfile(filename_par) & os.path.isfile(filename_dat):
                            shot = DataHandler.import_binary(filename_par, filename_dat, filename_pck)
                            if len(shot.traces) > 0:
                                shot.name = shot_name
                                shots.append(shot)
                            del shot

                measurement.shots = shots
                if len(measurement.shots) > 0:
                    measurements.append(measurement)
                del measurement

            project.measurements = measurements
            if len(project.measurements) > 0:
                projects.append(project)
            del project

        dataset.projects = projects
        return dataset

    @staticmethod
    def export_pck(trace_no: [int], receiver_depth: [float], source_depth: [float], pick_value: [float], filepath: str,
                   filename: str):
        """
        writes a single pick file with information of same sized column vectors
        :param trace_no: array of trace numbers
        :param receiver_depth: array of receiver depths
        :param source_depth: array of source depths
        :param pick_value: array of the pick positions
        :param filepath: path to write the file
        :param filename: filename (of the shot) to refer to the right pick file
        :return: None
        """

        fid = open(filepath + filename, 'wb')
        no_of_picks = int(len(pick_value))

        fid.write(Utils.pack_string('0'))
        fid.write(Utils.pack_string(' ' * 40))
        fid.write(Utils.pack_string('5'))
        fid.write(Utils.pack_string('METER' + ' ' * 15))
        fid.write(Utils.pack_string('2'))
        fid.write(Utils.pack_string('ms' + ' ' * 18))
        fid.write(Utils.pack_string('8'))
        fid.write(Utils.pack_string(filename + ' ' * (20 - len(filename))))
        fid.write(Utils.pack_string('0'))

        fid.write(pack('i', no_of_picks))
        fid.write(pack('2i', 0, 0))
        fid.write(pack('i', 255))  # color of pick
        fid.write(pack('6i', 0, 0, 0, 0, 0, 0))

        for i in range(no_of_picks):
            fid.write(pack('6d', source_depth[i], 0, 0, receiver_depth[i], 0, 0))
            fid.write(pack('d', receiver_depth[i]))
            fid.write(pack('d', pick_value[i]))
            fid.write(pack('2i', 0, 0))
            fid.write(pack('i', trace_no[i]))
            fid.write(pack('2i', 0, 0))

        fid.close()

    @staticmethod
    def export_pck_from_datalist(data_vectors, path_prefix, pick_probability: float = 0.9):
        """
        exports a whole data list into a project pickled
        :param data_vectors: data vectors object as in DataListQueryBuilder
        :param path_prefix: project pickled
        :param pick_probability: threshold the pick_probability to be a pick
        :return: a summary of all written pick files
        """
        summary = []

        export_sets = DataListQueryBuilder() \
            .init_with_data_vectors(data_vectors) \
            .group_by_column('path_id')

        for exp in export_sets:
            num_total = len(exp.datalist)
            exp_vectors = exp.filter_continuous_by('pick_probability', '>', pick_probability) \
                .get_data_vectors(auto_normalize=False)
            if len(exp_vectors['path_id']):
                file_id_split = exp_vectors['path_id'][0].split('/')
                filepath = '/'.join(file_id_split[:-1]) + '/LINEDATA/'
                Utils.mkdir_rec(filepath.split('/'), path_prefix)
                filename = file_id_split[-1:][0] + '.PCK'
                DataHandler.export_pck(exp_vectors['trace_no'], exp_vectors['receiver_depth'],
                                       exp_vectors['source_depth'], [0 for _ in range(len(exp_vectors['trace_no']))],
                                       path_prefix + '/' + filepath, filename)
            num_pos = len(exp_vectors['path_id'])
            summary.append([
                exp_vectors['path_id'][0],  # path_id
                num_total,  # total number of traces
                num_pos,  # number of predicted positive
                num_total - num_pos,  # number of predicted negative
                num_pos / num_total if num_total != 0 else 0  # positive / negative ratio
            ])

        summary.sort(key=lambda x: x[0])

        return summary

    @staticmethod
    def export_summary(summary: [], path: str):
        """
        writes a csv file from summary
        :param summary: summary object containing a 2d array with stats
        :param path: project pickled
        :return: None
        """
        csv_out = [['shot', 'no_traces', 'pred_pos', 'pred_neg', 'ratio']] + summary
        with open(path + '/' + '{}_summary.csv'.format(time.time()), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(csv_out)

    @staticmethod
    def pickle(obj: object, id: str, path=""):
        """
        Saves an object as pickle data dump
        :param obj: object to be saved, coudl be anything, but most of time data vectors
        :param id: nae of dump file
        :param path: path to folder where dump is should be stored
        :return: void
        """
        directory = path if path != "" else os.curdir
        if "pickled" not in os.listdir(directory):
            os.mkdir(directory + "\\pickled")
        directory += "\\pickled\\"

        if id in DataHandler.get_id_of_all_pickles(directory):
            answer = input("given id is already taken, do you want to overwrite {}.pickle? y/n".format(id))
            if answer.lower() != 'y':
                quit()

        try:
            with open("{}{}.pickle".format(directory, id), "wb") as f:
                pickle.dump(obj, f)
                f.close()
        except:
            print("couldn't dump dataset")

    @staticmethod
    def get_id_of_all_pickles(directory) -> list:
        """
        Returns a list of all pickled files in a given directory
        :param directory: directory to be searched
        :return: a list of filenames
        """
        ids = []
        for file in os.listdir(directory):
            if file.endswith('.pickle'):
                ids.append(str(file)[:file.rindex('.')])
        return ids

    @staticmethod
    def unpickle(id: str, path="") -> object:
        """
        converts a pickle dump into a python object
        :param id: filename of the pickle dump
        :param path: path to the dump file
        :return: the converted object
        """
        # get names of pickles in folder "pickled"
        directory = path + "\\" + "pickled\\" if path != "" else "pickled\\"

        list_of_pickles = DataHandler.get_id_of_all_pickles(directory)
        if id not in list_of_pickles:
            raise ValueError("couldn't find given id in folder {}".format(path))
        else:

            try:
                with open("{}{}.pickle".format(directory, id), "rb") as f:
                    ret = pickle.load(f)
                    f.close()
                    return ret
            except:
                print("pickle error, couldn't retrieve object")

    @staticmethod
    def path_contains_pickle(path: str) -> bool:
        """
        Checks whether a path contains a pickle dump
        :param path: path to be checked
        :return: a boolean
        """
        filenames = [str(x) for x in os.listdir(path)]
        for filename in filenames:
            if filename == 'pickled':
                return True
        return False
