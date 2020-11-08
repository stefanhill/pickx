class Measurement:
    """
    A measurement is a collection of shots, which are produced in the same series
    """
    name = None

    shots = []

    def get_sample_interval(self):
        """
        Get the sample interval of the measurement series
        :return: a number, usually 16kHz or 32 kHz
        """
        if len(self.shots) > 0:
            return self.shots[0].sample_interval
        else:
            return -1

    def get_measure_samples(self):
        """
        Returns the number of measured samples
        :return: a number, usually 1024, 2048 or 4096
        """
        if len(self.shots) > 0:
            return self.shots[0].measure_samples
        else:
            return -1
