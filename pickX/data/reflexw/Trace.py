class Trace:
    """
    A trace of the seismic wave with additional information about source and receiver depth
    """
    trace_no = None
    pick = None
    delay = None
    receiver_depth = None
    source_depth = None

    data = []

    def is_picked(self):
        """
        Depicts whether the trace is already picked
        :return: a boolean
        """
        if self.pick is not None:
            return 1
        else:
            return 0

    def get_max_abs_ampl(self):
        """
        Get the maximum of the trace
        Is used to normalize over multiple traces
        :return: the largest absolute value from all data points
        """
        return max(list(map(lambda x: abs(x), self.data)))
