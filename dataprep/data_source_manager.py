from csv import DictReader
import time


class DataSourceManager:
    """Used as a  data source that periodically yields timeseries data points"""

    @staticmethod
    def csv_line_reader(file_name):
        """Use data from a csv to periodically yield a row of data

        :param file_name: Name of csv file as source of data
        :return: none
        ..notes:: This static method has no return.  Instead, it yields a row of data that has been read from
        a data source.  The row is yielded as a dictionary
        """
        with open(file_name, "r") as read_obj:
            dict_reader = DictReader(read_obj)
            for row in dict_reader:
                # print("row in reader: {}".format(row))
                time.sleep(1 / 10)
                yield row
