from csv import DictReader
import time
from services.kafka_data_service import KafkaDataService
import json


class DataSourceManager:
    """Used as a  data source that periodically yields time series data points
    NOTE:  In the method, get_kafka_data() there are two options for getting a consumer.
            The default is already set.  If this code is to be run outside the cluster,
            comment out line 39 and uncomment line 37

    """

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

    @staticmethod
    def get_kafka_data(group_id):
        """
        Create a generator that yields one row at a time of raw sensor data
        """
        kafka_data_service = KafkaDataService()
        # TODO: The code below is for use when consumer is external to the cluster
        # consumer = kafka_data_service.get_Kafka_consumer_external()
        # TODO: The code belos is for use when consumer is inside cluster
        consumer = kafka_data_service.get_Kafka_consumer()

        try:
            for record in consumer:
                msg = record.value.decode("utf-8")
                topic = record.topic
                msg_list = list(json.loads(msg).values())
                id_g = int(record.key)
                msg_dict = kafka_data_service.message_to_dict(msg_list)

                if id_g == int(group_id):
                    print("YIELDING: {}".format(msg_dict))
                    yield msg_dict

        finally:
            print("Closing consumer...")
            consumer.close()
        print("Kafka consumer stopped.")
