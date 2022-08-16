from csv import DictReader
import time
from services.kafka_data_service import KafkaDataService
import json


class DataSourceManager:
    """Used as a  data source that periodically yields timeseries data points

    """
    @staticmethod
    def csv_line_reader(file_name):
        """Use data from a csv to periodically yield a row of data

        :param file_name: Name of csv file as source of data
        :return: none
        ..notes:: This static method has no return.  Instead, it yields a row of data that has been read from
        a data source.  The row is yielded as a dictionary
        """
        with open(file_name, 'r') as read_obj:
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
        consumer = kafka_data_service.get_Kafka_consumer()
        # sensor_name_list = KafkaDataService.sensor_names_list
        try:
            for record in consumer:
                msg = record.value.decode("utf-8")
                topic = record.topic
                msg_list = list(json.loads(msg).values())
                id = int(record.key)
                msg_dict = kafka_data_service.message_to_dict(msg_list)
                print(id, msg_dict)
                # do something with message (display in web UI, send to database)
                if id == group_id:
                    yield msg_dict

        finally:
            print("Closing consumer...")
            consumer.close()
        print("Kafka consumer stopped.")