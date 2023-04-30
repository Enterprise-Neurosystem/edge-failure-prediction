from kafka import KafkaConsumer
import json
import os
from datetime import datetime


class KafkaDataService:
    sensor_names_list = [
        "sensor_01",
        "sensor_02",
        "sensor_03",
        "sensor_04",
        "sensor_05",
        "sensor_06",
        "sensor_07",
        "sensor_08",
        "sensor_09",
        "sensor_10",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_16",
        "sensor_17",
        "sensor_18",
        "sensor_19",
        "sensor_20",
        "sensor_21",
        "sensor_22",
        "sensor_23",
        "sensor_24",
        "sensor_25",
        "sensor_26",
        "sensor_27",
        "sensor_28",
        "sensor_29",
        "sensor_30",
        "sensor_31",
        "sensor_32",
        "sensor_33",
        "sensor_34",
        "sensor_35",
        "sensor_36",
        "sensor_37",
        "sensor_38",
        "sensor_39",
        "sensor_40",
        "sensor_41",
        "sensor_42",
        "sensor_43",
        "sensor_44",
        "sensor_45",
        "sensor_46",
        "sensor_47",
        "sensor_48",
        "sensor_49",
    ]

    def __init__(self):
        # get environment variables (taken directly from kafka tutorial)

        # location of the Kafka Bootstrap Server loaded from the environment variable.
        # NOTE: Use the url below for use within the cluster
        self.KAFKA_BOOTSTRAP_SERVER = (
            "kafka-cluster-kafka-bootstrap.kafka-anomaly.svc.cluster.local"
        )
        # TODO:  Use the url below for use outside the cluster
        # self.KAFKA_BOOTSTRAP_SERVER = 'kafka-cluster-kafka-bootstrap-kafka-anomaly.apps.ieee.8goc.p1.openshiftapps.com:443'

        # SASL settings.  Defaults to SASL_SSL/PLAIN.
        # No auth would be PLAINTEXT/''
        # KAFKA_SECURITY_PROTOCOL = os.environ.get('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')
        self.KAFKA_SASL_MECHANISM = os.environ.get("KAFKA_SASL_MECHANISM", "PLAIN")
        self.KAFKA_SECURITY_PROTOCOL = "PLAINTEXT"

        # SASL username or client ID loaded from the environment variable
        self.KAFKA_USERNAME = os.environ.get("KAFKA_USERNAME")

        # SASL password or client secret loaded from the environment variable
        self.KAFKA_PASSWORD = os.environ.get("KAFKA_PASSWORD")

        # Name of the topic for the producer to send messages.
        # Consumers will listen to this topic for events.
        # self.KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "sensor-data")
        self.KAFKA_TOPIC = "sensor-data"

        # Kafka consumer group to which this consumer belongs
        now = datetime.now()
        current_time = now.strftime("%D:%H:%M:%S")
        self.KAFKA_CONSUMER_GROUP = current_time
        # self.sensor_names_list = ["sensor_" + str(i+1) for i in range(48)]

    def message_to_dict(self, msg):
        msg_dict = {"timestamp": msg[0], "machine_status": msg[2]}
        sensors = msg[1]
        # put sensors in dictionary :
        for i, name in enumerate(KafkaDataService.sensor_names_list):
            msg_dict[name] = sensors[i]
        return msg_dict

    def get_Kafka_consumer(self):
        consumer = KafkaConsumer(
            "sensor-data",
            group_id=self.KAFKA_CONSUMER_GROUP,
            bootstrap_servers=[self.KAFKA_BOOTSTRAP_SERVER],
            security_protocol=self.KAFKA_SECURITY_PROTOCOL,
            sasl_mechanism=self.KAFKA_SASL_MECHANISM,
            sasl_plain_username=self.KAFKA_USERNAME,
            sasl_plain_password=self.KAFKA_PASSWORD,
            api_version_auto_timeout_ms=30000,
            request_timeout_ms=450000,
            session_timeout_ms=20000,
            heartbeat_interval_ms=6000,
        )
        return consumer

    def get_Kafka_consumer_external(self):
        consumer = KafkaConsumer(
            "sensor-data",
            group_id=self.KAFKA_CONSUMER_GROUP,
            security_protocol="SSL",
            ssl_cafile="static/kafka_certificate/ca.crt",
            bootstrap_servers=[self.KAFKA_BOOTSTRAP_SERVER],
            sasl_plain_username=self.KAFKA_USERNAME,
            sasl_plain_password=self.KAFKA_PASSWORD,
            api_version_auto_timeout_ms=30000,
            request_timeout_ms=450000,
        )
        return consumer
