from kafka import KafkaConsumer
import json
import os


class KafkaDataService:
    sensor_names_list = ["sensor_" + str(i + 1) for i in range(48)]

    def __init__(self):
        # get environment variables (taken directly from kafka tutorial)

        # location of the Kafka Bootstrap Server loaded from the environment variable.
        # e.g. 'my-kafka-bootstrap.namespace.svc.cluster.local:9092'
        self.KAFKA_BOOTSTRAP_SERVER = 'kafka-cluster-kafka-bootstrap.kafka-anomaly.svc.cluster.local'

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
        #self.KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "sensor-data")
        self.KAFKA_TOPIC = 'sensor-data'

        # Kafka consumer group to which this consumer belongs
        self.KAFKA_CONSUMER_GROUP = "notebook-consumer"
        #self.sensor_names_list = ["sensor_" + str(i+1) for i in range(48)]

    def message_to_dict(self, msg):
        msg_dict = {"timestamp": msg[0], "machine_status": msg[2]}
        sensors = msg[1]
        # put sensors in dictionary :
        for i, name in enumerate(KafkaDataService.sensor_names_list):
            msg_dict[name] = sensors[i]
        return msg_dict

    def get_Kafka_consumer(self):
        self_type = type(self)
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
        )
        return consumer

