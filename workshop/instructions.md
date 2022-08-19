# Edge Anomaly Prediction - Workshop Instructions
<details>
<summary>Table of Contents</summary>
<p>

* [Stream Sensor Data](#stream-sensor-data)
</p>
</details>

## Stream Sensor Data
Now that you've selected a slice of synthetic data, it's time for you to stream your data via Apache Kafka to the sensor failure prediction model for ingestion. 

1. First, attach a fake timestamp to each instance of synthetic data, making it time series data, by running the first four cells in this section. 
    ![](/workshop/images/streaming_sensor_data.png)

2. Now that you've transformed your data into time series data, define the Kafka cluster credentials by running the following cell:
  
    ![](/workshop/images/kafka_connect.png)

3. Finally, stream your data by running the remaining two cells, which (1) connects to the Kafka cluster based on the credentials you defined in the previous step, (2) initializes a KafkaProducer object, (3) streams your data to the sensor failure prediction model.

    ![](/workshop/images/produce_data.png)