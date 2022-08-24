# Edge Anomaly Prediction - Workshop Instructions
<details>
<summary>Table of Contents</summary>
<p>

* [Preparing the Data](#preparing-the-data)
* [Training the Model](#training-the-model)
* [Testing the Model](#testing-the-model)
* [Making a Prediction with the Model](#making-a-prediction-with-the-model)
* [Streaming Sensor Data](#streaming-sensor-data)

</p>
</details>

## Preparing the Data

1. When the application first loads in your browser, a large amount of sensor data is being loaded to it. To track its progress, navigate to the **Data Prep** tab of the application where you will see a message stating **"Loading data..."** and the estimated time it will take before it is complete. 

    ![](/workshop/images/landing_page.png)

2. This data loading step consists of (1) dropping sensors that have several null values and (2) Principle Component Analysis (PCA) to reduce the dimensionality of the data, allowing our model to train faster. 

3. The resulting display labeled **Feature Choices** provides information on which sensors were dropped due to a large amount of null values and which linear combinations of sensors had the highest variance according to PCA. 

    As pictured below, we are dropping `sensor_00`, `sensor_15`, `sensor_50`, and `sensor_51` and the linear combinations of sensors, denoted with 'pc', are ordered from greatest to least variance.

    ![](/workshop/images/feature_choices.png)

    The model will be trained on the first linear combinations of sensors whose variance adds up to 95% or the first 12 pc's.

4. Once the data is fully loaded, the **Start Data Prep** button  will be enabled. Click on it, which will reshape the data into a form that the model can ingest in the background.

    ![](/workshop/images/start_data_prep.png)

6. While this processing is taking place, you will see the following progress bar:

    ![](/workshop/images/data_prep_action.png)

7. Once data preparation is complete, you will recieve a message stating, **"Data is prepared. Ready to train"** and the **Train Model** tab will be enabled. 

    ![](/workshop/images/data_is_prepared.png)

## Training the Model

1. Click on the **Train Model** tab. There are different choices for training parameters in the drop-down menu, however, for the purposes of this workshop we will proceed with the default values. 

    ![](/workshop/images/train_model.png)

2. Click on the **Train Model** button to start training the model.

    ![](/workshop/images/train_model_btn.png)

3. When training is complete, you will recieve a message stating, **"Training is finsihed. Click on Display Loss Graph Btn"**. Click on the **Display Loss Graph** to view the observe the loss graph.

    ![](/workshop/images/training_is_finished.png)

4. In addition to the graph, you will also notice that the **Train Model** and **Predict** tab.

    ![](/workshop/images/loss_graph.png)

## Testing the Model

1. Click on the **Test Model** tab.

2. Next, click on the **Test Model** button. The resulting display will look something like: 

    ![](/workshop/images/model_testing.png)

    Do not be surprised if you get a slightly different result. This is due to stochastic nature of training the model. Training will always involve uncertainties and randomness and as a result, the model is always an approximation.


## Making a Prediction with the Model

1. Click on the **Predict** tab.

2. Before you can make a prediction, you must first select a data source for the prediction data. There are two options: 

    (a) Use a CSV file which will be streamed one point at a time, simulating real time generation.  The data in the CSV file is taken from the original Kaggle data source as test data. 

    (b) Use a data stream of synthetic data with the help of Apache Kafka that also simulates the production of real time data. 

    **Make sure to select one of the options before proceeding. If you attempt to click on the Start Prediction Graph before selecting a data source you will get an error message:**

    ![](/workshop/images/model_prediction.png)

    (a) If you choose the CSV radio button, a select box will list CSV file names available. Simply select a CSV filename.  

    (b) If you choose the Kafka radio button, enter the Group ID that you have been given. 

3. After you select a data source, click on the **Start Prediction Graph** button. If you chose the Kafka radio button follow the additional instructions for [streaming sensor data](#streaming-sensor-data). 

## Streaming Sensor Data

1. If you have not already, follow the instructions for [generating sensor data](https://github.com/Enterprise-Neurosystem/edge-synthetic-data-generator/blob/main/workshop/instructions.md).

2. In the same Jupyter notebook referenced in the above instructions, `12_generate_sensor_data.ipynb`, go down to the section title **Streaming our sensor data**. Let's attach a fake timestamp to each instance of synthetic data, making it time series data, by running the first four cells in this section. 

    ![](/workshop/images/streaming_sensor_data.png)

2. Now that you've transformed your data into time series data, define the Kafka cluster credentials by running the following cell:
  
    ![](/workshop/images/kafka_connect.png)

3. Finally, stream your data by running the remaining two cells, which (1) connects to the Kafka cluster based on the credentials you defined in the previous step, (2) initializes a KafkaProducer object, (3) streams your data to the sensor failure prediction model. 

    ![](/workshop/images/produce_data.png)






