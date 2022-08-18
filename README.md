# edge-prediction-failure
This web app is was written to serve the needs of a hands on workshop where much of the data preparation work is automated and is in read only form.

Acknowledgement: The data preparation to train the model is an adaptation the work done by 
Xiaxiau, https://www.kaggle.com/code/xiaxiaxu/predictmachinefailureinadvance/notebook

The source of the data used in this application is found at: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data 

The source data was downloaded as a csv and then was uploaded to a PostgreSQL database.

The ML model used in this application needs improvement, particularly in the prediction.  The next version will include those improvements.

The prediction feature offers the user to use one of two data sources.  One choice is obtained from a list of csv files that each contain prediction data with 720 data points.  The other choice is to obtain the data from Apache Kafka which simulates data as it would be streamed from the sensors.

## Team Members
1. Audrey Reznik
1. Cameron Garrison
1. Cory Latschkowski
1. Eli Guidera


## Meeting Information
Meetings are held every Thursday, 9-10 MST

Contact Eli Guidera (guiderae@yahoo.com) for questions/comments/contributions.

## Quickstart

Setup local development
```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Run app locally
```
# devel
python3 wsgi.py

# gunicorn
gunicorn wsgi:application -b 0.0.0.0:8080
```
