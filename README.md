# edge-prediction-failure
This web app is was written to serve the needs of a hands on workshop where much of the data preparation work is automated and is in read only form.

Also, currently the data used by the app is in csv format.  To use the Kaggle data, sensors.csv, download it from https://www.kaggle.com/datasets/nphantawee/pump-sensor-data 
and place the file in the project as /static/sensors.csv.

We will be moving the data contained in sensors.csv to a database where data really belongs.

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
