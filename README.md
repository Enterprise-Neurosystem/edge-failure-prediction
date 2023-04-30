# Failure Prediction App

This web app was written to serve the needs of a hands-on workshop. Much of the data preparation work is automated.

The source data was downloaded as a csv and is served via a PostgreSQL database. The prediction feature can use one of two data sources.

- `csv files` that contain prediction data with 720 data points
- `Apache Kafka` which simulates data as it would be streamed from sensors

## Prerequisites

Last tested with:

- Python 3.8
- OpenShift 4.10
- Shell: bash
- OS: mac,linux (various)
- Podman 4.4.4
- DBever 23.*

## Workshop

You can find the workshop instructions [here](docs/instructions.md)

## OpenShift Setup

You can view the default parameters for `bootstrap.sh` [here](scripts/bootstrap.sh).

For more info about interacting with the database in OpenShift look [here](database/README.md)

```
# workshop user parameters
# note: run this if you are a workshop user before bootstrap.sh

export NAMESPACE=$(oc whoami)
export DB_HOSTNAME="predict-db.edge-failure-prediction.svc.cluster.local"
export KAFKA_HOSTNAME="kafka-cluster-kafka-bootstrap.edge-kafka.svc.cluster.local"
```

```
scripts/bootstrap.sh
```

## Local Quickstart

Setup local postgres container

```
. scripts/bootstrap.sh
container_setup_db
```

Setup local development

Note: Python version 3.9+ may fail. replace `python3` with `python3.8` if needed.

The following section only needs to be *run once*

```
python3 -m venv venv

# activate your virtual env with the following
. venv/bin/activate

pip install -r src/requirements.txt
```

Run local web app

```
# reactivate your virtual env with the following
. venv/bin/activate

cd src

python3 wsgi.py
```

Run local Jupyter Notebook

```
pip install -r notebooks/requirements.txt
jupyter-lab
```

## Acknowledgements

Data preparation to train the model is an adaptation the work done by Xiaxiau: [here](https://www.kaggle.com/code/xiaxiaxu/predictmachinefailureinadvance/notebook)

Source of the data used in this application is found: [here](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

## Future Improvements

The ML model used in this application needs improvement, particularly in the prediction.  The next version will include those improvements.

## Team Members

1. Audrey Reznik
1. Cameron Garrison
1. Cory Latschkowski
1. Eli Guidera

## Meeting Information

Meetings are held every Thursday, 9-10 MST

Contact Eli Guidera (guiderae@yahoo.com) for questions/comments/contributions.
