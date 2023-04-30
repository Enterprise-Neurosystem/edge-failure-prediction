# Failure Prediction App

This web app is was written to serve the needs of a hands on workshop where much of the data preparation work is automated and is in read only form.

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

You can view the default parameters for `bootstrap.sh` [here](scripts/bootstrap.sh)

```
# workshop user parameters
# note: run this if you are a workshop user before bootstrap.sh

export NAMESPACE=$(oc whoami)
export DB_HOSTNAME="predict-db.edge-failure-prediction.svc.cluster.local"
```

```
scripts/bootstrap.sh
```

## Local Quickstart

Setup local postgres db

```
cd database

. ./setup.sh
setup_container

cd ..
```

Setup local development

```
# note: python3.9+ may fail
python3 -m venv venv

. venv/bin/activate
pip install -r src/requirements.txt
```

Run app locally

```
cd src

python3 wsgi.py
```

## Acknowledgements

Data preparation to train the model is an adaptation the work done by Xiaxiau: [here](https://www.kaggle.com/code/xiaxiaxu/predictmachinefailureinadvance/notebook)

Source of the data used in this application is found: [here](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

The source data was downloaded as a csv and is served via a PostgreSQL database. The prediction feature can use one of two data sources.

- `csv files` that contain prediction data with 720 data points
- `Apache Kafka` which simulates data as it would be streamed from sensors

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
