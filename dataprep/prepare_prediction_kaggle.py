"""
Slice Kaggle data found in sensor.csv into partitions that start 12 hours before each failure.  All columns are saved.
These partitions are used to simulate real time data that is generated for model prediction.
"""

import pandas as pd
import numpy as np

# Read data, set 'timestamp' values as index
df = pd.read_csv("../tests/sensor.csv", index_col="timestamp", parse_dates=True)


# Column 'machine_status' has word values.
# Convert string values in target to numerics.  Notice that 'BROKEN' is mapped to 1
status_values = [
    (df["machine_status"] == "NORMAL"),
    (df["machine_status"] == "BROKEN"),
    (df["machine_status"] == "RECOVERING"),
]
numeric_status_values = [0, 1, 0.5]
df["machine_status"] = np.select(status_values, numeric_status_values, default=0)

# Get failure times
failure_times = df[df["machine_status"] == 1].index


# Write the data slices to csv files
for i, failure_time in enumerate(failure_times):
    df.loc[
        (failure_time - pd.Timedelta(seconds=60 * 60 * 12)) : failure_time, :
    ].to_csv("../tests/kaggle_prediction_data/prediction_slice" + str(i) + ".csv")
