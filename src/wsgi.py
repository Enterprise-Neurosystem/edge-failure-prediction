from os.path import join

from flask import Flask, render_template, Response, request, jsonify
import os
from dataprep.data_preparation import DataPreparation
from graphs.graph_manager import GraphManager
from managers.train_manager import TrainManager
from managers.test_manager import TestManager
from dataprep.process_realtime_data import ProcessRealtimeData
from utils.data_file_manager import DataFileManager
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# 'application' reference required for wgsi / gunicorn
# https://docs.openshift.com/container-platform/3.11/using_images/s2i_images/python.html#using-images-python-configuration
application = app


@app.route("/")
def main():
    """
    First create a cache directory if not exist.
    Just show the main.html without any data
    :return:
    """
    path = "static/cache/"
    if not os.path.exists(path):
        os.makedirs(path)
    csv_filenames = DataFileManager.get_file_names_in_path("static/prediction_data")

    return render_template("main.html", filenames=csv_filenames)


@app.route("/initData", methods=["POST"])
def init_data():
    """
    This is an asynchronous call so that the page, main.html can load immediately while the data it displays is
    prepared and loaded in the background.
    Create a DataPreparation object which gets the data and loads its parts into class variables.
    :return: JSON that contains the HTML that contains the data to be displayed.
    """
    DataPreparation()
    # Extract data from the appropriate class variables in the DataPreparation object.
    nulls_series = (
        DataPreparation.original_null_list
    )  # List of each sensor's count of nulls
    bad_cols = (
        DataPreparation.bad_cols
    )  # List of columns that will be dropped from the feature list
    ranked_features = DataPreparation.ranked_features  # PCA rank of the top n features
    ranked_array = ranked_features.to_numpy()
    num_features_to_include = (
        DataPreparation.num_features_to_include
    )  # How name features that will be included in the training
    feature_names_to_include = ranked_array[
        0:num_features_to_include, 0
    ]  # Feature names that will be included in the training

    # return a json form of the html that will contain the data to be displayed asynchronously.  When the javascript
    # receives the json dictionary (with one key: 'data'), it extracts the HTML and puts the HTML into the appropriate
    # div.
    return jsonify(
        {
            "data": render_template(
                "scrolldivs.html",
                nulls_series=nulls_series,
                bad_cols=bad_cols,
                ranked_features=ranked_array,
                features_in_model=feature_names_to_include,
            )
        }
    )


@app.route("/progress-shape-data")
def progress_shape_data():
    """
    The method DataPreparation.finish_data_prep() is a generator that yields progress status of the lengthy data
    preparation process.  This status information is pushed to the client.
    :return: none
    """
    DataPreparation.finish_data_prep()
    return Response(DataPreparation.finish_data_prep(), mimetype="text/event-stream")


@app.route("/train-model", methods=["GET", "POST"])
def train_model():
    """
    This is an asynchronous function that trains the model and generates the training graph stats.  The stats
    are cached in the class variable DataPreparation.train_history
    :return: An <img> tag that contains the src path to a "Finished" message
    """
    epochs = int(request.form.get("epochsSelect"))
    batch_size = int(request.form.get("batchSizeSelect"))
    learning_rate = float(request.form.get("learningRateSelect"))

    hidden_layer1_nodes = 128
    hidden_layer2_nodes = 128
    hidden_layer3_nodes = 64

    # Create a TrainManager which builds and compiles model.
    train_manager = TrainManager(
        hidden_layer1_nodes, hidden_layer2_nodes, hidden_layer3_nodes, learning_rate
    )
    # Get history of the training
    DataPreparation.train_history = train_manager.fit_model(
        DataPreparation.X_train, DataPreparation.y_train, epochs, batch_size
    )
    return "<img src='static/TrainFinished.png'/>"


@app.route("/display-train-graph", methods=["GET", "POST"])
def display_train_graph():
    """
    Get the history of the model.fit(), and graph the loss
    :return: the loss graph that has been encoded
    """
    train_history = DataPreparation.train_history
    encoded_image = GraphManager.plot_history(train_history)
    return encoded_image


@app.route("/test-model", methods=["GET", "POST"])
def test_model():
    """
    This is an asynchronous function that generates the test graph of predictions
    :return: A <div> that contains a plotly.js graph along with the javascript to enable the graph to be interactive.
    """
    buffer = TestManager.make_test_graph()
    return buffer


@app.route("/runPredict")
def run_predict():
    """
    This function is a generator for prediction
    """
    # file_name = 'static/prediction_data/prediction_slice1.csv'
    group_id = None  # Used for 'kafka' data source
    file_name = None  # Used for 'csv' data source
    predict_window_size = 20

    data_source = request.args.get("dataSource")
    if data_source == "csv":
        file_name_only = request.args.get("predictCSVFileName")
        path = "static/prediction_data"
        file_name = join(path, file_name_only)
    else:
        group_id = request.args.get("groupId")

    scaler_filename = "static/cache/training_scaler.gz"
    pca_filename = "static/cache/pca.gz"
    means_filename = "static/cache/mean.gz"
    bad_cols_filename = "static/cache/bad_cols.gz"
    model_filename = "static/cache/trained_model/saved_model.pb"

    rtd = ProcessRealtimeData(
        predict_window_size,
        scaler_filename,
        pca_filename,
        means_filename,
        bad_cols_filename,
        model_filename,
        data_source,
        csv_filename=file_name,
        group_id=group_id,
    )
    rtd.process_points()
    return Response(rtd.process_points(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")  # nosec

# run gunicorn manually
# gunicorn wsgi:application -b 0.0.0.0:8080
