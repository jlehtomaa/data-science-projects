"""
App interface for interacting with the deployed ML model.

Serve the app by running `python app.py`.
"""
import json
import pickle
import pandas as pd
from flask import Flask, request
from dynamic_risk_assessment.scoring import score_model
from dynamic_risk_assessment.training import process_data
from dynamic_risk_assessment.diagnostics import (model_predictions,
                                                 dataframe_summary,
                                                 missing_data_share,
                                                 execution_time,
                                                 list_outdated_packages)

app = Flask(__name__)

with open("./conf/config.json", "r", encoding="utf-8") as config_path:
    config = json.load(config_path)


@app.route("/")
def index():
    """App root."""
    user = request.args.get("user", default=None)
    return "Hello!\n" if user is None else f"Hello {user}!\n"


@app.route("/prediction", methods=['POST','OPTIONS'])
def prediction():
    """Prediction endpoint.

    Takes as input a dataset location and returns the model predictions.
    """

    data_path = request.json.get("data_path")

    preds, labels = model_predictions(
        model_path=config["paths"]["deployed_model"],
        data_path=data_path,
        features=config["data"]["features"],
        label=config["data"]["label"]
    )

    return {"Predictions": str(preds), "Labels": str(labels)}


@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    """Model scoring endpoint.

    Returns the F1 score of the deployed model on the test dataset.
    """

    # Load model
    with open(config["paths"]["deployed_model"], "rb") as file:
        model = pickle.load(file)

    # Read in test datasets.
    test_data = pd.read_csv(config["paths"]["test_data"])
    x_test, y_test = process_data(
        test_data, config["data"]["features"], config["data"]["label"])

    score = score_model(model, x_test, y_test, config["paths"]["model_score"])

    return {"F1 score": str(score)}


@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    """Data summary.

    Returns main summary statistics of the ingested dataset.
    """

    summary = dataframe_summary(
        path=config["paths"]["ingested_data"],
        stats=config["diagnostics"]["statistics"])

    return {"Summary statistics": str(summary)}


@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """Model diagnostics endpoint.

    Returns the folllowing model diagnostics:
    - Share of missing data per column in the ingested data.
    - Execution times of the data ingestion and model training scripts.
    - list of outdated dependencies.
    """
    missing_share = missing_data_share(config["paths"]["ingested_data"])
    timing = execution_time(config["diagnostics"]["timed_modules"])
    outdated = list_outdated_packages()

    return {"Missing share": missing_share,
            "Execution time": timing,
            "Outdated": outdated}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
