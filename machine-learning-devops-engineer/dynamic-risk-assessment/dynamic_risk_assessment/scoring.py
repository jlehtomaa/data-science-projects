"""Evaluate the classifier performace.

Run from the repository root as:
python dynamic_risk_assessment/scoring.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/scoring.py --config_path conf/config.json
"""
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import ClassifierMixin
from dynamic_risk_assessment.utils import load_config
from dynamic_risk_assessment.training import process_data

def score_model(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
    ) -> float:
    """Evaluate the classifier performance.

    Arguments
    ---------
    model:
        A trained scikit-learn classifier.

    x_test:
        Input test features.

    y_test:
        Test labels

    output_path:
        Where to store the evaluation metric(s).

    Returns
    -------
    The F1 score of the classifier.
    """

    # Score the model.
    preds = model.predict(x_test)
    score = metrics.f1_score(preds, y_test)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(str(score))

    return score


def main(args):
    """Model scoring procedure."""
    cfg = load_config(args.config_path)

    # Load model.
    with open(cfg["paths"]["trained_model"], "rb") as file:
        model = pickle.load(file)

    # Read in test datasets.
    test_data = pd.read_csv(cfg["paths"]["test_data"])
    x_test, y_test = process_data(
        test_data, cfg["data"]["features"], cfg["data"]["label"])

    score_model(model, x_test, y_test, cfg["paths"]["model_score"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Score the model.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
