"""
This script performs some simple reporting steps on the trained model.

Run from the repository root as:
python dynamic_risk_assessment/reporting.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/reporting.py --config_path conf/config.json
"""
import argparse
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from dynamic_risk_assessment.utils import load_config
from dynamic_risk_assessment.diagnostics import model_predictions


def plot_confusion_matrix(
    model_path: str,
    data_path: str,
    save_path: str,
    features: List[str],
    label: str
    ) -> None:
    """Plot confusion matrix and save to file.

    Arguments
    ---------
    model_path:
        Folder path to a trained model pickle.

    data_path:
        Folder path to a csv file used for model evaluation.

    save_path:
        Where to store the confusion matrix.

    features:
        Dataframe columns treated as training features.

    label:
        Dataframe column treated as training label.

    """

    preds, y_test = model_predictions(model_path, data_path, features, label)

    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig(save_path)


def main(args):
    """Run model scoring."""

    cfg = load_config(args.config_path)

    plot_confusion_matrix(
        model_path=cfg["paths"]["deployed_model"],
        data_path=cfg["paths"]["test_data"],
        save_path=cfg["paths"]["reports"],
        features=cfg["data"]["features"],
        label=cfg["data"]["label"]
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the model.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
