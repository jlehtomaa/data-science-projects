"""
This script implements an automated ingestion/re-training/re-deployment flow.
"""
import pickle
import logging
import argparse
from dynamic_risk_assessment.scoring import score_model
from dynamic_risk_assessment.training import process_data, train_model
from dynamic_risk_assessment.ingestion import merge_dataframes
from dynamic_risk_assessment.utils import load_config, check_data
from dynamic_risk_assessment.deployment import store_model_into_pickle

logging.basicConfig(filename='logfile',
                    filemode='w',
                    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )

log = logging.getLogger(__name__)

def main(args):
    """Main automation loop."""

    cfg = load_config(args.config_path)

    log.info("Checking for new data.")
    data_ok = check_data(cfg)

    if data_ok:
        log.info("Data up to date. Exiting.")
        return

    # Read in the new data.
    log.info("Found new data. Re-merging a new dataset.")
    new_data = merge_dataframes(
        src_dir=cfg["paths"]["input_data"],
        out_path_data=cfg["paths"]["ingested_data"],
        out_path_record=cfg["paths"]["ingested_record"])

    x_test, y_test = process_data(
        new_data, cfg["data"]["features"], cfg["data"]["label"])

    # Read in the currently deployed model.
    with open(cfg["paths"]["deployed_model"], "rb") as file:
        model = pickle.load(file)

    # Evaluate the model with the new data.
    new_score = score_model(model, x_test, y_test, cfg["paths"]["model_score"])

    # Load the old score for comparison.
    with open(cfg["paths"]["deployed_score"], "r", encoding="utf-8") as file:
        old_score = float(file.read())

    # Check for model drift.
    log.info("Checking for model drift.")
    has_model_drift = (new_score < old_score)

    if not has_model_drift:
        log.info("No model drift detected. Exiting.")
        return

    log.info("Detected model drift. Starting the re-training pipeline.")

    # Re-train model.
    x_train, y_train = process_data(
        new_data, cfg["data"]["features"], cfg["data"]["label"])

    train_model(
        x_train, y_train, cfg["model"]["params"], cfg["paths"]["trained_model"])

    # Re-deploy.
    store_model_into_pickle(cfg["paths"])

    log.info("Finished.")


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
