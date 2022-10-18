"""
This script implements an automated ingestion/re-training/re-deployment flow.
"""
import os
import pickle
import logging
from dynamic_risk_assessment.scoring import score_model
from dynamic_risk_assessment.training import process_data
from dynamic_risk_assessment.ingestion import merge_dataframes
from dynamic_risk_assessment.utils import load_config, list_csv_files

logging.basicConfig(filename='logfile',
                    filemode='w',
                    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )

log = logging.getLogger(__name__)

def check_data(cfg: dict) -> bool:
    """Check if the data folder has new files.

    Arguments
    ---------
    cfg:
        Model configuration setup.

    Returns
    -------
    data_up_to_date:
        True if no new data were found, else False.
    """

    # Read the old ingestion records:
    with open(cfg["paths"]["ingested_record"], "r", encoding="utf-8") as file:
        record = file.readlines()
        record = [line.rstrip() for line in record] # strip \n from the end.

    # Read the contents of the source data folder
    source = list_csv_files(cfg["paths"]["input_data"])

    # Compare the datasets.
    data_up_to_date = set(source).issubset(set(record))

    return data_up_to_date


def main(cfg):
    """Main automation loop."""

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
    root = "python dynamic_risk_assessment"
    # Re-train.
    os.system(f"{root}/training.py")

    # Diagnostics.
    os.system(f"{root}/diagnostics.py")

    # Re-deployment.
    os.system(f"{root}/deployment.py")

    # Reporting.
    os.system(f"{root}/reporting.py")

    # API calls.
    os.system(f"{root}/apicalls.py")

    log.info("Finished.")


if __name__ == "__main__":

    CONFIG_PATH = "./conf/config.json"
    CFG = load_config(CONFIG_PATH)
    main(CFG)
