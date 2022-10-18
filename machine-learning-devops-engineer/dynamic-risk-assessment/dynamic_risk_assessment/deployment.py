"""This script copies trained model components into deployment folder.

Run from the repository root as:
python dynamic_risk_assessment/deployment.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/deployment.py --config_path conf/config.json
"""
import shutil


def store_model_into_pickle(paths: dict) -> None:
    """Copy the model, metrics, and ingestion info into deployment folder.

    Arguments
    ---------
    paths:
        Collection of relevant path names.
    """

    # Copy trained model.
    shutil.copy(
        src=paths["trained_model"],
        dst=paths["deployed_model"]
    )

    # Copy F1 performance metrics.
    shutil.copy(
        src=paths["model_score"],
        dst=paths["deployed_score"]
    )

    # Copy ingestion record.
    shutil.copy(
        src=paths["ingested_record"],
        dst=paths["deployed_record"]
    )
