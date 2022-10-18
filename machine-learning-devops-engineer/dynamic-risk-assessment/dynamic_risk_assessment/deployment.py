"""This script copies trained model components into deployment folder.

Run from the repository root as:
python dynamic_risk_assessment/deployment.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/deployment.py --config_path conf/config.json
"""
import shutil
import argparse
from dynamic_risk_assessment.utils import load_config


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

def main(args):
    """Copy model components into deployment."""
    cfg = load_config(args.config_path)
    store_model_into_pickle(cfg["paths"])


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
