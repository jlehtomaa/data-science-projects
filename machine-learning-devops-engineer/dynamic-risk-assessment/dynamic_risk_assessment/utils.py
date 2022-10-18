"""Shared utility functions."""
import os
import json
from typing import List


def load_config(path: str) -> dict:
    """Load a json-formatted config file.

    Arguments
    ---------
    path:
        Config json location.

    Returns
    -------
    config:
        A dictionary model config file.
    """

    with open(path,'r') as file:
        config = json.load(file)

    return config


def list_csv_files(src_dir: str) -> List[str]:
    """List all csv file in a directory.

    Arguments
    ---------
    src_dir:
        Directory to scan for csv files.

    Returns
    -------
    csv_files:
        A list of all detected csv files.
    """
    files = os.listdir(src_dir)
    csv_files = [os.path.join(src_dir, file) for file in files
                 if ".csv" in file]

    return csv_files


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
