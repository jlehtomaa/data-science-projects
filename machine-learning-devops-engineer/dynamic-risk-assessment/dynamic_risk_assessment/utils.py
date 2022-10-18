"""Shared utility functions."""
import os
import json
from typing import List

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
