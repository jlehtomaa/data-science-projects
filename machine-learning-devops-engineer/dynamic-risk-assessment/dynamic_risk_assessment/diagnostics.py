"""This module runs various diagnostic tests for the data and trained model.

Run from the repository root as:
python dynamic_risk_assessment/diagnostics.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/diagnostics.py --config_path conf/config.json
"""
import os
import pickle
import timeit
import argparse
import subprocess
from typing import List, Tuple
import numpy as np
import pandas as pd
from dynamic_risk_assessment.utils import load_config
from dynamic_risk_assessment.training import process_data


def model_predictions(
    model_path: str,
    data_path: str,
    features: List[str],
    label: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Read the deployed model and a test dataset, calculate predictions.

    Arguments
    ---------
    model_path:
        Source path to read a trained model from.

    data_path:
        A source path for the input data.

    features:
        A list of column names treated as input features.

    label:
        Column name that contains the prediction label.

    Returns
    -------
    A tuple containing model predictions and the corresponding test labels.
    """

    # Load the model.
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    x_test, y_test = process_data(pd.read_csv(data_path), features, label)

    return model.predict(x_test), y_test


def dataframe_summary(
    path: str,
    stats: List[str]
    ) -> List[str]:
    """Produce summary statistics of the dataset.

    Arguments
    ---------
    path:
        Source path to read the data csv.

    stats:
        List of statistics to output, corresponding to those produced by
        the pandas df.describe() method.

    Returns
    -------
    output:
        A list containing all of the summary statistics for every numerical
        column of the input dataset.
    """

    numeric_data = pd.read_csv(path).select_dtypes(include=np.number)
    summary = numeric_data.describe().loc[stats].to_dict()

    # Rounding and formatting.
    output = []
    for feature, stat_dict  in summary.items():
        rounded = {key: round(stat, 3) for (key, stat) in stat_dict.items()}
        output.append({feature: rounded})

    return output


def missing_data_share(path: str) -> List[float]:
    """Return the share of missing values per dataframe column.

    Arguments
    ---------
    path:
        Source input csv path.

    Returns
    -------
    A list corresponding to each column in the original dataframe.
    """

    data = pd.read_csv(path)

    nans_per_col = data.isnull().sum(axis=0)
    nan_share_per_col = nans_per_col / len(data)

    return list(nan_share_per_col.values)


def execution_time(modules: List[str]) -> List[float]:
    """Time the specified modules.

    Arguments
    ---------
    modules:
        List of module paths to time.

    Returns
    -------
    results:
        A list of time measurements in seconds, one for each module in modules.
    """

    results = []
    for module in modules:
        start_time = timeit.default_timer()
        os.system(f"python {module}")
        duration = timeit.default_timer() - start_time
        results.append(round(duration, 4))

    return results


def list_outdated_packages(dep_manager: str="pip") -> str:
    """Check for out-of-date dependencies.

    Arguments
    ---------
    dep_manager:
        Dependency manager backend, from ['pip', 'poetry'], default='pip'.

    Returns
    -------
    outdated:
        Summary of out-of-date dependencies.
    """
    if dep_manager == "pip":
        outdated = subprocess.check_output(
            ["pip", "list", "--outdated"]).decode("utf-8")
    elif dep_manager == "poetry":
        outdated = subprocess.check_output(
            ["poetry", "show", "--outdated"]).decode("utf-8")
    else:
        raise ValueError("Unknown dependency manager. Use 'pip' or 'poetry'.")

    return outdated


def main(args):
    """Run all model diagnostics."""

    cfg = load_config(args.config_path)

    model_predictions(
        model_path=cfg["paths"]["deployed_model"],
        data_path=cfg["paths"]["ingested_data"],
        features=cfg["data"]["features"],
        label=cfg["data"]["label"]
    )

    dataframe_summary(
        path=cfg["paths"]["ingested_data"],
        stats=cfg["diagnostics"]["statistics"]
    )

    missing_data_share(cfg["paths"]["ingested_data"])
    execution_time(cfg["diagnostics"]["timed_modules"])
    list_outdated_packages()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run model diagnostics")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
