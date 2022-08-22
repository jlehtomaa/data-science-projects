"""
This module runs unit tests on all functions specified in the churn_library.py.
The logging details are specified in the pytest.ini file (in the same
directory) and the final logs are stored to the ./logs folder.

By default, the logging level is set to INFO.
To run the test, switch to the project root folder and run in terminal:
pytest churn_script_logging_and_tests.py

Author: JL
Date created: 2022-08-16
"""
import os
import logging
import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize
import churn_library as cl


@pytest.fixture(name="config", scope="module")
def fixture_config():
    """Fixture for experiment config dictionary."""

    # Read a hydra configuration file as a python dict.
    initialize(config_path="./conf/", job_name="run_tests", version_base=None)
    cfg_yaml = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg_yaml)

    return cfg

@pytest.fixture(name="raw_df", scope="module")
def fixture_raw_df(config):
    """Fixture for raw training data dataframe."""
    try:
        data_path = config["paths"]["raw_data"]
        data = cl.import_data(data_path)
        logging.info("SUCCESS: Created raw data fixture.")
    except FileNotFoundError as error:
        logging.error("ERROR: Raw data file not found.")
        raise error
    return data

def test_input_data(raw_df):
    """Test import_data: file is not empty."""
    error_msg = "Input data set is empty."
    try:
        assert raw_df.shape[0] > 0 and raw_df.shape[1] > 0, error_msg
        logging.info("SUCCESS: Input dataset contains rows and columns.")
    except AssertionError as error:
        logging.error("ERROR: %s", error_msg )
        raise error

def test_perform_eda(config, raw_df):
    """Test that exploratory data analysis produces all relevant figures."""

    img_path = config["paths"]["eda"]
    plot_vars = config["data_processing"]["plot_vars"]
    fig_names = [var + ".png" for var in plot_vars]

    cl.perform_eda(raw_df, img_path, plot_vars)
    files = os.listdir(img_path)

    try:
        for fig in fig_names:
            assert fig in files
        logging.info("SUCCESS: Produced all EDA figures.")
    except AssertionError as error:
        logging.error("ERROR: EDA figures missing.")
        raise error

def test_category_mean_encoder(raw_df, config):
    """Test that the category_mean_enocder generates all relevant features."""

    category_lst = config["data_processing"]["category_lst"]
    encoded_data = cl.category_mean_encoder(raw_df, category_lst)
    try:
        new_vars = [cat + "_Churn" for cat in category_lst]
        assert set(new_vars).issubset(encoded_data.columns)
        logging.info("SUCCESS: Mean encodings exist in the new data.")
    except AssertionError as error:
        logging.error("ERROR: Encoded data entries missing.")
        raise error

def test_perform_feature_engineering(raw_df, config):
    """Test that the training data splits are all non-empty."""
    preproc = config["data_processing"]
    train_data, test_data = cl.perform_feature_engineering(
        raw_df, preproc["category_lst"], preproc["feature_lst"])

    try:
        for data in train_data + test_data:
            assert len(data) > 0, "Empty data split."
        logging.info("SUCCESS: Created training data split.")
    except AssertionError as error:
        logging.error("ERROR: Empty training data split encountered.")
        raise error
