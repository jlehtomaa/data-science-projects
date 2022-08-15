"""
This module contains tests to check all functions in the churn_library.py
module.
"""
import os
import logging
import pytest
import joblib
import churn_library as cl


@pytest.fixture(name="data_path", scope="module")
def fixture_data_path():
    """Fixture for raw training data path."""
    return cl.BANK_DATA_PTH

@pytest.fixture(name="categorical_lst", scope="module")
def fixture_categorical_lst():
    """Fixture for dataset categorical variables."""
    return cl.CATEGORICAL_VARS

@pytest.fixture(name="feature_names", scope="module")
def fixture_feature_names():
    """Fixture for final dataset columns used as features."""
    return cl.FEATURE_NAMES

@pytest.fixture(name="raw_df", scope="module")
def fixture_raw_df(data_path):
    """Fixture for raw training data dataframe."""
    try:
        data = cl.import_data(data_path)
        logging.info("SUCCESS: Created raw data fixture.")
    except FileNotFoundError as error:
        logging.error("ERROR: Raw data file not found.")
        raise error
    return data

@pytest.fixture(name="train_data_split", scope="module")
def fixture_train_data_split(raw_df, categorical_lst, feature_names):
    """Fixture for clean (=encoded) training data dataframe."""
    data_split = cl.perform_feature_engineering(
        raw_df, categorical_lst, feature_names)
    return data_split

def test_input_data(raw_df):
    """Test import_data: file is not empty."""
    error_msg = "Input data set is empty."
    try:
        assert raw_df.shape[0] > 0 and raw_df.shape[1] > 0, error_msg
        logging.info("SUCCESS: Input dataset contains rows and columns.")
    except AssertionError as error:
        logging.error("ERROR: %s", error_msg )
        raise error

def test_perform_eda(raw_df):
    """Test that exploratory data analysis produces all relevant figures."""

    img_path = "./images/eda/"
    fig_names = [
        "Churn.png",
        "Customer_Age.png",
        "heatmap_all.png",
        "Marital_Status.png",
        "Total_Trans_Ct.png"
    ]
    cl.perform_eda(raw_df)
    files = os.listdir(img_path)

    try:
        for fig in fig_names:
            assert fig in files
        logging.info("SUCCESS: Produced all EDA figures.")
    except AssertionError as error:
        logging.error("ERROR: EDA figures missing.")
        raise error

def test_category_mean_encoder(raw_df, categorical_lst):
    """Test that the category_mean_enocder generates all relevant features."""

    encoded_data = cl.category_mean_encoder(raw_df, categorical_lst)
    try:
        new_vars = [cat+"_Churn" for cat in categorical_lst]
        assert set(new_vars).issubset(encoded_data.columns)
        logging.info("SUCCESS: Mean encodings exist in the new data.")
    except AssertionError as error:
        logging.error("ERROR: Encoded data entries missing.")
        raise error

def test_perform_feature_engineering(train_data_split):
    """Test that the training data splits are all non-empty."""

    try:
        for data in train_data_split:
            assert len(data) > 0, "Empty data split."
        logging.info("SUCCESS: Created training data split.")
    except AssertionError as error:
        logging.error("ERROR: Empty training data split encountered.")
        raise error

def test_train_models(train_data_split, feature_names):
    """Test that the training loop generates all trained models."""
    cl.train_models(*train_data_split, feature_names)

    try:
        joblib.load("models/random_forest.pkl")
        joblib.load("models/logistic_regression.pkl")
        logging.info("SUCCESS: Trained and stored all models.")
    except AssertionError as error:
        logging.error("ERROR: Model training failed.")
        raise error
