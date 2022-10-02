import pytest
import numpy as np
from hydra import initialize, compose
from hydra.utils import get_original_cwd, to_absolute_path
from sklearn.model_selection import train_test_split

from build_ml_pipeline.utils import load_data
from build_ml_pipeline.model import process_data, train_model

CONFIG_PATH = "./../conf/"


@pytest.fixture(name="config", scope="module")
def fixture_config():
    """Fixture for experiment config dictionary."""

    # Read a hydra configuration file as a python dict.
    initialize(config_path=CONFIG_PATH, version_base=None)
    config = compose(config_name="config")

    return config


@pytest.fixture(name="raw_data", scope="module")
def fixture_raw_data(config):
    """Fixture for experiment raw dataset."""
    path = to_absolute_path(config["paths"]["raw_data"])
    return load_data(path)


def test_data_exists(raw_data):
    """Test that the raw dataset has rows and columns."""
    assert raw_data.shape[0] > 0 and raw_data.shape[1] > 0, "Dataset is empty"


def test_label_in_data(raw_data, config):
    """Test that the training label is included in the training set."""
    label = config["data"]["label"]
    assert label in raw_data.columns, "Label not in dataset."


def test_model_output(raw_data, config):
    """Test that trained model produces correct output types."""

    train, test = train_test_split(raw_data)

    x_train, y_train, encoder, lab_bin = process_data(
        data=train,
        cat_features=config["data"]["cat_features"],
        label=config["data"]["label"],
        training=True)

    x_test, _, _, _ = process_data(
        data=test,
        cat_features=config["data"]["cat_features"],
        label=config["data"]["label"],
        training=False,
        encoder=encoder,
        lab_bin=lab_bin)

    model = train_model(x_train, y_train, config["random_forest"])
    preds = model.predict(x_test)

    assert isinstance(preds, np.ndarray), "Incorrect output type."
