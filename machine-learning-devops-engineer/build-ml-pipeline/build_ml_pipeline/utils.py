import os
import pandas as pd
import joblib
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """ Loads a csv file to a dataframe.

    Arguments
    ---------
    path: Path to the csv file.
    """
    return pd.read_csv(path, skipinitialspace=True)

def save(
    model: BaseEstimator,
    encoder: OneHotEncoder,
    lab_bin: LabelBinarizer,
    path: str
    ) -> None:
    """Save trained model and fitted encoders."""
    joblib.dump(model, os.path.join(path, "model.pkl"))
    joblib.dump(encoder, os.path.join(path, "encoder.pkl"))
    joblib.dump(lab_bin, os.path.join(path, "lab_bin.pkl"))

def load(
    path: str
    ) -> Tuple[BaseEstimator, OneHotEncoder, LabelBinarizer]:
    """Load trained model and fitter encoder from disk.

    Arguments
    ---------
    path:
        Path to model artifacts.

    Returns
    -------
    model:
        A trained scikit-learn model instance.
    encoder:
        A fitted one-hot encoder
    lab_bin:
        A fitter label binarizer.
    """

    model = joblib.load(os.path.join(path, "model.pkl"))
    encoder = joblib.load(os.path.join(path, "encoder.pkl"))
    lab_bin = joblib.load(os.path.join(path, "lab_bin.pkl"))

    return model, encoder, lab_bin
