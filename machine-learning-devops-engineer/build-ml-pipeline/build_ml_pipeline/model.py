"""
Individual functionalities for data processing and model training.
"""
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score
from build_ml_pipeline.utils import load, hyphen_to_underscore


def process_data(
    data: pd.DataFrame,
    cat_features: Optional[List[str]]=None,
    label: Optional[str]=None,
    training: bool=True,
    encoder: Optional[OneHotEncoder]=None,
    lab_bin: Optional[LabelBinarizer]=None
    ) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Arguments
    ---------
    data:
        Dataframe containing the features and label.
    cat_features:
        List containing the names of the categorical features.
    label:
        Name of the label column in `data`. If None, then an empty array
        will be returned for labels.
    training:
        Indicator if training mode or inference/validation mode.
    encoder:
        Trained sklearn OneHotEncoder. Only used if training=False.
    lab_bin:
        Trained sklearn LabelBinarizer. Only used if training=False.

    Returns
    -------
    features:
        Processed features data.
    labels:
        Processed labels data.
    encoder:
        Trained OneHotEncoder if training=True, otherwise the encoder passed in.
    lab_bin:
        Trained LabelBinarizer if training=True, otherwise the binarizer passed in.
    """

    # Clean up column names.
    data.columns = [hyphen_to_underscore(col) for col in data.columns]

    if cat_features is None:
        cat_features = []

    if label is not None:
        labels = data[label]
        data = data.drop([label], axis=1)
    else:
        labels = np.array([])

    cat_feat = data[cat_features].values
    num_feat = data.drop(*[cat_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lab_bin = LabelBinarizer()
        cat_feat = encoder.fit_transform(cat_feat)
        labels = lab_bin.fit_transform(labels.values).ravel()

    else:
        cat_feat = encoder.transform(cat_feat)
        try:
            labels = lab_bin.transform(labels.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    features = np.concatenate([num_feat, cat_feat], axis=1)
    return features, labels, encoder, lab_bin

def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict
    ) -> RandomForestClassifier:
    """ Trains a machine learning model and returns it.

    Arguments
    ---------
    x_train:
        Training data features.
    y_train:
        Training data labels.
    config:
        Model parameters.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(**config)
    model.fit(x_train, y_train)

    return model

def do_inference(
    model: ClassifierMixin,
    features: np.array
    ) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Arguments
    ---------
    model:
        Trained machine learning model.
    features:
        Data used for prediction.

    Returns
    -------
    Predictions from the model.
    """
    return model.predict(features)

def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray
    ) -> Tuple[float, float, float]:
    """Validates the trained model using precision, recall, and F1.

    Arguments
    ---------
    labels:
        Known labels, binarized.
    preds:
        Predicted labels, binarized.

    Returns
    -------
    A (precision, recall, fbeta) tuple.
    """
    fbeta = fbeta_score(labels, preds, beta=1, zero_division=1)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    return [round(res, 4) for res in [precision, recall, fbeta]]

def compute_slice_metrics(test_data: pd.DataFrame, cfg: DictConfig) -> None:
    """Compute model performance on individual slices of the test data.

    Arguments
    ---------
    test_data
        The raw test split of the full dataset.
    cfg
        The dict-parsed config file. Needs to contain paths for loading the
        model, the label column, and a path for storing the slice metrics.
    """

    model, encoder, lab_bin = load(cfg["paths"]["model"])
    cat_features = cfg["data"]["cat_features"]

    x_test, y_test, _, _ = process_data(
        data=test_data,
        cat_features=cat_features,
        label=cfg["data"]["label"],
        training=False,
        encoder=encoder,
        lab_bin=lab_bin)

    # E.g. for 'education' in cat_features, compute model metrics for each
    # slice of data that has a particular value for education.
    metrics_list = []
    for feature in cat_features: # education, ...
        for category in test_data[feature].unique(): # masters, doctorate, ...

            indx = test_data[feature] == category

            # Make sure that the category is represented in the test data slice.
            if not any(indx):
                continue

            preds = model.predict(x_test[indx])
            precision, recall, fbeta = compute_metrics(y_test[indx], preds)

            res = (
                f"{feature} - {category}: "
                f"precision {precision} recall {recall} fbeta {fbeta}"
            )

            metrics_list.append(res)

    # Dump results to a file.
    outfile = cfg["paths"]["slice_metrics"]
    with open(outfile, "w", encoding="utf-8") as file:
        for item in metrics_list:
            file.write(item + "\n")
