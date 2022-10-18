"""Evaluate the classifier performace.

Run from the repository root as:
python dynamic_risk_assessment/scoring.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/scoring.py --config_path conf/config.json
"""
import numpy as np
from sklearn import metrics
from sklearn.base import ClassifierMixin


def score_model(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
    ) -> float:
    """Evaluate the classifier performance.

    Arguments
    ---------
    model:
        A trained scikit-learn classifier.

    x_test:
        Input test features.

    y_test:
        Test labels

    output_path:
        Where to store the evaluation metric(s).

    Returns
    -------
    The F1 score of the classifier.
    """

    # Score the model.
    preds = model.predict(x_test)
    score = metrics.f1_score(preds, y_test)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(str(score))

    return score
