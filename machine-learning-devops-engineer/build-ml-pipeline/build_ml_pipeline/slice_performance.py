"""
This module checks the performance of the trained model on data slices,
that is, for each categorical variable (e.g. education), holding the value
fixed for a specific category (e.g. doctorate) to see that the performance
is high throughout.
"""

import logging
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from build_ml_pipeline.model import process_data, compute_metrics
from build_ml_pipeline.utils import load, load_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info("Starting slice performance check.")

    model, encoder, lab_bin = load(cfg["paths"]["model"])
    cat_features = cfg["data"]["cat_features"]

    raw_data = load_data(cfg["paths"]["raw_data"])
    _, test = train_test_split(raw_data, random_state=42)

    x_test, y_test, _, _ = process_data(
        data=test,
        cat_features=cat_features,
        label=cfg["data"]["label"],
        training=False,
        encoder=encoder,
        lab_bin=lab_bin)

    # E.g. for 'education' in cat_features, compute model metrics for each
    # slice of data that has a particular value for education.
    metrics_list = []
    for feature in cat_features: # education, ...
        for category in raw_data[feature].unique(): # masters, doctorate, ...

            indx = test[feature] == category

            # Make sure that the category is represented in the test data slice.
            if not any(indx):
                continue

            preds = model.predict(x_test[indx])
            precision, recall, fbeta = compute_metrics(y_test[indx], preds)

            res = (
                f"{feature} - {category}: "
                f"precision {precision:.3f} recall {recall:.3f} fbeta {fbeta:.3f}"
            )

            metrics_list.append(res)

    # Dump results to a file.
    outfile = cfg["paths"]["metrics"] + "/slice_output.txt"
    with open(outfile, "w", encoding="utf-8") as file:
        for item in metrics_list:
            file.write(item + "\n")

    log.info("Finished slice performance check.")

if __name__ == "__main__":
    main()