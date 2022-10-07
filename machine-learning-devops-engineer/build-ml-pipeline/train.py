import logging
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from build_ml_pipeline.utils import load, save, load_data
from build_ml_pipeline.model import (process_data,
                                     train_model,
                                     do_inference,
                                     compute_metrics)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info("Loading raw data.")
    data = load_data(cfg["paths"]["raw_data"])
    train, test = train_test_split(
        data,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"])

    log.info("Processing train/test data.")
    x_train, y_train, encoder, lab_bin = process_data(
        data=train,
        cat_features=cfg["data"]["cat_features"],
        label=cfg["data"]["label"],
        training=True)

    x_test, y_test, _, _ = process_data(
        data=test, cat_features=cfg["data"]["cat_features"],
        label=cfg["data"]["label"],
        training=False,
        encoder=encoder,
        lab_bin=lab_bin)

    log.info("Training model.")
    model = train_model(x_train, y_train, cfg["random_forest"])

    log.info("Saving model and encoders.")
    save(model, encoder, lab_bin, cfg["paths"]["model"])

    log.info("Calculating performance metrics.")
    model, encoder, lab_bin = load(cfg["paths"]["model"])
    preds = do_inference(model, x_test)
    metrics = compute_metrics(y_test, preds)
    logging.info("precision: %s, recall: %s, fbeta: %s", *metrics)
    logging.info("Finished.")


if __name__ == "__main__":
    main()
