"""
This scipt tests the running app instance by sending a single dummy
input to run the inference model.
"""

import json
import logging
import requests
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Test the app with a dummy input feature."""

    dummy_input = dict(
            age=50,
            workclass="Private",
            fnlgt=193524,
            education="Doctorate",
            education_num=16,
            marital_status="Married-civ-spouse",
            occupation="Prof-specialty",
            relationship="Husband",
            race="White",
            sex="Male",
            capital_gain=0,
            capital_loss=0,
            hours_per_week=50,
            native_country="United-States"
        )

    resp = requests.post(
        url=cfg["api"]["url"],
        data=json.dumps(dummy_input))

    assert resp.status_code == 200
    log.info("Sending query to the app.")
    log.info("App response code is: %s", resp.status_code)
    log.info("App response is: %s", resp.json())

if __name__ == "__main__":
    main()
