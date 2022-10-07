import json
import logging
import requests
from hydra import compose, initialize

# Load in the hydra config.
initialize(version_base=None, config_path="./conf")
cfg = compose(config_name="config")

log = logging.getLogger(__name__)


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
logging.info("Sending query to the app.")
logging.info("App response %s", resp.json())
print(resp.json())
print("status code:", resp.status_code)
