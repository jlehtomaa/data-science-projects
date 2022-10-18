"""
This script evaluates all endpoints in the app.py application and stores
the reponses to file.
"""
import argparse
import requests
from dynamic_risk_assessment.utils import load_config


def main(args):
    """Run and store all API calls."""

    responses = {}
    cfg = load_config(args.config_path)
    url = cfg["app"]["url"]

    # Model index / root.
    responses["index"] = requests.get(url)

    # Prediction endpoint.
    json = {"data_path": cfg["paths"]["ingested_data"]}
    responses["pred"] = requests.post(url + "/prediction", json=json)

    # Scoring endpoint.
    responses["score"] = requests.get(url + "/scoring")

    # Summary endpoint.
    responses["summary"] = requests.get(url + "/summarystats")

    # Diagnostics endpoint.
    responses["diagnostics"] = requests.get(url + "/diagnostics")

    # Write responses to file.
    with open(cfg["paths"]["api_returns"], "w", encoding="utf-8") as file:

        for endpoint, response in responses.items():
            status = response.status_code
            assert status == 200, f"Bad status for {endpoint}: {status}."
            file.write(f"\n\nAPI endpoint: {endpoint}\n")
            file.write(response.text)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Score the model.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
