"""
This script evaluates all endpoints in the app.py application and stores
the reponses to file.
"""

import requests
from dynamic_risk_assessment.utils import load_config

URL = "http://127.0.0.1:8000"
CONFIG = load_config("./conf/config.json")

def main():
    """Run and store all API calls."""

    responses = {}

    # Model index / root.
    responses["index"] = requests.get(URL)

    # Prediction endpoint.
    json = {"data_path": CONFIG["paths"]["ingested_data"]}
    responses["pred"] = requests.post(URL + "/prediction", json=json)

    # Scoring endpoint.
    responses["score"] = requests.get(URL + "/scoring")

    # Summary endpoint.
    responses["summary"] = requests.get(URL + "/summarystats")

    # Diagnostics endpoint.
    responses["diagnostics"] = requests.get(URL + "/diagnostics")

    # Write responses to file.
    with open(CONFIG["paths"]["api_returns"], "w", encoding="utf-8") as file:

        for endpoint, response in responses.items():
            status = response.status_code
            assert status == 200, f"Bad status for {endpoint}: {status}."
            file.write(f"\n\nAPI endpoint: {endpoint}\n")
            file.write(response.text)



if __name__ == "__main__":
    main()
