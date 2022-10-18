"""
This script contains all data ingestion functionalities. That is, for
scraping the contents of a source folder, merging the found csv files
into a single dataframe, and storing that result to disk.

Run from the repository root as:
python dynamic_risk_assessment/ingestion.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/ingestion.py --config_path conf/config.json
"""
import argparse
import pandas as pd
from dynamic_risk_assessment.utils import list_csv_files, load_config

def merge_dataframes(
    src_dir: str,
    out_path_data: str,
    out_path_record: str
    ) -> pd.DataFrame:
    """Merge all csv files in the source folder into a single frame.

    Arguments
    ---------
    src_dir:
        Source directory to query for input csv files.

    out_path_data:
        Destination path to store the final dataframe

    out_path_record:
        Destination path to store the record of ingested files.

    Returns
    -------
    A merged dataframe.

    """

    csv_files = list_csv_files(src_dir)
    data = pd.DataFrame()

    # Compile all csv files into a single dataframe.
    for file in csv_files:
        sub_data = pd.read_csv(file)

        # If the dataframe is not empty, check that the columns match between
        # the to-be-merged frames.
        if not data.empty:
            assert all(data.columns == sub_data.columns), "Columns do not match."

        data = pd.concat([data, sub_data], axis=0, ignore_index=True)

    # Remove duplicate rows.
    data.drop_duplicates(inplace=True, ignore_index=True)

    # Store to file.
    data.to_csv(out_path_data, index=False)

    # Save a record.
    with open(out_path_record, "w", encoding="utf-8") as file:
        for item in csv_files:
            file.write(item + "\n")

    return data

def main(args):
    """Data ingestion main script."""

    cfg = load_config(args.config_path)

    merge_dataframes(
        src_dir=cfg["paths"]["input_data"],
        out_path_data=cfg["paths"]["ingested_data"],
        out_path_record=cfg["paths"]["ingested_record"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform data ingestion.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
