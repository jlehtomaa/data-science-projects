"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import os
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact.
    logger.info("Downloading artifact")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("Removing outliers")
    idx_price = df["price"].between(args.min_price, args.max_price)
    idx_coord = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx_price & idx_coord].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    logger.info("Logging artifact")
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact (raw data)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact (cleaned data)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Short description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Lowest price to consider in the data, else outlier",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Highest price to consider in the data, else outlier",
        required=True
    )

    args = parser.parse_args()
    main(args)
