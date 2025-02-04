import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
import warnings
import logging

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log", mode="w")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        logger.info(f"Loading parameters from {params_path}")
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        test_size = params["data_ingestion"]["test_size"]
        logger.debug(f"Test size retrieved: {test_size}")
        return test_size
    except FileNotFoundError:
        logger.error("File not found: params.yaml")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Reading data from {url}")
        df = pd.read_csv(url)
        logger.debug(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading data from {url}: {e}")
        return pd.DataFrame()


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Processing data")
        if df.empty:
            logger.warning("DataFrame is empty before processing")
            raise ValueError("DataFrame is empty")

        df.drop(columns=["tweet_id"], inplace=True, errors="ignore")
        final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        final_df.loc[:, "sentiment"] = final_df["sentiment"].replace(
            {"happiness": 1, "sadness": 0}
        )

        logger.debug(f"Processed data shape: {final_df.shape}")
        return final_df
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return pd.DataFrame()


def save_data(
    data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> None:
    try:
        logger.info(f"Saving data to {data_path}")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")


def main() -> None:
    try:
        logger.info("Starting data ingestion pipeline")
        test_size = load_params("params.yaml")
        df = read_data(
            "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        )
        final_df = process_data(df)

        if final_df.empty:
            logger.error("Processed DataFrame is empty, stopping execution.")
            raise ValueError("Processed DataFrame is empty, stopping execution.")

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        logger.debug(
            f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}"
        )

        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
