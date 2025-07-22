import os
import logging
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_dataset(url: str) -> DataFrame:
    """
    Loads a CSV dataset from a given URL.
    """
    logging.info(f"Loading dataset from {url}")
    return pd.read_csv(url)

def preprocess_dataset(df: DataFrame) -> DataFrame:
    """
    Drops unnecessary columns and filters for 'happiness' and 'sadness' sentiments.
    Converts sentiment to binary (happiness=1, sadness=0).
    """
    logging.info("Dropping 'tweet_id' column")
    df = df.drop(columns=['tweet_id'])
    logging.info("Filtering for 'happiness' and 'sadness' sentiments")
    df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    logging.info("Converting sentiment labels to binary")
    df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
    return df

def split_dataset(df: DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
    """
    Splits the DataFrame into train and test sets.
    """
    logging.info(f"Splitting dataset: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

def save_datasets(train_data: DataFrame, test_data: DataFrame, dir_path: str = "data/raw") -> None:
    """
    Saves train and test DataFrames to CSV files in the specified directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train.csv")
    test_path = os.path.join(dir_path, "test.csv")
    logging.info(f"Saving train data to {train_path}")
    train_data.to_csv(train_path, index=False)
    logging.info(f"Saving test data to {test_path}")
    test_data.to_csv(test_path, index=False)

def main() -> None:
    """
    Main function to orchestrate data ingestion.
    """
    url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    df = load_dataset(url)
    processed_df = preprocess_dataset(df)
    train_data, test_data = split_dataset(processed_df)
    save_datasets(train_data, test_data)

if __name__ == "__main__":
    main()