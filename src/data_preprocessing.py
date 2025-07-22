import os
import re
import logging
from typing import Any
import numpy as np
import pandas as pd
import nltk
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """
    Lemmatize each word in the text.
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text: str) -> str:
    """
    Remove stop words from the text.
    """
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in str(text).split() if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text: str) -> str:
    """
    Remove all digits from the text.
    """
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """
    Convert all words in the text to lowercase.
    """
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    """
    Remove punctuations and extra whitespace from the text.
    """
    text = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    """
    Remove URLs from the text.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: DataFrame, min_words: int = 3) -> DataFrame:
    """
    Set text to NaN if sentence has fewer than min_words.
    """
    logging.info(f"Removing sentences with fewer than {min_words} words.")
    df['content'] = df['content'].apply(lambda x: x if len(str(x).split()) >= min_words else np.nan)
    return df

def normalize_text(df: DataFrame) -> DataFrame:
    """
    Apply all preprocessing steps to the 'content' column of the DataFrame.
    """
    logging.info("Normalizing text in DataFrame.")
    df['content'] = df['content'].astype(str)
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

def normalized_sentence(sentence: str) -> str:
    """
    Apply all preprocessing steps to a single sentence.
    """
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

def save_processed_data(train_data: DataFrame, test_data: DataFrame, dir_path: str = "data/processed") -> None:
    """
    Save processed train and test DataFrames to CSV files.
    """
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train.csv")
    test_path = os.path.join(dir_path, "test.csv")
    logging.info(f"Saving processed train data to {train_path}")
    train_data.to_csv(train_path, index=False)
    logging.info(f"Saving processed test data to {test_path}")
    test_data.to_csv(test_path, index=False)

def main() -> None:
    """
    Main function to orchestrate data preprocessing.
    """
    logging.info("Loading raw train and test data.")
    train_data = pd.read_csv("data/raw/train.csv")
    test_data = pd.read_csv("data/raw/test.csv")

    logging.info("Normalizing train data.")
    train_data = normalize_text(train_data)
    logging.info("Normalizing test data.")
    test_data = normalize_text(test_data)

    train_data = remove_small_sentences(train_data)
    test_data = remove_small_sentences(test_data)

    save_processed_data(train_data, test_data)

if __name__ == "__main__":
    main()