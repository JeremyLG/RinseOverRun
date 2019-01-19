import pandas as pd
from typing import Tuple


def easy_read(path):
    return sort_data(pd.read_csv(path, index_col=0, parse_dates=["timestamp"]))


def initial_read() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = easy_read("data/raw/train_values.csv")
    test = easy_read("data/raw/test_values.csv")
    labels = pd.read_csv("data/raw/train_labels.csv")
    return train, test, labels


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=['process_id', 'timestamp'])


def add_labels(train: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels_ix = labels.set_index("process_id")
    train_ix = train.set_index("process_id")
    return train_ix.join(labels_ix, on="process_id", rsuffix="_")


def read():
    train, test, labels = initial_read()
    train, test = sort_data(train), sort_data(test)
    return add_labels(train, labels), test
