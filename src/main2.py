import pandas as pd
import sys
sys.path.append("/home/jeremy/Documents/rinseOverRun/src")
from dataPrep.dataReader import easy_read, add_labels  # noqa
from dataPrep.dataBalancer import (check_repartition_phases, create_phases_serie)  # noqa
from dataPrep.dataSpliter import split  # noqa
from dataFeaturing.featuringv2 import prepare_features_dico  # noqa
from dataModeling.h2oModeling import train_gbm  # noqa
import h2o  # noqa
h2o.init(nthreads=-1, port=42222)
import numpy as np  # noqa


def agg_balanced(read_path, write_path):
    train_balanced = easy_read(read_path)
    train_balanced["nicecol"] = np.maximum(train_balanced.return_flow,
                                           0) * train_balanced.return_turbidity
    if "test" in read_path:
        train_balanced.set_index("process_id", inplace=True)
        s_train_balanced = create_phases_serie(train_balanced)
        check_repartition_phases(s_train_balanced)
        train_balanced = train_balanced.join(s_train_balanced, rsuffix="_")
    dd = prepare_features_dico(train_balanced)
    train_agg = train_balanced.groupby(["process_id", "phase_"]).agg(dd)
    train_agg.columns = train_agg.columns.droplevel(0)
    train_agg.reset_index(inplace=True)
    train_agg.set_index("process_id", inplace=True)
    train_agg.to_csv(write_path)


def train_valid(read_path):
    train_agg = pd.read_csv(read_path, index_col=0)
    train_final = add_labels(train_agg)
    train, valid = split(train_final, 0.8)
    train.to_csv("data/processed/train.csv")
    valid.to_csv("data/processed/valid.csv")


train_agg = agg_balanced("data/interim/train_balanced.csv", "data/interim/train_agg2.csv")
train_valid("data/interim/train_agg2.csv")


def modeling():
    train = pd.read_csv("data/processed/train.csv", index_col=0)
    valid = pd.read_csv("data/processed/valid.csv", index_col=0)
    gbm = train_gbm(train, valid)
    return gbm


def create_test():
    agg_balanced("data/raw/test_values.csv", "data/processed/test2.csv")


create_test()


def score_test(model):
    test = pd.read_csv("data/processed/test2.csv", index_col=0)
    tf = h2o.H2OFrame(test)
    preds = model.predict(tf)
    df = preds.as_data_frame()
    df["process_id"] = test.index.unique().values
    df.set_index("process_id", inplace=True)
    df.to_csv("data/processed/preds2.csv")
