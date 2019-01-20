import pandas as pd
import sys
sys.path.append("/home/jeremy/Documents/rinseOverRun/src")
from dataPrep.dataReader import easy_read, read, add_labels  # noqa
from dataPrep.dataBalancer import (write_train_balanced, check_repartition_phases,
                                   rebalance_phases, create_phases_serie)  # noqa
from dataPrep.dataSpliter import split  # noqa
from dataFeaturing.dataFeaturer import prepare_features_dico, prep_all_features  # noqa
from dataViz.dataViz import avg_duration_per_phase, avg_duration_final_rinse  # noqa
from dataModeling.h2oModeling import train_gbm, train_rf  # noqa
import h2o  # noqa
h2o.init(nthreads=-1, port=42222)


def read_and_balance():
    train, test = read()
    s_train_balanced = rebalance_phases(train)
    check_repartition_phases(s_train_balanced)
    train = train.join(s_train_balanced, rsuffix="_")
    write_train_balanced(train[train.phase != "final_rinse"])


def agg_balanced(read_path, write_path):
    train_balanced = easy_read(read_path)
    if "test" in read_path:
        train_balanced.set_index("process_id", inplace=True)
        s_train_balanced = create_phases_serie(train_balanced)
        check_repartition_phases(s_train_balanced)
        train_balanced = train_balanced.join(s_train_balanced, rsuffix="_")
    dd = prepare_features_dico(train_balanced)
    train_agg = train_balanced.groupby(["process_id", "phase", "phase_"]).agg(dd)
    train_agg.columns = train_agg.columns.droplevel(0)
    train_agg.reset_index(inplace=True)
    train_agg.set_index("process_id", inplace=True)
    train_agg = prep_all_features(train_agg)
    train_agg.to_csv(write_path)


def train_valid():
    train_agg = pd.read_csv("data/interim/train_agg.csv", index_col=0)
    train_final = add_labels(train_agg)
    train, valid = split(train_final, 0.8)
    train.to_csv("data/processed/train.csv")
    valid.to_csv("data/processed/valid.csv")


def modeling():
    train = pd.read_csv("data/processed/train.csv", index_col=0)
    valid = pd.read_csv("data/processed/valid.csv", index_col=0)
    gbm = train_gbm(train, valid)
    gbm


def main():
    read_and_balance()
    agg_balanced("data/interim/train_balanced.csv", "data/interim/train_agg.csv")
    train_valid()
    modeling()


def create_test():
    agg_balanced("data/raw/test_values.csv", "data/processed/test.csv")


train = pd.read_csv("data/processed/train.csv", index_col=0)
valid = pd.read_csv("data/processed/valid.csv", index_col=0)
test = pd.read_csv("data/processed/test.csv", index_col=0)

gbm = train_rf(train, valid)
del train
del valid
liste = ["tank_lsh_clean_water_sum_caustic", "tank_lsh_clean_water_sum_acid",
         "tank_lsh_clean_water_sum_intermediate_rinse", "tank_lsh_clean_water_sum_pre_rinse"]
for t in liste:
    test[t] = test[t].astype("float64")
tf = h2o.H2OFrame(test)

preds = gbm.predict(tf)

df = preds.as_data_frame()
df["process_id"] = test.index.unique().values
df.set_index("process_id", inplace=True)
df.to_csv("data/processed/preds.csv")
tr = pd.read_csv("data/raw/train_labels.csv", index_col=0)
tr[tr < 2e7].hist(bins=25)
df[df < 2e7].hist(bins=25)


def plot():
    train_balanced, test = read()
    avg_duration_per_phase(train_balanced)
    avg_duration_final_rinse(train_balanced)


if __name__ == '__main__':
    print("bonjour")
    # main()
