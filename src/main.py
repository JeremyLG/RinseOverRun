import sys
sys.path.append("/home/jeremy/Documents/rinseOverRun/src")

from dataPrep.dataReader import easy_read, read  # noqa
from dataPrep.dataBalancer import (write_train_balanced, check_repartition_phases,
                                   rebalance_phases)  # noqa
from dataFeaturing.dataFeaturer import prepare_features_dico, prep_all_features
from dataViz.dataViz import avg_duration_per_phase, avg_duration_final_rinse  # noqa


def read_and_balance():
    train, test = read()
    s_train_balanced = rebalance_phases(train)
    check_repartition_phases(s_train_balanced)
    train.join(s_train_balanced, rsuffix="_")
    train = train.join(s_train_balanced, rsuffix="_")
    write_train_balanced(train)


def agg_balanced():
    pass


train_balanced = easy_read("data/interim/train_balanced.csv")
test_values = easy_read("data/raw/test_values.csv")
train_balanced.groupby("pipeline").pipeline.value_counts()
check_repartition_phases(train_balanced.pipeline)
check_repartition_phases(test_values.pipeline)


train_balanced = easy_read("data/interim/train_balanced.csv")
dd = prepare_features_dico(train_balanced)
train_agg = train_balanced.groupby(["process_id", "phase", "phase_"]).agg(dd)
train_agg.columns = train_agg.columns.droplevel(0)
print(train_agg.columns)
train_agg.head()




def plot():
    train_balanced, test = read()
    avg_duration_per_phase(train_balanced)
    avg_duration_final_rinse(train_balanced)


plot()


if __name__ == '__main__':
    pass
