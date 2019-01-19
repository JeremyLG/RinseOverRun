import numpy as np


def split(train_agg, ratio):
    process_list = train_agg.process_id.unique()
    np.random.seed(7)
    np.random.shuffle(process_list)
    train_ix = process_list[:int(len(process_list) * ratio)]
    test_ix = process_list[int(len(process_list) * ratio):]
    train_set = train_agg[train_agg.index.isin(train_ix)]
    validation_set = train_agg[train_agg.index.isin(test_ix)]
    return train_set, validation_set
