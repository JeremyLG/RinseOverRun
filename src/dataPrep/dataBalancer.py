def create_phases_serie(df):
    return df.groupby('process_id').phase.unique().apply(lambda x: x.sum())


def check_repartition_phases(series):
    print(series.groupby(series).value_counts() / 1. / len(series))


def change_n_elements(df_len, percent):
    return int(percent * df_len)


def choose_random_scope(s, phase, n_elem):
    sub_s = s[s == "pre_rinsecausticintermediate_rinseacidfinal_rinse"]
    scope = sub_s.sample(n=n_elem, replace=False, random_state=1).index
    s[scope] = phase
    return s


def rebalance_phases(data):
    s_train = create_phases_serie(data)
    length = len(s_train)
    dict_rebalance = {"pre_rinsecausticfinal_rinse": 19.6/100,
                      "pre_rinsefinal_rinse": 9.7/100,
                      "pre_rinsecaustic_intermediate_rinse": 22.6/100}
    for key, value in dict_rebalance.items():
        n_elem = change_n_elements(length, value)
        s_train = choose_random_scope(s_train, key, n_elem)
    return(s_train)


def concat_phases(data, s_data):
    return data.join(s_data, rsuffix="_")


def write_train_balanced(data):
    scope = data.apply(lambda x: x['phase'] in x['phase_'], axis=1)
    train_balanced = data[scope]
    train_balanced.reset_index(inplace=True, drop=False)
    train_balanced.to_csv("data/interim/train_balanced.csv")
