functions = [("mean", lambda x: x.mean()), ("std", lambda x: x.std())]
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]


def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def prepare_features_dico(df):
    dico = {}
    types = dict(df.dtypes)
    for key, value in types.items():
        if key == "supply_flow":
            dico[key] = {"count": lambda x: x.count()}
        if value == "bool" and key != "target_time_period":
            dico[key] = {key + "_sum": lambda x: 1. * x.sum() / len(x)}
        elif value == "float64" and "tank" not in key:
            temp_dico = {}
            for quant in quantiles:
                temp_dico[key + "_q" + str(quant)] = percentile(quant)
            for function in functions:
                temp_dico[key + "_" + function[0]] = function[1]
            dico[key] = temp_dico
        elif value == "float64":
            pass
    return dico


def prep_all_features(train_set):
    d_set = train_set.copy()
    d_set["phase_"] = d_set["phase_"].str.replace("final_rinse", "")
    dd = d_set.set_index(['phase'], append=True).unstack()
    dd.columns = dd.columns.map('_'.join)
    dd["target"] = dd[["final_rinse_total_turbidity_liter_acid",
                       "final_rinse_total_turbidity_liter_caustic",
                       "final_rinse_total_turbidity_liter_intermediate_rinse",
                       "final_rinse_total_turbidity_liter_pre_rinse"]].max(axis=1)
    dd["phase_"] = dd[["phase__acid", "phase__caustic", "phase__intermediate_rinse",
                       "phase__pre_rinse"]].apply(lambda x: x.any(), raw=True, axis=1)
    dd.drop(["final_rinse_total_turbidity_liter_acid",
             "final_rinse_total_turbidity_liter_caustic",
             "final_rinse_total_turbidity_liter_intermediate_rinse",
             "final_rinse_total_turbidity_liter_pre_rinse",
             "phase__acid", "phase__caustic",
             "phase__intermediate_rinse", "phase__pre_rinse"],
            inplace=True, axis=1)
    return dd


def process(train_set, validation_set):
    train = prep_all_features(train_set)
    valid = prep_all_features(validation_set)
    train.to_csv("processed/train.csv")
    valid.to_csv("processed/valid.csv")
