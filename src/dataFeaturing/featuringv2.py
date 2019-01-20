liste = ["supply_flow", "supply_pressure", "return_temperature", "return_conductivity",
         "return_turbidity", "return_flow"]
# liste2 = ["tank_level_pre_rinse", "tank_level_caustic", "tank_level_acid",
#           "tank_level_clean_water", "tank_temperature_pre_rinse", "tank_temperature_caustic",
#           "tank_temperature_acid", "tank_concentration_caustic", "tank_concentration_acid"]
functions = [("mean", lambda x: x.mean()), ("std", lambda x: x.std()),
             ("kurt", lambda x: x.kurt()), ("skew", lambda x: x.skew()),
             ("sum", lambda x: x.sum()), ("var", lambda x: x.var())
             ]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]


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
        if key in liste:
            temp_dico = {}
            for quant in quantiles:
                temp_dico[key + "_q" + str(quant)] = percentile(quant)
            for function in functions:
                temp_dico[key + "_" + function[0]] = function[1]
            dico[key] = temp_dico
    return dico
