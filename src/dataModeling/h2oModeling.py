from .mape import MapeMetric
import h2o
from h2o.estimators import (H2OGradientBoostingEstimator, H2OXGBoostEstimator,
                            H2ORandomForestEstimator)

h2o.init(nthreads=-1, port=42222)
mape_func = h2o.upload_custom_metric(MapeMetric, func_name="MAPE", func_file="mape.py")


def convert_frames(train, valid):
    return h2o.H2OFrame(train), h2o.H2OFrame(valid)


def train_gbm(train, valid):
    hf, vf = convert_frames(train, valid)
    gbm = H2OGradientBoostingEstimator(model_id="Ayaya_gbm",
                                       seed=1337,
                                       ntrees=500,
                                       stopping_metric="custom",
                                       stopping_rounds=10,
                                       stopping_tolerance=0.001,
                                       custom_metric_func=mape_func)
    gbm.train(training_frame=hf, y="final_rinse_total_turbidity_liter", validation_frame=vf)
    return gbm


def train_rf(train, valid):
    hf, vf = convert_frames(train, valid)
    best_rf = H2ORandomForestEstimator(
        model_id="best_rf",
        ntrees=90,
        max_depth=30,
        stopping_rounds=10,
        score_each_iteration=True,
        stopping_metric="custom",
        stopping_tolerance=0.001,
        custom_metric_func=mape_func,
        seed=1337)

    best_rf.train(y="final_rinse_total_turbidity_liter",
                  training_frame=hf,
                  validation_frame=vf)
    return best_rf


def xgb(train, valid):
    hf, vf = convert_frames(train, valid)
    xgb = H2OXGBoostEstimator(model_id="Ayaya_xgb")
    xgb.train(training_frame=hf, y="target", validation_frame=vf)
    return xgb
