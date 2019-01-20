import pandas as pd
import sys
sys.path.append("/home/jeremy/Documents/rinseOverRun/src")
from dataModeling.mape import MapeMetric  # noqa
import h2o  # noqa
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from sklearn.preprocessing import MinMaxScaler  # noqa

h2o.init(port=42222, nthreads=-1)
mape_func = h2o.upload_custom_metric(MapeMetric, func_name="MAPE", func_file="mape.py")

train = pd.read_csv("data/processed/train.csv", index_col=0)
valid = pd.read_csv("data/processed/valid.csv", index_col=0)

scaler = MinMaxScaler()
target = 'final_rinse_total_turbidity_liter'
train[[target]] = scaler.fit_transform(train[[target]])
valid[[target]] = scaler.transform(valid[[target]])

hf, vf = h2o.H2OFrame(train), h2o.H2OFrame(valid)
gbm = H2OGradientBoostingEstimator(model_id="Ayaya_gbm",
                                   seed=1337,
                                   ntrees=300,
                                   min_split_improvement=1e-4,
                                   learn_rate=1e-3,
                                   stopping_metric="custom",
                                   stopping_rounds=10,
                                   stopping_tolerance=0.001,
                                   custom_metric_func=mape_func)
gbm.train(training_frame=hf, y="final_rinse_total_turbidity_liter", validation_frame=vf)

dd = scaler.inverse_transform(gbm.predict(hf).as_data_frame())
tt = scaler.inverse_transform(train[[target]])
np.sum(np.abs((dd - tt) / np.maximum(tt, 290000))) / len(dd)

rf = H2ORandomForestEstimator(model_id="Ayaya_gbm",
                              seed=1337,
                              ntrees=1,
                              max_depth=30,
                              histogram_type="Random",
                              # stopping_metric="custom",
                              # stopping_rounds=10,
                              # stopping_tolerance=0.00001,
                              custom_metric_func=mape_func)
rf.train(training_frame=hf, y="final_rinse_total_turbidity_liter", validation_frame=vf)
rf

train.dtypes
target = "final_rinse_total_turbidity_liter"
plt.hist(train[train["final_rinse_total_turbidity_liter"] < 0.2*1e8][target])
plt.hist(valid[valid["final_rinse_total_turbidity_liter"] < 0.2*1e8][target])


def plot_errors(model):
    df = pd.DataFrame(model.score_history())
    df.set_index("number_of_trees", inplace=True)
    plt.plot(df[["training_custom", "validation_custom"]])


plot_errors(rf)
from h2o.automl import H2OAutoML  # noqa

aml = H2OAutoML(max_models=20, seed=1, max_runtime_secs=120,
                stopping_metric="RMSLE",
                stopping_rounds=10,
                stopping_tolerance=0.001)
aml.train(y="final_rinse_total_turbidity_liter", training_frame=hf)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb
