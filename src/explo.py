import pandas as pd
import sys
sys.path.append("/home/jeremy/Documents/rinseOverRun/src")
from dataModeling.mape import MapeMetric  # noqa
import h2o  # noqa
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator  # noqa
import matplotlib.pyplot as plt  # noqa

h2o.init(nthreads=-1)
mape_func = h2o.upload_custom_metric(MapeMetric, func_name="MAPE", func_file="mape.py")

train = pd.read_csv("data/processed/train.csv", index_col=0)
valid = pd.read_csv("data/processed/valid.csv", index_col=0)

hf, vf = h2o.H2OFrame(train), h2o.H2OFrame(valid)
gbm = H2OGradientBoostingEstimator(model_id="Ayaya_gbm",
                                   seed=1337,
                                   ntrees=5000,
                                   min_split_improvement=1e-4,
                                   learn_rate=1e-3,
                                   stopping_metric="custom",
                                   stopping_rounds=10,
                                   stopping_tolerance=0.001,
                                   custom_metric_func=mape_func)
gbm.train(training_frame=hf, y="final_rinse_total_turbidity_liter", validation_frame=vf)
gbm = H2ORandomForestEstimator(model_id="Ayaya_gbm",
                               seed=1337,
                               ntrees=500,
                               stopping_metric="custom",
                               stopping_rounds=10,
                               stopping_tolerance=0.001,
                               custom_metric_func=mape_func)
gbm.train(training_frame=hf, y="final_rinse_total_turbidity_liter", validation_frame=vf)
gbm
train.dtypes
target = "final_rinse_total_turbidity_liter"
plt.hist(train[train["final_rinse_total_turbidity_liter"] < 0.2*1e8][target])
plt.hist(valid[valid["final_rinse_total_turbidity_liter"] < 0.2*1e8][target])
df = pd.DataFrame(gbm.score_history())
df
df.set_index("number_of_trees", inplace=True)
plt.plot(df[["training_custom", "validation_custom"]])
