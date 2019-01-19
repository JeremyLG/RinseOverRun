import pandas as pd
pd.set_option('display.max_columns', 40)


train = pd.read_csv("data/raw/train_values.csv", index_col=0, parse_dates=["timestamp"])
test = pd.read_csv("data/raw/test_values.csv", index_col=0, parse_dates=["timestamp"])
labels = pd.read_csv("data/raw/train_labels.csv")
train_with = train.set_index("process_id").join(labels.set_index("process_id"), on="process_id",
                                                rsuffix="_")
target = "final_rinse_total_turbidity_liter"
df = train_with.groupby(["process_id", "phase", "pipeline", target])["return_flow","return_turbidity"].agg(["mean"])
import seaborn as sns
df.columns = df.columns.droplevel(1)
df.reset_index(inplace=True)
list_pips = ["L4", "L8", "L9", "L11"]
sns.scatterplot(x="return_flow", y=target, hue="pipeline", data=df[~df.pipeline.isin(list_pips)])
import matplotlib.pyplot as plt
for pipe in df.pipeline.unique():
    plt.figure()
    print("PIPELINE IS : " + pipe)
    sns.distplot(df[df.pipeline == pipe][target])

sns.scatterplot(x="return_turbidity", y=target, hue="pipeline", data=df)
