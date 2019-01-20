import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("data/raw/train_values.csv", index_col=0, parse_dates=["timestamp"])
labels = pd.read_csv("data/raw/train_labels.csv").set_index("process_id")


def plot(process, column):
    sns.lineplot(x="timestamp", y=column, hue="phase", legend="brief",
                 data=df[df.process_id == process])
    plt.show()


liste = ["supply_flow", "supply_pressure", "return_temperature", "return_conductivity",
         "return_turbidity", "return_flow"]
listee = [26042, 20001, 26367]
liste2 = ["tank_level_pre_rinse", "tank_level_caustic", "tank_level_acid",
          "tank_level_clean_water", "tank_temperature_pre_rinse", "tank_temperature_caustic",
          "tank_temperature_acid", "tank_concentration_caustic", "tank_concentration_acid"]
print(labels[labels.index == 20001])
for t in liste2:
    plot(26042, t)
for t in liste:
    plot(26042, t)

df.sample(frac=0.001)[['process_id', 'phase_']].apply(lambda x: ''.join(str(x)), axis=1).unique()
