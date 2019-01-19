import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def avg_duration_final_rinse(train_balanced: pd.DataFrame) -> None:
    temp = train_balanced[train_balanced.phase == 'final_rinse'].groupby("process_id").size()
    temp.plot()
    print(
        ("Moyenne : " + str(int(temp.mean())) + " / Maximum : "
         + str(temp.max()) + " / Minimum : " + str(temp.min()))
        )


def avg_duration_per_phase(train_balanced: pd.DataFrame) -> None:
    phases = train_balanced.phase.unique()
    print(phases)
    temp = train_balanced.groupby(["process_id", "phase"]).size()
    temp.reset_index(inplace=True)
    g = sns.FacetGrid(temp[temp[0] < 1000], col="phase", col_order=phases, size=10)
    g = g.map(plt.hist, 0, bins=100)


def viz_columns(train_balanced):
    cols = ["supply_pump", "supply_pre_rinse", "supply_caustic", "return_caustic", "supply_acid",
            "return_acid", "supply_clean_water", "return_recovery_water", "object_low_level"]
    for col in cols:
        sns.countplot(data=train_balanced, hue="phase", x=col)
        plt.show()
