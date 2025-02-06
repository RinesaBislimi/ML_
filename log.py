
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("results.pkl", "rb") as f:
    results = pickle.load(f)
results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(x="index", y="Score", hue="Metric", data=results_df, palette="viridis")
plt.title("Model Performance Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.show()

print("Results visualization completed.")