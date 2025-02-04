import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.xlabel("Modeli")
    plt.ylabel("Saktësia")
    plt.title("Krahasimi i Modeleve")
    plt.show()
