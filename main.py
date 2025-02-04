from data_preprocessing import load_and_preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models
from visualization import plot_results


X_train, X_test, y_train, y_test = load_and_preprocess_data("dataset.csv")

models = train_models(X_train, y_train)


results = evaluate_models(models, X_test, y_test)

plot_results(results)
