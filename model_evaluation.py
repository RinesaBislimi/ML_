from sklearn.metrics import accuracy_score, classification_report

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred))
    
    return results
