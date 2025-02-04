import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df.fillna(df.mean(), inplace=True)  
    df = pd.get_dummies(df) 
    
    X = df.drop(columns=["target"])  
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
