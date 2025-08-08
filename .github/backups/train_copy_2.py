# Import libraries
import argparse
import glob
import os
import pandas as pd
import mlflow
# import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# define functions
def main(args):
    # Enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train and evaluate model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # Save model to outputs folder for registration
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

# def get_csvs_df(path):
#     if not os.path.exists(path):
#         raise RuntimeError(f"Cannot use non-existent path provided: {path}")
#     csv_files = glob.glob(f"{path}/*.csv")
#     if not csv_files:
#         raise RuntimeError(f"No CSV files found in provided data path: {path}")
#     return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    
    # If it's a single file, read it
    if os.path.isfile(path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise RuntimeError(f"Provided file is not a CSV: {path}")
    
    # If it's a directory, read all CSV files inside
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    X = df.drop('Diabetic', axis=1)
    y = df['Diabetic']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    return model

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # parse args
    return parser.parse_args()

# run script
if __name__ == "__main__":
    print("\n\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n\n")

# python src/model/train.py --training_data experimentation/data/diabetes-dev.csv --reg_rate 0.01
# mlflow ui