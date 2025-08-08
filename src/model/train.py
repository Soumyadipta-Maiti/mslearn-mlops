# Import libraries
import argparse
import glob
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


def get_csvs_df(path):
    """Read CSV(s) from file or directory path."""
    if not os.path.exists(path):
        raise RuntimeError(f"Path does not exist: {path}")

    if os.path.isfile(path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        raise RuntimeError(f"Provided file is not a CSV: {path}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files in path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """Split dataframe into features and label."""
    X = df.drop("Diabetic", axis=1)
    y = df["Diabetic"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    """Train logistic regression model and log metrics."""
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    return model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    return parser.parse_args()


def main(args):
    """Main training function."""
    mlflow.autolog()
    df = get_csvs_df(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")


if __name__ == "__main__":
    print("\n" + "*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60 + "\n")
