import mlflow
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Obesity Category")

data = pd.read_csv("train.csv")
 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("ObesityCategory", axis=1),
    data["ObesityCategory"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]
with mlflow.start_run():
    # Log parameters
    C=1.0,
    kernel='rbf',
    max_iter=-1
    # Train model
    model = SVC(kernel='linear', C=1.0, max_iter=-1)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)