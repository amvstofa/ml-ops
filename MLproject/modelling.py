import sys
import warnings
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil parameter dari argumen command line
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    kernel = sys.argv[2] if len(sys.argv) > 2 else "rbf"
    gamma = sys.argv[3] if len(sys.argv) > 3 else "scale"
    dataset_path = sys.argv[4] if len(sys.argv) > 4 else "train.csv"

    # Load data
    data = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("ObesityCategory", axis=1),
        data["ObesityCategory"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train.head(5)

    with mlflow.start_run():
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("gamma", gamma)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"Model trained with accuracy: {accuracy}")
