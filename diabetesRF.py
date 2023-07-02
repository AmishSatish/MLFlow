import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    with mlflow.start_run() as run:
        # load the data
        data = load_diabetes()
        X_train,X_test,y_train,y_test = train_test_split(data.data,data.target)

        # params

        params = {
            "n_estimators":100,
            "max_depth":6,
            "max_features":3
        }

        # train the model
        rf = RandomForestRegressor(**params)
        rf.fit(X_train,y_train)

        # use the model to make predictions on the test data
        y_pred = rf.predict(X_test)

        signature = infer_signature(X_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})
        
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="sk-learn-random-forest-reg-model",
        )

        print(f'Run ID {run.info.run_id}')