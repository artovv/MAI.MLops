import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://mlflow:5000")

experiment_name = "Housing"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    mlflow.create_experiment(
        experiment_name,
        artifact_location="file:/mlflow/artifacts/Housing"
    )

mlflow.set_experiment(experiment_name)

# Загрузка данных
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Параметры модели
n_estimators = 100
max_depth = 5

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Логирование в MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"Logged model with MSE: {mse}")