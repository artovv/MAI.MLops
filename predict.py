import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_boston
import numpy as np

mlflow.set_tracking_uri("http://mlflow:5000")
experiment_name = "Boston_Housing_Regression"
client = MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"Experiment '{experiment_name}' not found!")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise Exception("No runs found in experiment.")

run = runs[0]
run_id = run.info.run_id
print(f"Using model from run: {run_id}")

model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

data = load_boston()
sample = data.data[0].reshape(1, -1)

prediction = model.predict(sample)
print(f"Predicted price: {prediction[0]:.2f}")