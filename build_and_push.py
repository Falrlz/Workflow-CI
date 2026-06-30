import os
import sys
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

def main():
    # Setup tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI environment variable is not set!")
        sys.exit(1)
        
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Search for the latest run in the retraining experiment
    experiment_name = "Telco_Customer_Churn_Retraining"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found in DagsHub MLflow!")
        sys.exit(1)
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print(f"Error: No runs found in experiment '{experiment_name}'!")
        sys.exit(1)
        
    latest_run_id = runs[0].info.run_id
    print(f"Found latest retraining run ID: {latest_run_id}")

    # Define Docker image name
    docker_user = os.environ.get("DOCKERHUB_USERNAME")
    if not docker_user:
        print("Error: DOCKERHUB_USERNAME environment variable is not set!")
        sys.exit(1)
        
    image_name = f"{docker_user.lower()}/telco-churn-model:latest"
    print(f"Building Docker image: {image_name}")

    # Build docker image using mlflow models build-docker
    build_cmd = [
        "mlflow", "models", "build-docker",
        "-m", f"runs:/{latest_run_id}/model",
        "-n", image_name,
        "--env-manager", "local"
    ]
    
    # Run the build process
    subprocess.run(build_cmd, check=True)
    print("Docker image build completed successfully.")

    # Push to Docker Hub
    print(f"Pushing Docker image: {image_name}")
    push_cmd = ["docker", "push", image_name]
    subprocess.run(push_cmd, check=True)
    print("Docker image pushed successfully to Docker Hub!")

if __name__ == "__main__":
    main()
