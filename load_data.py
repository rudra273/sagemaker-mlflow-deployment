# Define session, role, and region so we can
# perform any SageMaker tasks we need
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.serve import SchemaBuilder
from sagemaker.serve import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
import mlflow
from mlflow import MlflowClient
import boto3
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
sagemaker_session = sagemaker.Session()

role = os.getenv("role_env")

region = sagemaker_session.boto_region_name

# S3 prefix for the training dataset to be uploaded to
prefix = "DEMO-scikit-iris"

# Provide the ARN of the Tracking Server that you want to track your training job with
tracking_server_arn = os.getenv("tracking_server_arn")

os.makedirs("./data", exist_ok=True)

s3_client = boto3.client("s3") 
s3_client.download_file(
    f"sagemaker-example-files-prod-{region}", "datasets/tabular/iris/iris.data", "./data/iris.csv"
)

df_iris = pd.read_csv("./data/iris.csv", header=None)
df_iris[4] = df_iris[4].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
iris = df_iris[[4, 0, 1, 2, 3]].to_numpy()
np.savetxt("./data/iris.csv", iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f")

WORK_DIRECTORY = "data"

train_input = sagemaker_session.upload_data(
    WORK_DIRECTORY, key_prefix="{}/{}".format(prefix, WORK_DIRECTORY)
)