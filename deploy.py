from mlflow.models.signature import infer_signature
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
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.enums import EndpointType
load_dotenv()
import argparse
import joblib
import os
import pandas as pd

from sklearn import tree

import mlflow
if __name__ == '__main__':
    mlflow.set_tracking_uri(os.getenv("tracking_server_arn"))
    sagemaker_session = sagemaker.Session()
    role = os.get_env("role_env")
    client = MlflowClient()
    registered_model = client.get_registered_model(name="sm-job-experiment-model")
    source_path = registered_model.latest_versions[0].source

    sklearn_input = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, -1)
    sklearn_output = 1
    sklearn_schema_builder = SchemaBuilder(
        sample_input=sklearn_input,
        sample_output=sklearn_output,
    )

    # Create model builder with the schema builder.

    # source_path = 's3://sagemaker-studio-750573229682-fffkyjouino/models/0/315260a40bb242a2a0346d3da5a549ca/artifacts/model'
    model_builder = ModelBuilder(
        mode=Mode.SAGEMAKER_ENDPOINT,
        schema_builder=sklearn_schema_builder,
        role_arn=role,
        model_metadata={"MLFLOW_MODEL_PATH": source_path},
    )

    built_model = model_builder.build()

    data_capture_config = DataCaptureConfig(enable_capture=True, 
                                            sampling_percentage=100, 
                                            destination_s3_uri="s3://mlflow-sagemaker-us-east-1-750573229682/loggs/", 
                                            capture_options=["REQUEST", "RESPONSE"],  # Log both inputs and outputs 
                                            )

    predictor = built_model.deploy(initial_instance_count=1, 
                                instance_type="ml.m5.large",
                                data_capture_config=data_capture_config,

                                
                                )

    predictor.predict(sklearn_input)