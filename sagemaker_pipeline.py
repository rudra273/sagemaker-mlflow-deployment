import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
import yaml
import os
from dotenv import load_dotenv
from sagemaker import get_execution_role

load_dotenv()
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)



def create_pipeline(
    role,
    preprocessing_script,
    training_script,
    deploy_script,
    input_data_path,
    bucket_name=config['s3_bucket'],
    pipeline_name="diabetes-classification-pipelines",
):
    sagemaker_session = sagemaker.Session()
    
    # Create script processor that will run our Python scripts
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='python:3.8',  # Base Python image
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        base_job_name='diabetes-pipeline',
        sagemaker_session=sagemaker_session
    )

    # Step 1: Preprocessing
    preprocessing_step = ProcessingStep(
        name="PreprocessingStep",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=input_data_path,
                destination="/opt/ml/processing/input/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed_data",
                source="/opt/ml/processing/output/",
                destination=f"s3://{bucket_name}/artifacts/iris_classification/"
            ),
        ],
        code=preprocessing_script
    )
    
    # Step 2: Training
    training_step = ProcessingStep(
        name="TrainingStep",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["preprocessed_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/models/",
                destination=f"s3://{bucket_name}/artifacts/iris_classification/"
            ),
        ],
        code=training_script,
        depends_on=[preprocessing_step]
    )


    # Step 3: deploy
    deploy_step = ProcessingStep(
        name="DeployStep",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["preprocessed_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/models/",
                destination=f"s3://{bucket_name}/artifacts/iris_classification/"
            ),
        ],
        code=deploy_script,
        depends_on=[training_step]
    )

    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[preprocessing_step,training_step,deploy_step],
        sagemaker_session=sagemaker_session
    )
    
    return pipeline

if __name__ == "__main__":
    # Define your parameters
    sagemaker_session = sagemaker.Session()
    role = os.getenv("role_arn")
    preprocessing_script = "load_data.py"
    training_script = "train.py"
    deploy_script = "deploy.py"
    input_data_path = "./data/iris.csv"
    # Create and start the pipeline
    pipeline = create_pipeline(
        role=role,
        preprocessing_script=preprocessing_script,
        training_script=training_script,
        deploy_script=deploy_script,
        input_data_path=input_data_path
    )
    
    # Submit the pipeline definition
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    print(f"Pipeline execution started with ARN: {execution.arn}")

