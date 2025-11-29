import os
import sys
import mlflow 
from ai_dropout.src.component.data_ingestion import DataIngestion
from ai_dropout.src.component.data_transformation import DataTransformation
from ai_dropout.src.component.model_trainer import ModelTrainer
from ai_dropout.src.logger import logging   

class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # Set MLflow tracking URI (optional - defaults to local ./mlruns)
            # mlflow.set_tracking_uri("http://localhost:5000")  # Uncomment if using MLflow server
            
            # Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            
            # Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            X_train_df, y_train_series, X_test_df, y_test_series, preprocessor_path, preprocessor_obj = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data transformation completed")
            
            # Set MLflow tracking URI to ai_dropout/mlruns
            mlflow.set_tracking_uri("file:./ai_dropout/mlruns")
            
            # Model Training with MLflow tracking
            logging.info("Starting model training with MLflow tracking")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(X_train_df, y_train_series, X_test_df, y_test_series, preprocessor_obj)
            logging.info("Model training completed")
            
            logging.info("Training pipeline completed successfully")
            logging.info("Check MLflow UI to view tracked experiments: mlflow ui")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.initiate_training_pipeline()








