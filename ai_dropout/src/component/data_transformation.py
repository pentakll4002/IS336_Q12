import os 
import sys 
import pandas as pd
import numpy as np

from ..logger import logging
from ..exception import CustomException

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('ai_dropout/artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def drop_unnecessary_columns(self, df):
        """
        Drop columns that are not needed for model training
        """
        try:
            logging.info("Starting column dropping")
            
            cols_to_drop = ["student_id", "full_name"]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            logging.info("Column dropping completed successfully")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_categorical_features(self):
        """
        Return list of categorical features
        """
        return ["gender", "course_level"]
    
    def get_data_transformer_object(self, df):
        """
        This function is responsible for data transformation pipeline
        Matching test.ipynb format: dynamically get numeric features
        """
        try:
            logging.info("Starting data transformer object creation")
            
            categorical_features = self.get_categorical_features()
            
            # Get numeric features dynamically (matching test.ipynb)
            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_features = [col for col in numeric_features if col not in categorical_features]
            
            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numeric features: {list(numeric_features)}")
            
            # Create preprocessing pipeline (matching test.ipynb format)
            preprocessor = ColumnTransformer([
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                ("scale", StandardScaler(), numeric_features)
            ])
            
            logging.info("Data transformer object created successfully")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiate data transformation process
        """
        try:
            logging.info("Starting data transformation process")
            
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Apply transformations
            train_df = self.drop_unnecessary_columns(train_df)
            test_df = self.drop_unnecessary_columns(test_df)
            
            target_column_name = 'dropout'
            
            # Separate features and target (matching test.ipynb: X = df.drop(columns=["dropout"]))
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Get preprocessing object (pass input_feature_train_df to get numeric features dynamically, matching test.ipynb)
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)
            
            # Save preprocessor
            import joblib
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)
            
            logging.info("Data transformation completed successfully")
            
            # Return DataFrames instead of arrays (matching test.ipynb format)
            # ColumnTransformer needs DataFrames to work with column names
            return (
                input_feature_train_df,
                target_feature_train_df,
                input_feature_test_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
            
        except Exception as e:
            raise CustomException(e, sys)