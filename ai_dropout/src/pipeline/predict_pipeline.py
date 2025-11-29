import sys
import pandas as pd 
from datetime import datetime

from ..exception import CustomException
from ..logger import logging
from ..utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            # Load the complete pipeline model (includes preprocessor + classifier)
            model_path = 'ai_dropout/artifacts/model.pkl'
            model = load_object(file_path=model_path)
            
            # The model is a Pipeline, so we can directly predict
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_proba(self, features):
        try:
            # Load the complete pipeline model (includes preprocessor + classifier)
            model_path = 'ai_dropout/artifacts/model.pkl'
            model = load_object(file_path=model_path)
            
            # Get prediction probabilities
            pred_proba = model.predict_proba(features)
            return pred_proba
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(
        self,
        gender: str,
        age: int,
        course_level: str,
        attendance_rate: float,
        homework_completion: float,
        test_score: float,
        sessions_missed_last_30_days: int,
        complaints_count: int,
        last_interaction_days_ago: int,
        tuition_paid_ratio: float,
        late_payment_count: int
    ):
        self.gender = gender
        self.age = age
        self.course_level = course_level
        self.attendance_rate = attendance_rate
        self.homework_completion = homework_completion
        self.test_score = test_score
        self.sessions_missed_last_30_days = sessions_missed_last_30_days
        self.complaints_count = complaints_count
        self.last_interaction_days_ago = last_interaction_days_ago
        self.tuition_paid_ratio = tuition_paid_ratio
        self.late_payment_count = late_payment_count

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "course_level": [self.course_level],
                "attendance_rate": [self.attendance_rate],
                "homework_completion": [self.homework_completion],
                "test_score": [self.test_score],
                "sessions_missed_last_30_days": [self.sessions_missed_last_30_days],
                "complaints_count": [self.complaints_count],
                "last_interaction_days_ago": [self.last_interaction_days_ago],
                "tuition_paid_ratio": [self.tuition_paid_ratio],
                "late_payment_count": [self.late_payment_count]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
