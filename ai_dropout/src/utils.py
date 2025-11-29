import os
import sys
import joblib

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {str(e)}")

