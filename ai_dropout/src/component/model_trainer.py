import os 
import sys 

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import joblib

from ..exception import CustomException
from ..logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("ai_dropout/artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_df, y_train_series, X_test_df, y_test_series, preprocessor):
        try:
            logging.info("Using training and test DataFrames (matching test.ipynb format)")
            # Use DataFrames directly (matching test.ipynb)
            X_train = X_train_df
            y_train = y_train_series
            X_test = X_test_df
            y_test = y_test_series
            
            logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Create model pipeline with preprocessor (matching test.ipynb format)
            model = Pipeline([
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
            ])
            
            # Define hyperparameter grid (matching test.ipynb)
            lr_params = [
                {
                    "clf__solver": ["liblinear"],
                    "clf__penalty": ["l1", "l2"],
                    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "clf__class_weight": [None, "balanced", {0:1,1:2}, {0:1,1:5}],
                    "clf__fit_intercept": [True, False],
                },
                {
                    "clf__solver": ["saga"],
                    "clf__penalty": ["l1", "l2", "elasticnet"],
                    "clf__l1_ratio": [0.1, 0.5, 0.9],
                    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "clf__class_weight": [None, "balanced", {0:1,1:2}, {0:1,1:5}, {0:1,1:10}],
                    "clf__fit_intercept": [True, False],
                    "clf__max_iter": [100, 200, 500],
                }
            ]
            
            # Start MLflow run
            mlflow.set_experiment("student_dropout_prediction")
            
            with mlflow.start_run():
                logging.info("Starting GridSearchCV with MLflow tracking")
                
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=lr_params,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1,
                    verbose=2
                )
                
                grid.fit(X_train, y_train)
                
                # Get best model
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
                
                logging.info(f"Best parameters: {best_params}")
                logging.info(f"Best CV AUC score: {best_score:.4f}")
                
                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average='weighted')
                test_recall = recall_score(y_test, y_test_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
                
                # Log hyperparameters to MLflow
                for param, value in best_params.items():
                    mlflow.log_param(param, value)
                
                # Log metrics to MLflow
                mlflow.log_metric("best_cv_auc", best_score)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1_score", test_f1)
                mlflow.log_metric("test_roc_auc", test_roc_auc)
                
                # Log confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                mlflow.log_metric("tn", int(cm[0][0]))
                mlflow.log_metric("fp", int(cm[0][1]))
                mlflow.log_metric("fn", int(cm[1][0]))
                mlflow.log_metric("tp", int(cm[1][1]))
                
                # Log model
                mlflow.sklearn.log_model(best_model, "model")
                
                # Log classification report as artifact
                report = classification_report(y_test, y_test_pred, output_dict=True)
                import json
                with open("classification_report.json", "w") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact("classification_report.json")
                os.remove("classification_report.json")
                
                logging.info(f"Train Accuracy: {train_accuracy:.4f}")
                logging.info(f"Test Accuracy: {test_accuracy:.4f}")
                logging.info(f"Test Precision: {test_precision:.4f}")
                logging.info(f"Test Recall: {test_recall:.4f}")
                logging.info(f"Test F1 Score: {test_f1:.4f}")
                logging.info(f"Test ROC AUC: {test_roc_auc:.4f}")
                
                # Save the best model (complete pipeline)
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
                
                logging.info("Model training completed successfully")
                logging.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
                
                return test_roc_auc

        except Exception as e:
            raise CustomException(e, sys)
