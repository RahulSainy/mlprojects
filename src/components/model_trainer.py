import os
import sys

print("sys_exeutable ", sys.executable)
print("sys_path ", sys.path)

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        This function initiates the model training process by calling the necessary functions.
        X_train = train_array[:,:-1] selects all rows (:) and all columns except the last one (:-1) from the train_array. This means it is selecting all the features or input data for training.
        y_train = train_array[:,-1] selects all rows (:) and the last column (-1) from the train_array. This means it is selecting the target or output data for training.
        X_test = test_array[:,:-1] selects all rows (:) and all columns except the last one (:-1) from the test_array. This means it is selecting all the features or input data for testing.
        y_test = test_array[:,-1] selects all rows (:) and the last column (-1) from the test_array. This means it is selecting the target or output data for testing.
        """
        try:
            logging.info("Spliting Training And Testing Input Data")
            # Splitting the data into training and testing data using a tuple unpacking method
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            # Evaluating the model function in utils.py
            # This function will run the model and return the model report
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )
            # Sorting the model report in descending order from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(
                f"Best Model found on both trainig and testing dataset: {best_model_name}"
            )

            # Saving the best model in the trained_model_file_path as model.pikel
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
