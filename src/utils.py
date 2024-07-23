import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    This function is used to save the object in the pickle format
    :param obj: object to save
    :param file_path: file path to save the object
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test, models):
    """
    This function is used to evaluate the model
    :param X_train: training data
    :param y_train: training target
    :param X_test: testing data
    :param y_test: testing target
    :param models: models to evaluate
    :return: model_report
    """
    try:
        report = {}

        for i in range(len(models)):
            # Get the model name
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)