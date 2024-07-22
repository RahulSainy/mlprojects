import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # for data transformation
from sklearn.impute import SimpleImputer  # for handling missing values
from sklearn.pipeline import Pipeline  # for creating the pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)  # for encoding and scaling

from src.utils import save_object


# data transformation config class will have input as train_data_path, preprocessor_obj_path
@dataclass  # The @dataclass decorator automatically generates the __init__, __repr__, and __eq__ methods for the class.
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts/preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )  # object of DataTransformationConfig this will get the values of class (file paths

    def get_data_transformer_object(self):
        """
        This function is used to for the data transformation and create all the pickel files inorder to perform standard scaling and one hot encoding
        """
        logging.info("Data Transformation Started")
        try:
            # since we already did EDA, we know all the data analysis
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            # create the pipeline for numerical columns on training data
            num_pipline = Pipeline(
                steps=[
                    # impute - means fill the missing values
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            # create the pipeline for categorical columns on training data
            categorical_pipeline = Pipeline(
                steps=[
                    # impute - means fill the missing values
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # since while performing EDA we know there were very less number of categorical columns so we can use one hot encoding otherwise we can use label encoding or targetguided encoding
                    ("onehot", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
        
            # combine the numerical and categorical columns pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipline", num_pipline, numerical_columns),
                    ("cat_pipline", categorical_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(
                e, sys
            )  # raise the custom exception if any error occurs

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ["reading score", "writing score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            # The fit_transform method is used to both fit a preprocessing object to the training data and transform the training data using that object.
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # The transform method is used to apply the learned parameters of the preprocessing object to new data.
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            # The reason for using fit_transform on the training data and transform on the test data is to prevent data leakage. Data leakage occurs when information from the test data is used to inform the model training process, leading to overly optimistic performance estimates.

        

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Save the preprocessed data in the train and test data path")

            # save the data in the train and test data path using utils.py save 
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


