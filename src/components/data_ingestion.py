import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# In datIngestion there should be some inputs like where to save the data, what is the data type, etc
# so for this we will create DataIngestion class


@dataclass  # decorator that will call the DataIngestion class
#The @dataclass decorator automatically generates the __init__, __repr__, and __eq__ methods for the class.
# DataIngestionConfig class will have input as data_path, data_type, test_size:
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )  # default path for train data
    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )  # default path for test data
    raw_data_path: str = os.path.join("data", "data.csv")  # default path for raw data


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = (
            DataIngestionConfig()
        )  # object of DataIngestionConfig this will get the values of class (file paths) in the variable

    def initiate_data_ingestion(self):
        """
        This function is used to intiate and read the data from the raw data path
        """
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv("data/stud.csv")
            logging.info("Dataset read successfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )  # create a directory if it does not exist

            # save the data in the raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # split the data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save the data in the train and test data path
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Train test split completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )  # return the train and test data set for data transformation
        except Exception as e:
            raise CustomException(
                e, sys
            )  # raise the custom exception if any error occurs


# to test the model_trainer.py
if __name__ == '__main__':
    data_ingestion = DataIngestion(DataIngestionConfig())
    # data_ingestion.initiate_data_ingestion()
    train_data,test_data=data_ingestion.initiate_data_ingestion()
#to test data_transformation.py
    # initiate the data transformation
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

#to test model_trainer.py
    # initiate the model trainer
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
