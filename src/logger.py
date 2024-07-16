# Lern from https://docs.python.org/3/library/logging.html

import logging
import os
from datetime import datetime

# Create a custom logger
LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y__%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# logs path is the path where the logs will be stored
os.makedirs(logs_path, exist_ok=True)
# exist_ok= true if the directory already exists still create the directory

# log_file_path is the path where the log file will be stored
LOG_FILE_Path = os.path.join(logs_path, LOG_FILE)

#logging.basicConfig is used to configure the root logger it is built-in function
logging.basicConfig(
    filename=LOG_FILE_Path,
    datefmt="%d-%m-%Y %H:%M:%S",
    format="[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s %(message)s",
    # example of format: [2021-07-07 12:00:00] - 10 logger - INFO This is an info message
    level=logging.INFO,
)

# # # Only to test the logger
# if __name__ =="__main__":
#    logging.info("This is an info message")
# # # it creates the output as [16-07-2024 05:11:45] - 28 root - INFO This is an info message
