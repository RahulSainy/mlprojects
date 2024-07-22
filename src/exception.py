import sys  # System-specific parameters and functions
from src.logger import logging  # Import the logging module

# learn form https://docs.python.org/3/tutorial/errors.html
def error_message_detail(error, error_detail: sys):
    '''
    This function is used to get the error message with the line number and file name
    The exc_info() method returns a tuple containing three elements: exc_type, exc_value, and exc_traceback. In this case, we only need the exc_traceback as it contains the line number and file name.
    '''
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name
    error_message = f"Error occured in python script name [{0}] line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    '''This class is used to create custom exception'''
    def __init__(self, error_message, error_detail: sys):
        '''
        This function is used to initialize the custom exception
        super() is a built-in function that allows you to call a method from a parent class. In this case, it is being used inside the __init__ method of a class that is inheriting from the Exception class.
        '''
        self.error_detail = error_detail
        self.error_message = error_message_detail(error_message, error_detail = error_detail)
        super().__init__(
            self.error_message
        )  # since we are inheriting form the Exception class we need to call the super class constructor
        # when we raise custom exception it will inherit the custom exception class and the error message will be displayed


    def __str__(self):
        '''
        This function is used to return the error message when we raise the exception
        '''
        #when we raise the exception it will return the error message to print it
        return f"{self.error_message}"

# # to test logger.py 
# if __name__ =="__main__":
   
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Divide by Zero error")
#         raise CustomException(e,sys)
   