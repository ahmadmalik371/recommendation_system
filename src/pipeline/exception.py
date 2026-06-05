import sys

def error_message_detail(error, error_detail: sys):
    """
    Extracts the file name and line number where the exception occurred.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message

# Small test snippet to make sure it works if run directly
if __name__ == "__main__":
    from logger import logging  # Relative import for testing locally
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by Zero Error")
        raise CustomException(e, sys)