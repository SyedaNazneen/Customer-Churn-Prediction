import logging
import os
import sys
from datetime import datetime

def setup_logging(module_name):
    # Log file ka naam date ke saath
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(module_name)