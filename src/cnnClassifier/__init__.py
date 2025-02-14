import os  # Importing the os module for interacting with the operating system
import sys  # Importing sys module for system-specific parameters and functions
import logging  # Importing logging module to log messages

# Define the logging format with timestamp, log level, module name, and message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory where log files will be stored
log_dir = "logs"

# Path to the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the logs directory if it does not exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Use the defined format for logs
    handlers=[
        logging.FileHandler(log_filepath),  # Log messages to a file
        logging.StreamHandler(sys.stdout)  # Also print logs to the console
    ]
)

# Create a logger instance with a specific name
logger = logging.getLogger("cnnClassifierLogger")  # Remove the extra closing parenthesis
