import logging
import logging.handlers
import os
import json

class Logger:
    _loggers = {}  # Dictionary to store multiple logger instances

    def __new__(cls, log_file="rag.log", level=logging.INFO, json_format=False):
        """
        Creates a new instance of the Logger class for each unique log file.(made use of Multiton Pattern)

        Args:
            cls (type): The class type to create an instance of.
            log_file (str): The log file name to store logs (default is 'rag.log').
            level (int): The logging level (default is logging.INFO).
            json_format (bool): Whether to use JSON formatting for logs (default is False).
        
        Returns:
            Logger: The logger instance for the specified log file.
        """
        # Use log_file as key to create separate loggers for different files
        if log_file not in cls._loggers:
            instance = super(Logger, cls).__new__(cls)
            instance._initialize(log_file, level, json_format)
            cls._loggers[log_file] = instance
        return cls._loggers[log_file]

    def _initialize(self, log_file, level, json_format):
        """
        Initialize the logger with file and console handlers, and set log formatting.

        Args:
            log_file (str): The log file name to store logs.
            level (int): The logging level.
            json_format (bool): Whether to use JSON formatting for logs.
        
        Returns:
            None
        """
        self.log_file = log_file
        
        # Create a unique logger name based on the log file
        logger_name = f"RAG Application - {os.path.basename(log_file)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs
        
        # Only add handlers if they don't already exist
        if not self.logger.hasHandlers():
            # Formatter
            formatter = logging.Formatter('%(asctime)s - RAG Application - %(levelname)s - %(message)s')
            if json_format:
                formatter = JsonFormatter()

            # File Handler (Rotating logs)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=0
            )
            file_handler.setFormatter(formatter)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Add Handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the configured logger instance.

        Args:
            None
        
        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger

class JsonFormatter(logging.Formatter):
    """Custom JSON Formatter for structured logs."""
    def format(self, record):
        """
        Format the log record into JSON structure.

        Args:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: The log record in JSON format.
        """
        log_record = {
            "timestamp": self.formatTime(record),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record, ensure_ascii=False)