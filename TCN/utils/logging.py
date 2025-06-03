import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name. If None, generates timestamp-based name
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure handlers
    handlers = [
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_config(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log configuration dictionary.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_metrics(metrics: dict, prefix: str = "", logger: Optional[logging.Logger] = None) -> None:
    """
    Log metrics dictionary.
    
    Args:
        metrics: Metrics dictionary
        prefix: Prefix for log message
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    message = f"{prefix} " if prefix else ""
    metric_strings = []
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strings.append(f"{key}={value:.4f}")
        else:
            metric_strings.append(f"{key}={value}")
    
    message += ", ".join(metric_strings)
    logger.info(message)


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that works well with tqdm progress bars.
    Prevents log messages from breaking progress bar display.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fallback to regular print if tqdm not available
            print(self.format(record))
        except Exception:
            self.handleError(record)