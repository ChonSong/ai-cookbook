"""
Logging Setup for Poker Automation

Provides centralized logging configuration with rotation, formatting,
and multiple output handlers.

Requirements:
    - Python logging (built-in)

Usage:
    from logging_setup import setup_logging
    
    logger = setup_logging(log_dir='./logs', debug=True)
    logger.info("System started")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logging(
    log_dir='./logs',
    debug=False,
    console_output=True,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5,
    log_name='poker_automation'
):
    """
    Configure logging with rotation and formatting.
    
    Args:
        log_dir (str): Directory for log files
        debug (bool): Enable DEBUG level logging
        console_output (bool): Output to console
        max_bytes (int): Maximum size of each log file
        backup_count (int): Number of backup files to keep
        log_name (str): Name for the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation (detailed logs)
    file_handler = RotatingFileHandler(
        log_dir / f'{log_name}.log',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Separate error log file
    error_handler = RotatingFileHandler(
        log_dir / f'{log_name}_errors.log',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Console handler (simpler format)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Log initial message
    logger.info("="*60)
    logger.info(f"Logging initialized - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log directory: {log_dir.absolute()}")
    logger.info(f"Log level: {'DEBUG' if debug else 'INFO'}")
    logger.info("="*60)
    
    return logger


def get_logger(name='poker_automation'):
    """
    Get existing logger instance.
    
    Args:
        name (str): Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger, level):
        """
        Initialize context manager.
        
        Args:
            logger: Logger instance
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(logger):
    """
    Decorator to log function calls and execution time.
    
    Args:
        logger: Logger instance
    
    Usage:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            logger.debug(f"Calling {func.__name__}()")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"{func.__name__}() completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__}() failed after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def create_session_logger(base_log_dir='./logs', session_id=None):
    """
    Create a logger for a specific session.
    
    Args:
        base_log_dir (str): Base directory for logs
        session_id (str): Session identifier (defaults to timestamp)
    
    Returns:
        logging.Logger: Session logger
    """
    if session_id is None:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    session_log_dir = Path(base_log_dir) / 'sessions' / session_id
    session_log_dir.mkdir(parents=True, exist_ok=True)
    
    logger_name = f'session_{session_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Session-specific log file
    handler = logging.FileHandler(
        session_log_dir / 'session.log',
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    
    logger.info(f"Session started: {session_id}")
    logger.info(f"Session log directory: {session_log_dir}")
    
    return logger


def main():
    """Test logging setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test logging setup")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-dir', default='./test_logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_dir=args.log_dir,
        debug=args.debug,
        console_output=True
    )
    
    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Test function decorator
    @log_function_call(logger)
    def test_function():
        import time
        time.sleep(0.1)
        return "Success"
    
    result = test_function()
    logger.info(f"Function result: {result}")
    
    # Test session logger
    session_logger = create_session_logger(base_log_dir=args.log_dir)
    session_logger.info("This is a session-specific log entry")
    
    print(f"\n✓ Logging test complete. Check logs in: {args.log_dir}")


if __name__ == "__main__":
    main()
