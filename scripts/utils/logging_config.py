#!/usr/bin/env python3
"""Centralized Logging Configuration for Serie A Betting Pipeline.

Provides consistent logging across all modules with:
- Console output (colored)
- File output (rotating)
- Performance tracking
- Error alerting
- API call logging
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log files
MAIN_LOG = LOG_DIR / "pipeline.log"
ERROR_LOG = LOG_DIR / "errors.log"
API_LOG = LOG_DIR / "api_calls.log"
PERFORMANCE_LOG = LOG_DIR / "performance.log"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """Set up the main pipeline logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to files
        json_format: Whether to use JSON format for file logs

    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Main file handler (rotating, 10MB max, 5 backups)
    if file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            MAIN_LOG,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s"
            ))
        logger.addHandler(file_handler)

        # Error file handler (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d\n"
            "MESSAGE: %(message)s\n"
            "---"
        ))
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class APICallLogger:
    """Specialized logger for API calls with cost tracking."""

    def __init__(self):
        self.logger = logging.getLogger("api_calls")
        self.log_file = API_LOG

        # Set up dedicated file handler
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_call(
        self,
        api_name: str,
        endpoint: str,
        status_code: int,
        response_time_ms: float,
        credits_used: int = 0,
        error: str = None,
    ):
        """Log an API call with details."""
        extra = {
            "api": api_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "credits_used": credits_used,
        }
        if error:
            extra["error"] = error
            self.logger.error(f"API call failed: {api_name} {endpoint}", extra={"extra_data": extra})
        else:
            self.logger.info(f"API call: {api_name} {endpoint} ({status_code})", extra={"extra_data": extra})


class PerformanceLogger:
    """Logger for tracking pipeline performance metrics."""

    def __init__(self):
        self.logger = logging.getLogger("performance")
        self.log_file = PERFORMANCE_LOG

        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.timings = {}

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = datetime.now()

    def end_timer(self, operation: str, success: bool = True, details: dict = None):
        """End timing and log the result."""
        if operation not in self.timings:
            return

        duration = (datetime.now() - self.timings[operation]).total_seconds()
        del self.timings[operation]

        extra = {
            "operation": operation,
            "duration_seconds": duration,
            "success": success,
        }
        if details:
            extra.update(details)

        if success:
            self.logger.info(f"Operation completed: {operation} ({duration:.2f}s)", extra={"extra_data": extra})
        else:
            self.logger.warning(f"Operation failed: {operation} ({duration:.2f}s)", extra={"extra_data": extra})


class PipelineLogger:
    """Main pipeline logger with all features combined."""

    def __init__(self, name: str = "pipeline"):
        setup_logging()
        self.logger = get_logger(name)
        self.api_logger = APICallLogger()
        self.perf_logger = PerformanceLogger()

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = True):
        self.logger.critical(message, exc_info=exc_info)

    def log_api_call(self, api_name: str, endpoint: str, status_code: int,
                     response_time_ms: float, credits_used: int = 0, error: str = None):
        self.api_logger.log_call(api_name, endpoint, status_code, response_time_ms, credits_used, error)

    def start_operation(self, operation: str):
        self.perf_logger.start_timer(operation)
        self.logger.info(f"Starting: {operation}")

    def end_operation(self, operation: str, success: bool = True, details: dict = None):
        self.perf_logger.end_timer(operation, success, details)
        status = "completed" if success else "failed"
        self.logger.info(f"Operation {status}: {operation}")

    def log_prediction(self, match: str, prediction: str, confidence: str, probabilities: dict):
        """Log a prediction with details."""
        self.logger.info(
            f"Prediction: {match} -> {prediction} ({confidence})",
            extra={"extra_data": {"probabilities": probabilities}}
        )

    def log_bet(self, match: str, market: str, selection: str, odds: float, stake: float, value_pct: float):
        """Log a bet recommendation."""
        self.logger.info(
            f"Bet: {match} | {market} | {selection} @ {odds:.2f} | "
            f"Stake: ${stake:.2f} | Value: +{value_pct:.1f}%"
        )


# Global logger instance
pipeline_logger = PipelineLogger()


def log_pipeline_start(bankroll: float, mode: str):
    """Log pipeline start."""
    pipeline_logger.info("=" * 60)
    pipeline_logger.info(f"PIPELINE START | Bankroll: ${bankroll:.2f} | Mode: {mode}")
    pipeline_logger.info("=" * 60)
    pipeline_logger.start_operation("full_pipeline")


def log_pipeline_end(success: bool, total_bets: int = 0, total_value: float = 0):
    """Log pipeline end."""
    pipeline_logger.end_operation("full_pipeline", success, {
        "total_bets": total_bets,
        "total_value_pct": total_value
    })
    pipeline_logger.info("=" * 60)
    status = "SUCCESS" if success else "FAILED"
    pipeline_logger.info(f"PIPELINE {status} | Bets: {total_bets} | Avg Value: +{total_value:.1f}%")
    pipeline_logger.info("=" * 60)


def log_step(step_num: int, total_steps: int, description: str):
    """Log a pipeline step."""
    pipeline_logger.info(f"[{step_num}/{total_steps}] {description}")
    pipeline_logger.start_operation(f"step_{step_num}_{description.lower().replace(' ', '_')}")


def log_step_complete(step_num: int, description: str, success: bool = True, details: str = None):
    """Log step completion."""
    operation = f"step_{step_num}_{description.lower().replace(' ', '_')}"
    pipeline_logger.end_operation(operation, success)
    if details:
        pipeline_logger.info(f"  -> {details}")


if __name__ == "__main__":
    # Test the logging system
    print("Testing logging system...")

    log = PipelineLogger("test")
    log.info("This is an info message")
    log.warning("This is a warning")
    log.error("This is an error")

    log.start_operation("test_operation")
    import time
    time.sleep(0.5)
    log.end_operation("test_operation", success=True, details={"items_processed": 10})

    log.log_bet("Lecce vs Udinese", "h2h", "HOME", 2.98, 3.00, 35.3)

    print(f"\nLogs written to:")
    print(f"  - {MAIN_LOG}")
    print(f"  - {ERROR_LOG}")
    print(f"  - {API_LOG}")
    print(f"  - {PERFORMANCE_LOG}")
