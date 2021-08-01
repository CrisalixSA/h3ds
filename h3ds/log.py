from os import getenv
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
from logging import (getLogger, getLoggerClass, setLoggerClass, addLevelName,
                     StreamHandler, Formatter, _nameToLevel)

PRINT = 100
COLORS = {
    PRINT: '\u001b[32m{}\033[0m',
    CRITICAL: '\u001b[35m{}\033[0m',
    ERROR: '\u001b[31m{}\033[0m',
    WARNING: '\u001b[33m{}\033[0m',
    INFO: '\u001b[36m{}\033[0m',
    DEBUG: '\u001b[37m{}\033[0m',
}


class Logger(getLoggerClass()):
    """Custom logger class"""

    def __init__(self, name: str, *args, **kwargs) -> None:
        """Logger class constructor.

        Args:
            name (str): Name of the module.
        """
        super().__init__(name, *args, **kwargs)
        log_limit = int(getenv('LOGLIMIT', 10000)) + len(COLORS[DEBUG]) - 1
        fmt = f'%(levelname)8.8s [%(asctime)23.23s] %(message).{log_limit}s'
        self.propagate = False
        handler = StreamHandler()
        self.loadLevel()
        handler.setLevel(NOTSET)
        formatter = Formatter(fmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        addLevelName(PRINT, 'PRINT')

    def loadLevel(self) -> None:
        """Load log level from environment variable"""
        self.setLevel(_nameToLevel.get(getenv('LOGLEVEL', 'INFO'), INFO))

    def print(self, msg: str, *args, **kwargs) -> None:
        """Log a print-level message"""
        if self.isEnabledFor(PRINT):
            self._log(PRINT, msg, args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical-level message and raise and exception"""
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)
            raise Exception(msg)

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log method wrapper for colouring"""
        super()._log(level, COLORS[level].format(msg), *args, **kwargs)


class ErrorLogger:

    def __init__(self):
        self.errors = []

    def log(self, execution):
        if execution != 0:
            self.errors.append(execution)


def set_logger():
    """Set custom logger class as the default class"""
    setLoggerClass(Logger)


def get_logger():
    """Get current package logger class"""
    set_logger()
    package = __name__.split('.')[0]
    return getLogger(package)


logger = get_logger()
