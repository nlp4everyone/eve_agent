from enum import Enum

class AudioType(Enum):
    LOCAL_FILE = 0
    LINK = 1
    BINARY_IO = 2
    BYTES = 3

class StatusCode(Enum):
    SUCCESS = 0
    FAILED = 1