from pydantic import BaseModel
from typing import Union
import os

class Word(BaseModel):
    text :str
    start :Union[int,float]
    end :Union[int,float]
    probability :float

class BaseRecognizer():
    def __init__(self, model = None):
        """Base class for Recognizer """
        self.__model = model

    def transcribe(self,
                   audio_file :str) -> str:
        """Transcribe audio into string"""
        raise NotImplementedError

    @staticmethod
    def _convert_to_millisecond(time :float) -> int:
        """Convert from second to millisecond"""
        assert isinstance(time,float), "Time must be in float type"
        # Return
        return int(time*1000)

    @staticmethod
    def _convert_to_second(time :int) -> float:
        """Convert from millisecond to second"""
        assert isinstance(time, int), "Time must be in int type"
        # Return
        return float(time/1000)

    @staticmethod
    def _is_link(path :str) -> bool:
        # Strip the path
        path = path.strip()
        # Lowercase
        path = path.lower()
        # Check condition
        return True if path.startswith("https:") or path.startswith("http:") else False

    @staticmethod
    def _is_existed_path(path) -> bool:
        return True if os.path.exists(path) else False

class AdvancedRecognizer(BaseRecognizer):
    def __init__(self):
        super().__init__()

    def segment(self,
                audio_file :str,
                in_milliseconds :bool = True):
        """Function for getting information( segmentation, info) about audio"""
        raise NotImplementedError