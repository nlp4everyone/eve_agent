from pydantic import BaseModel
from typing import Union, BinaryIO, List
from .base_entities import AudioType, StatusCode
import os

class Word(BaseModel):
    text :str
    start :Union[int,float]
    end :Union[int,float]
    confidence :float

class RecognizerResponse(BaseModel):
    status_code :StatusCode = StatusCode.SUCCESS
    transcription :Union[str,None] = None
    segments :Union[List[Word],None] = None
    description :str = None

class BaseRecognizer():
    def __init__(self, model = None):
        """Base class for Recognizer """
        self.__model = model

    def transcribe(self,
                   audio :str) -> str:
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

    def get_audio_type(self,
                        audio :Union[str, bytes, BinaryIO]):
        """
        Return Audio Type of input path
        :param audio:
        :return:
        """
        if isinstance(audio, bytes):
            # Return Bytes Type
            return AudioType.BYTES
        elif isinstance(audio, BinaryIO):
            # Return Binary Type
            return AudioType.BINARY_IO
        elif isinstance(audio, str):
            # Strip the path
            path = audio.strip()
            # Lowercase
            path = path.lower()
            # Return link type
            if path.startswith("http"):
                return AudioType.LINK
            # Return local file type
            return AudioType.LOCAL_FILE

class AdvancedRecognizer(BaseRecognizer):
    def __init__(self):
        super().__init__()
