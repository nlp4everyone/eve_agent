from typing import Tuple,List

class BaseRecognizer():
    def __init__(self, model = None):
        """Base class for Recognizer """
        self.__model = model

    def transcribe(self,
                   audio_file :str) -> str:
        """Transcribe audio into string"""
        raise NotImplementedError

class AdvancedRecognizer(BaseRecognizer):
    def __init__(self):
        super().__init__()

    def segment(self,audio_file :str):
        """Function for getting information( segmentation, info) about audio"""
        raise NotImplementedError