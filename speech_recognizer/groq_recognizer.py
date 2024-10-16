from ..utils.types import BaseRecognizer
from typing import Literal
from ..config import GROQ_KEY
from groq import Groq
import json

class GroqRecognizer(BaseRecognizer):
    def __init__(self,
                 model :Literal["whisper-large-v3-turbo","distil-whisper-large-v3-en","whisper-large-v3"] = "distil-whisper-large-v3-en",
                 api_key :str = GROQ_KEY,
                 **kwargs):
        """
        Initialize Groq recognizer service
        :param model: Have 3 type of models:
        - If your application is error-sensitive and requires multilingual support, use whisper-large-v3.
        - If your application is less sensitive to errors and requires English only, use distil-whisper-large-v3-en.
        - If your application requires multilingual support and you need the best price for performance, use whisper-large-v3-turbo.
        :param api_key: Groq key
        """
        super().__init__()
        # Set model name
        self.__model_name = model
        # Set API key
        self.__client = Groq(api_key = api_key)

    def transcribe(self,
                   audio_file :str,
                   temperature :float = 0.0,
                   **kwargs) -> str:
        """
        Transcribe audio into string
        :param audio_file: Path to the input file
        :return:
        """
        # Verify
        if not self._is_existed_path(audio_file):
            raise FileNotFoundError

        # Read the transcription
        with open(audio_file, "rb") as file:
            # Create a transcription of the audio file
            transcription = self.__client.audio.transcriptions.create(
                file = (audio_file, file.read()),  # Required audio file
                model = self.__model_name,
                temperature = temperature  # Optional
            )
        return transcription.text

    @property
    def model_name(self) -> str:
        """Return model name property"""
        return self.__model_name