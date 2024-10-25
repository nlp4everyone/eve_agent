from ..utils.types import AdvancedRecognizer, Word, AudioType, StatusCode, TranscriptionResponse
from typing import Literal, List, BinaryIO, Union
from ..config import ASSEMBLYAI_KEY
import assemblyai as aai
import os

class AssemblyRecognizer(AdvancedRecognizer):
    def __init__(self,
                 model :Literal["best","nano"] = "nano",
                 api_key :str = ASSEMBLYAI_KEY,
                 **kwargs):
        """
        Initialize Assembly recognizer service
        :param model: Supported best and nano version. Best version is the most accurate and capable,
        useful for most use case. Nano is less accurate, but lower cost models to product results.
        (Language supported: https://www.assemblyai.com/docs/getting-started/supported-languages)
        :param api_key: AssemblyAI key
        """
        super().__init__()
        # Set API key
        aai.settings.api_key = api_key
        # Define model
        self.__speech_model = aai.SpeechModel.best if model == "best" else aai.SpeechModel.nano
        # Define config
        self.__config = aai.TranscriptionConfig(speech_model = self.__speech_model,
                                                **kwargs)
        # Define transcriber
        self.__client = aai.Transcriber(config = self.__config)

    def __contruct_segments(self,
                            segments :List[aai.types.Word],
                            in_milliseconds: bool = True) -> List[Word]:
        """
        Recontruct segments under standard format
        :param segments: A response from Deepgram Speech to Text
        :param in_milliseconds: Whether return time under second or millisecond type
        :return: List[Word]
        """
        print(type(segments[0]))
        output = []
        for word in segments:
            # Specify second or millisecond format
            start = self._convert_to_second(word.start) if not in_milliseconds else word.start
            end = self._convert_to_second(word.end) if not in_milliseconds else word.end
            # Append to output
            output.append(Word(text=word.text, start=start, end=end, confidence=word.confidence))
        return output

    def transcribe(self,
                   audio :Union[str, BinaryIO,bytes],
                   in_milliseconds: bool = True,
                   detect_words: bool = False,
                   **kwargs) -> TranscriptionResponse:
        """
        Synchronous function to return transcription from audio
        :param audio: Audio object ( Accepted types: str (file path), bytes and BinaryIO)
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :return: str
        """
        # Get AudioType from audio input
        audio_type = self.get_audio_type(audio)

        # When local file not found
        if audio_type == AudioType.LOCAL_FILE and not os.path.exists(audio):
            description = f"File: {audio} not found"
            return TranscriptionResponse(status_code = StatusCode.FAILED,
                                         description = description)

        # Get transcription
        transcription = self.__client.transcribe(audio)

        # Set status
        status_code = StatusCode.SUCCESS if aai.TranscriptStatus.completed else StatusCode.FAILED

        segments = None
        # Add segments
        if detect_words:
            segments = self.__contruct_segments(segments = transcription.words,
                                                in_milliseconds = in_milliseconds)

        # Return
        return TranscriptionResponse(status_code = status_code,
                                     transcription = transcription.text,
                                     confidence = transcription.confidence,
                                     segments = segments)

