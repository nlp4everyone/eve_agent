from speech_components.utils.types import AdvancedRecognizer
from typing import Literal
from speech_components.config import ASSEMBLYAI_KEY
import assemblyai as aai
# Define key

class AssemblyRecognizer(AdvancedRecognizer):
    def __init__(self,model :Literal["best","nano"] = "best", api_key :str = ASSEMBLYAI_KEY):
        super().__init__()
        # Set API key
        print(api_key)
        aai.settings.api_key = api_key
        # Define model
        self.__speech_model = aai.SpeechModel.best if model == "best" else aai.SpeechModel.nano
        # Define config
        self.__config = aai.TranscriptionConfig(speech_model = self.__speech_model)
        # Define transcriber
        self.__transcriber = aai.Transcriber(config = self.__config)

    def transcribe(self,
                   audio_file :str) -> str:
        """
        Transcribe audio into string
        :param audio_file: Path to the input file
        :return:
        """
        # Get transcription
        transcript = self.__transcriber.transcribe(audio_file)

        # Return status
        if transcript.status != aai.TranscriptStatus.completed:
            raise Exception(f"{transcript.error}")
        return transcript.text

    def segment(self,audio_file :str):
        """
        Returning a list of dictionary information for words appeared in audio:
        :param audio_file: Path to the input file
        :return:
        """
        # Get transcription
        transcript = self.__transcriber.transcribe(audio_file)
        # Return status
        if transcript.status != aai.TranscriptStatus.completed:
            raise Exception(f"{transcript.error}")
        print(transcript.words)
