from ..utils.types import BaseSynthesizer
from ..utils.encoding import DeepGramEncoding
from ..config import DEEPGRAM_KEY
from typing import Union
from strenum import StrEnum
from deepgram import (DeepgramClient,
                      SpeakOptions)

class VoiceSetting(StrEnum):
    ASTERIA_FEMALE = "aura-asteria-en"
    LUNA_FEMALE = "aura-luna-en"
    STELLA_FEMALE = "aura-stella-en"
    ATHENA_FEMALE = "aura-athena-en"
    HERA_FEMALE = "aura-hera-en"
    ORION_MALE = "aura-orion-en"
    ARCAS_MALE = "aura-arcas-en"
    PERSEUS_MALE = "aura-perseus-en"
    ANGUS_MALE = "aura-angus-en"
    ORPHEUS_MALE = "aura-orpheus-en"
    HELIOS_MALE = "aura-helios-en"
    ZEUS_MALE = "aura-zeus-en"

class DeepGramSynthesizer(BaseSynthesizer):
    def __init__(self,
                 model :Union[VoiceSetting,str] = VoiceSetting.ASTERIA_FEMALE,
                 api_key :str = DEEPGRAM_KEY,
                 encoding :Union[str,DeepGramEncoding] = DeepGramEncoding.LINEAR16,
                 **kwargs):
        """
        Initialize DeepGram Synthesizer service
        :param model: Supports various voice types. For more information,
        visit (https://developers.deepgram.com/docs/tts-models)
        :param api_key: DeepGram key
        """
        super().__init__()
        # Set model name
        self.__model_name = model
        # Set API key
        self.__client = DeepgramClient(api_key)

        # Define option
        self.__options = SpeakOptions(
            model = self.__model_name,
            encoding = encoding,
            container = "wav"
        )

    def generate(self,
                 text :str,
                 file_path :str) -> None:
        """
        Synchronously generate audio from text
        :param text: Text for generation
        :param file_path: Local file path of generated audio
        :return:
        """
        # Define text
        speak_options = {"text": text}
        # Get response
        response = self.__client.speak.v("1").save(filename = file_path,
                                                   source = speak_options,
                                                   options = self.__options)