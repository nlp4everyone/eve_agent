from ..utils.types import BaseSynthesizer
from typing import Literal
from TTS.api import TTS
import torch, os

class CoquiSynthesizer(BaseSynthesizer):
    def __init__(self,
                 model :str = "tts_models/en/ljspeech/tacotron2-DDC",
                 device :Literal["cpu","cuda","auto"] = "auto",
                 progress_bar :bool = False):
        """
        Initialize Coqui Synthesizer service.
        :param model: To gel all supported model, type: tts-server --list_models.
        Default (tts_models/en/ljspeech/tacotron2-DDC)
        :param device: Enable GPU acceleration (cuda) or only CPU (cpu). Default is auto.
        :param progress_bar: Print progression statement or not. Default False.
        """
        super().__init__()
        # Enable
        if device == "auto":
            # Auto mode
            self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Self-config mode
            self.__device = device

        # Define var
        self.__model_name = model
        self.__model = TTS(model_name = self.__model_name,
                           progress_bar = progress_bar).to(device = self.__device)

    def generate(self,
                 text :str,
                 generated_path :str,
                 lang :str = "en",
                 voice = None,
                 speed :float = 1.0,
                 **kwargs) -> None:
        """
        Synchronously generate synthesis audio
        :param text: Text for generation
        :param generated_path: Local file path of generated audio
        :param lang: Language destination. Default: en
        :param voice: Speaker voice. Default: None.
        :param speed: Describe how fast of speech is. Floating point value between 0.00 (slow) and 2.0 (fast).
        :param kwargs:
        :return: None
        """
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = generated_path)
        # Check speed
        if speed < 0.0 or speed >2.0:
            raise ValueError("Speed value must be in range from 0.0 to 2.0")
        # Check multilingual model
        model_lang = self.__model_name.split("/")[1]

        # Default lang for non-multilingual model
        destination_lang = None
        # When using multilingual model
        if model_lang == "multilingual":
            destination_lang = lang

        # Run TTS
        self.__model.tts_to_file(text = text,
                                 file_path = generated_path,
                                 language = destination_lang,
                                 speaker_wav = voice,
                                 speed = speed)

    def clone(self,
              text: str,
              generated_path: str,
              reference_voice :str,
              lang: str = "en",
              speed: float = 1.0,
              **kwargs):
        """
        Synchronously clone voice from reference voice
        :param text: Text for generation
        :param generated_path: Local file path of generated audio
        :param reference_voice: A file path of reference voice for cloning.
        :param lang: Language destination. Default: en
        :param speed: Describe how fast of speech is. Floating point value between 0.00 (slow) and 2.0 (fast).
        :param kwargs:
        :return: None
        """
        # Check reference path exist
        if not os.path.exists(reference_voice):
            raise FileNotFoundError(f"Reference path: {reference_voice} not found!")
        # Check path overlap
        if generated_path == reference_voice:
            raise ValueError(f"Reference path shouldn't same as generated path!")

        # Cloning
        self.generate(text = text,
                      generated_path = generated_path,
                      lang = lang,
                      voice = reference_voice,
                      speed = speed)

    def voice_converting(self,
                         source_path :str,
                         target_path :str,
                         generated_path :str,
                         **kwargs) -> None:
        """
        Converting the voice in source path to the voice of target path. Result in generated path
        :param source_path: Local source path (Required)
        :param target_path: Local target path (Required)
        :param generated_path: Local generated path (Required)
        :param kwargs:
        :return: None
        """
        # Check model supported!
        model_type = self.__model_name.split("/")[0]
        if model_type != "voice_conversion_models":
            raise ValueError("Only supported voice conversion model for converting!")

        # Check source path exist
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path: {source_path} not found!")

        # Check reference path exist
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Tart path: {target_path} not found!")

        # Converting
        self.__model.voice_conversion_to_file(source_wav = source_path,
                                              target_wav = target_path,
                                              file_path = generated_path)