from ..utils.types import BaseSynthesizer
from typing import Tuple, List, Dict
from gtts import gTTS
from gtts.lang import tts_langs

class GoogleTTSSynthesizer(BaseSynthesizer):
    def __init__(self):
        """
        Initialize GTTS Synthesizer service.
        """
        super().__init__()

    @property
    def language_supported(self) -> Tuple[List[str],Dict[str,str]]:
        # Get language abbreviation with its long form
        languages = tts_langs()
        # Return list of abbreviation lang
        abbreviations = [key for key in languages.keys()]
        return (abbreviations,languages)

    def generate(self,
                 text :str,
                 generated_path :str,
                 lang :str = "en",
                 **kwargs):
        """
        Synchronously generate synthesis audio
        :param text: Text for generation
        :param generated_path: Local file path of generated audio
        :param lang: Language destination (Check language supported function first).
        :param kwargs:
        :return:
        """
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = generated_path)

        # Initialize object
        tts = gTTS(text = text,
                   lang = lang)
        # Save file
        tts.save(generated_path)