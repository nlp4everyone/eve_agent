from gtts import gTTS
from ..utils.types import BaseSynthesizer
import os

class GoogleTTSSynthesizer(BaseSynthesizer):
    def __init__(self):
        """
        Initialize GTTS Synthesizer service.
        """
        super().__init__()

    def generate(self,
                 text :str,
                 file_path :str,
                 lang :str = "en",
                 voice = None,
                 **kwargs):
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = file_path)

        # Initialize object
        tts = gTTS(text = text,
                   lang = lang)
        # Save file
        tts.save(file_path)