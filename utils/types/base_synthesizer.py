from pathlib import Path
audio_extension = [".aac",".mp3",".flac",".ogg",".wav"]

class BaseSynthesizer():
    def __init__(self, model = None):
        """Base class for Synthesizer """
        self.__model = model
        self._audio_extension = audio_extension

    def _is_audio_path(self,file_path :str) -> bool:
        # Get extension
        extension = Path(file_path).suffix
        # Return True or False
        return True if extension.lower() in audio_extension else False

    def _check_generation_condition(self,
                                    text :str,
                                    file_path) -> bool:
        # Check text
        assert text, "Text cant be empty"

        # Check file path
        if not self._is_audio_path(file_path=file_path):
            raise TypeError(f"Wrong audio format! File path must be end with ({','.join(self._audio_extension)})")
        return True

    def generate(self,
                 text :str,
                 file_path :str,
                 voice = None):
        """Synchronous function to synthesize a voice from define accent"""
        raise NotImplementedError
