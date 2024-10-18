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

    def generate(self,
                 text :str,
                 file_path :str,
                 voice = None):
        """Synchronous function to synthesize a voice from define accent"""
        raise NotImplementedError
