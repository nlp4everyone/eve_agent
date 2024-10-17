class BaseSynthesizer():
    def __init__(self, model = None):
        """Base class for Synthesizer """
        self.__model = model

    def generate(self,
                 text :str,
                 file_path :str):
        """Function to synthesize a voice from define accent"""
        raise NotImplementedError