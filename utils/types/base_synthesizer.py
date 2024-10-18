class BaseSynthesizer():
    def __init__(self, model = None):
        """Base class for Synthesizer """
        self.__model = model

    def generate(self,
                 text :str,
                 file_path :str,
                 voice = None):
        """Synchronous function to synthesize a voice from define accent"""
        raise NotImplementedError

    async def agenerate(self,
                        text :str,
                        file_path :str,
                        voice = None):
        """Asynchronous Function to synthesize a voice from define accent"""
        raise NotImplementedError