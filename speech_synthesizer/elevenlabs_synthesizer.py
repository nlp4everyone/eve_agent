from elevenlabs.client import ElevenLabs, AsyncElevenLabs, DEFAULT_VOICE
from ..utils.types import BaseSynthesizer
from ..config import ELEVEN_API_KEY
from typing import Optional, List, Literal
from elevenlabs import save
from elevenlabs.types import Voice, VoiceSettings
import httpx

class ElevenLabsSynthesizer(BaseSynthesizer):
    def __init__(self,
                 model :Literal["eleven_multilingual_v2","eleven_monolingual_v1"] = "eleven_monolingual_v1",
                 api_key :str = ELEVEN_API_KEY,
                 use_async :bool = False,
                 timeout : Optional[float] = 60):
        """
        Initialize ElevenLabs Synthesizer service
        :param model: Currently supported 2 model: eleven_multilingual_v2 and eleven_monolingual_v1.
        eleven_multilingual_v2 supported 29 different languages. eleven_monolingual_v1 only supported English speech.
        For more information (https://github.com/elevenlabs/elevenlabs-python)
        :param api_key: Eleven Labs API Key (Required)
        :param timeout: Timeout in float type
        """
        super().__init__()
        # Define ElevenLab client
        self.__client = ElevenLabs(api_key = api_key,
                                   timeout = timeout)
        # Default no async client
        self.__async_client = None
        # Async ElevenLab
        if use_async: self.__async_client = AsyncElevenLabs(api_key = api_key,
                                                            timeout = timeout,
                                                            httpx_client = httpx.AsyncClient())
        # Define model
        self.__model_name = model

    @property
    def supported_voice(self) -> List[Voice]:
        """Return list of supported voice"""
        return self.__client.voices.get_all().voices

    def generate(self,
                 text :str,
                 file_path :str,
                 voice :str | Voice = DEFAULT_VOICE,
                 voice_settings: VoiceSettings | None = DEFAULT_VOICE.settings,
                 stream: bool = False,
                 **kwargs) -> None:
        """
        Synchronously generate audio from text
        :param text: Text for generation
        :param file_path: Local file path of generated audio
        :param voice: Selected voice for generation ( Default: DEFAULT_VOICE)
        :param voice_settings: Selected voice for generation ( Default: DEFAULT_VOICE.settings)
        :param stream: Enable stream mode or not
        :param kwargs:
        :return:
        """
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = file_path)

        # Generate audio
        audio = self.__client.generate(text = text,
                                       voice = voice,
                                       voice_settings = voice_settings,
                                       stream = stream)
        # Save audio
        save(audio = audio, filename = file_path)

    async def agenerate(self,
                        text :str,
                        file_path :str,
                        voice :str | Voice = DEFAULT_VOICE,
                        voice_settings: VoiceSettings | None = DEFAULT_VOICE.settings,
                        stream: bool = False,
                        **kwargs) -> None:
        """
        Asynchronously generate audio from text
        :param text: Text for generation
        :param file_path: Local file path of generated audio
        :param voice: Selected voice for generation ( Default: DEFAULT_VOICE)
        :param voice_settings: Selected voice for generation ( Default: DEFAULT_VOICE.settings)
        :param stream: Enable stream mode or not
        :param kwargs:
        :return:
        """
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = file_path)

        # When async doesnt turn on
        assert self.__async_client, "Please enable use_async"
        # Generate audio
        audio = await self.__async_client.generate(text=text,
                                                   voice=voice,
                                                   voice_settings=voice_settings,
                                                   stream=stream)
        # Add bytes
        out = b''
        async for value in audio:
            out += value
        # Save audio
        save(audio = out, filename=file_path)
