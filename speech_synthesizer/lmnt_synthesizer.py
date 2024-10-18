from lmnt.api import Speech
from ..utils.types import BaseSynthesizer
from ..config import LMNT_KEY
from typing import List, Literal, Optional
import os

class LmntSynthesizer(BaseSynthesizer):
    def __init__(self,
                 api_key: str = LMNT_KEY):
        """
        Initialize LMNT Synthesizer service.
        :param api_key: LMNT Key
        """
        super().__init__()
        # Define key
        self.__api_key = api_key

    async def list_voices(self,
                          owner :Literal["system","me","all"] = "all") -> List[dict]:
        """
        Return list voices corresponding with api key
        :param owner: Specify which voices to return. Choose from system, me, or all
        :return:
        """
        async with Speech(self.__api_key) as speech:
            return await speech.list_voices(owner = owner)

    async def voice_info(self,
                         voice_id :str):
        """
        Returns details of a specific voice.
        :param voice_id: The id of the voice to update. If you don’t know the id, you can get it from list_voices()
        :return:
        """
        async with Speech(self.__api_key) as speech:
            return await speech.voice_info(voice_id = voice_id)

    async def create_voice(self,
                           name :str,
                           file_name :List[str],
                           enhance: bool = True,
                           type :Literal['instant','professional'] = 'instant',
                           gender :Literal["male","female","nonbinary"]|None = None,
                           description :Optional[str] = None):
        """
        Creates a new voice from a set of audio files. Returns the voice metadata object.
        :param name: The name of the voice.
        :param file_name: A list of filenames to use for the voice.
        :param enhance: For unclean audio with background noise, applies processing to attempt to improve quality.
        Not on by default as it can also degrade quality in some circumstances (Default: True)
        :param type: The type of voice to create. Must be one of instant or professional (Default: instant).
        :param gender: The gender of the voice, e.g. male, female, nonbinary (Default: None).
        :param description: A description of the voice (Default: None).
        :return:
        """

        # When empty
        if len(file_name) == 0:
            raise FileNotFoundError
        # Check file path existed
        path_existed = [os.path.exists(file_path) for file_path in file_name]
        # If path not existed
        if path_existed.count(True) < len(path_existed):
            raise Exception("Some path not existed!")

        # Create voice with params
        async with Speech(self.__api_key) as speech:
            voice = await speech.create_voice(name = name,
                                              enhance = enhance,
                                              filenames = file_name,
                                              type = type,
                                              gender = gender,
                                              description = description)

    async def update_voice(self,
                           voice_id :str,
                           name :str,
                           starred :bool = False,
                           gender :Literal["male","female","nonbinary"]|None = None,
                           description :Optional[str] = None):
        """
        Updates metadata for a specific voice. A voice that is not owned by you can only have its starred field updated.
        Only provided fields will be changed.
        :param voice_id: The id of the voice to update. If you don’t know the id, you can get it from list_voices()
        :param name: The name of the voice.
        :param starred: Whether the voice is starred by you
        :param gender: The gender of the voice, e.g. male, female, nonbinary (Default: None).
        :param description: A description of the voice (Default: None).
        :return:
        """
        # Get all supported voice
        user_voices = await self.list_voices(owner="me")
        # Voice ids
        voice_ids = [voice["id"] for voice in user_voices]
        # Check id
        if not voice_id.lower() in voice_ids:
            print("List supported voices:")
            print(user_voices)
            raise Exception(f"Voice: {voice_id} not existed! Please create_voice first")

        # Update voice with params
        async with Speech(self.__api_key) as speech:
            voice = await speech.update_voice(voice_id = voice_id,
                                              name = name,
                                              starred = starred,
                                              gender = gender,
                                              description = description)

    async def agenerate(self,
                        text :str,
                        file_path :str,
                        voice :str = "ava",
                        format :Literal["acc","mp3","wav"] = "mp3",
                        language :Literal["de","en","es","fr","pt","zh"] = "en",
                        sample_rate :Literal[8000,16000,24000] = 24000,
                        speed :float = 1.0,
                        **kwargs) -> None:
        """
        Asynchronously generate audio from text
        :param text: Text for generation
        :param file_path: Local file path of generated audio
        :param voice: Which voice to render, id is found using the list_voices call
        :param format: aac, mp3, wav. Defaults to mp3 (24kHz 16-bit mono).
        :param language: The desired language of the synthesized speech. Two letter ISO 639-1 code.
        One of de, en, es, fr, pt, zh.
        :param sample_rate: The desired output sample rate in Hz, one of: 8000, 16000, 24000.
        Defaults to 24000 for all formats except mulaw which defaults to 8000.
        :param speed: Floating point value between 0.25 (slow) and 2.0 (fast).
        :param kwargs:
        :return:
        """
        # Check generation condition
        self._check_generation_condition(text = text,
                                         file_path = file_path)

        # Check voice exited
        try:
            await self.voice_info(voice_id = voice)
        except Exception as e:
            raise Exception(e)

        # Validate speed infor
        if speed < 0.25 or speed > 2.0:
            raise ValueError("Speed value must be in range from 0.25 to 1.0")

        # Synthesize audio
        async with Speech(self.__api_key) as speech:
            synthesis = await speech.synthesize(text = text,
                                                voice = voice,
                                                format = format,
                                                language = language,
                                                sample_rate = sample_rate,
                                                speed = speed)
        # Save audio file
        with open(file_path, 'wb') as f:
            f.write(synthesis['audio'])

    async def aclone(self,
                     text: str,
                     file_path: str,
                     voice_id: str,
                     format: Literal["acc", "mp3", "wav"] = "mp3",
                     language: Literal["de", "en", "es", "fr", "pt", "zh"] = "en",
                     sample_rate: Literal[8000, 16000, 24000] = 24000,
                     speed: float = 1.0,
                     **kwargs) -> None:
        """
        Asynchronously clone voice from specified id
        :param text: Text for generation
        :param file_path: Local file path of generated audio
        :param voice_id: Which voice to clone. If not existed, create voice.
        :param format: aac, mp3, wav. Defaults to mp3 (24kHz 16-bit mono).
        :param language: The desired language of the synthesized speech. Two letter ISO 639-1 code.
        One of de, en, es, fr, pt, zh.
        :param sample_rate: The desired output sample rate in Hz, one of: 8000, 16000, 24000.
        Defaults to 24000 for all formats except mulaw which defaults to 8000.
        :param speed: Floating point value between 0.25 (slow) and 2.0 (fast).
        :param kwargs:
        :return: None
        """
        # Check empty
        assert voice_id, "Voice cant be empty"

        # Get all supported voice
        user_voices = await self.list_voices(owner = "me")
        # Voice ids
        voice_ids = [voice["id"] for voice in user_voices]
        # Check id
        if not voice_id.lower() in voice_ids:
            print("List supported voices:")
            print(user_voices)
            raise Exception(f"Voice: {voice_id} not existed! Please create_voice first")

        # Generate
        await self.agenerate(text = text,
                             file_path = file_path,
                             voice = voice_id,
                             format = format,
                             language = language,
                             sample_rate = sample_rate,
                             speed = speed)