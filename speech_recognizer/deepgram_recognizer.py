from ..utils.types import AdvancedRecognizer,Word
from typing import Union, Literal, Optional, Tuple, List
from ..config import DEEPGRAM_KEY
from deepgram import (DeepgramClient,
                      PrerecordedOptions,
                      FileSource)
import httpx, aiofiles

class DeepGramRecognizer(AdvancedRecognizer):
    def __init__(self,
                 model :Union[Literal["nova-2","nova-2-general"],str] = "nova-2",
                 api_key :str = DEEPGRAM_KEY,
                 **kwargs):
        """
        Initialize Groq recognizer service
        :param model: Basically, there are 2 main model (nova-2 and nova-2-general). For more information,
        visit (https://developers.deepgram.com/docs/models-languages-overview)
        :param api_key: DeepGram key
        """
        super().__init__()
        # Set model name
        self.__model_name = model
        # Set API key
        # self.__client = DeepgramClient(api_key = api_key)
        self.__client = DeepgramClient(api_key)
        # Define option
        self.__options = PrerecordedOptions(
            model = self.__model_name,
            smart_format = True,
        )

    def __contruct_segments(self,
                            segments :List,
                            in_milliseconds: bool = True) -> List[Word]:
        """
        Recontruct segments under standard format
        :param segments: A response from Deepgram Speech to Text
        :param in_milliseconds: Whether return time under second or millisecond type
        :return: List[Word]
        """
        # When detect segments
        output = []
        # Define segments
        for segment in segments:
            # Specify second or millisecond format
            start = self._convert_to_millisecond(segment['start']) if in_milliseconds else segment['start']
            end = self._convert_to_millisecond(segment['end']) if in_milliseconds else segment['end']
            # Get text and confidence
            word = segment['punctuated_word']
            confidence = segment['confidence']
            # Append to output
            output.append(Word(start=start, end=end, text=word, confidence=confidence))
        # Return segments
        return output

    def _detect_segments(self,
                         audio_file: str,
                         timeout: Optional[float] = None,
                         connect_time: float = 5,
                         in_milliseconds: bool = True,
                         detect_words :bool = True,
                         **kwargs) -> Tuple[str|None,List[Word]|None]:
        """
        Synchronous function to detect information ( including word, start, end and confident) each word.
        :param audio_file: Path/URL to the input file
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :param kwargs:
        :return: Tuple[str|None,List[Word]|None]
        """
        # Verify
        if not self._is_existed_path(audio_file) and not self._is_link(audio_file):
            raise FileNotFoundError

        # Define timeout
        if timeout != None:
            timeout = httpx.Timeout(timeout = timeout, connect = connect_time)

        # Detect segment from file
        try:
            # With Local file
            if self._is_existed_path(audio_file):
                # Read buffer
                with open(audio_file, "rb") as file:
                    buffer_data = file.read()
                # Create payload
                payload: FileSource = {
                    "buffer": buffer_data,
                }
                # Return response from prerecorded file
                response = self.__client.listen.rest.v("1").transcribe_file(source=payload,
                                                                            options=self.__options,
                                                                            timeout = timeout)
            else:
                # Return response from url
                response = self.__client.listen.rest.v("1").transcribe_url(source=audio_file,
                                                                           options=self.__options,
                                                                           timeout = timeout)

            # Get info
            info = response["results"]["channels"][0]["alternatives"][0]
            # Define transcript
            transcript = str(info["transcript"])

            # When only detect transcription
            if not detect_words:
                return transcript, None

            # Return both transcript and segments
            segments = self.__contruct_segments(segments = info['words'],
                                                in_milliseconds = in_milliseconds)
            return transcript, segments
        except:
            return None, None

    async def _adetect_segments(self,
                                audio_file: str,
                                timeout: Optional[float] = None,
                                connect_time: float = 5,
                                in_milliseconds: bool = True,
                                detect_words :bool = True,
                                **kwargs) -> Tuple[str|None,List[Word]|None]:
        """
        Asynchronous function to detect information ( including word, start, end and confident) each word.
        :param audio_file: Path/URL to the input file
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :param kwargs:
        :return: Tuple[str|None,List[Word]|None]
        """
        # Verify
        if not self._is_existed_path(audio_file) and not self._is_link(audio_file):
            raise FileNotFoundError

        # Define timeout
        if timeout != None:
            timeout = httpx.Timeout(timeout = timeout, connect = connect_time)
        # Detect segment from file
        try:
            # With Local file
            if self._is_existed_path(audio_file):
                # Read buffer
                async with aiofiles.open(audio_file, "rb") as audio:
                    buffer_data = await audio.read()

                # Create payload
                payload: FileSource = {
                    "buffer": buffer_data,
                }

                # Return response from prerecorded file
                response = await self.__client.listen.asyncrest.v("1").transcribe_file(source = payload,
                                                                                       options = self.__options,
                                                                                       timeout = timeout)
            else:
                # Return response from url
                response = await self.__client.listen.asyncrest.v("1").transcribe_url(source = audio_file,
                                                                                      options = self.__options,
                                                                                      timeout = timeout)
            # Get info
            info = response["results"]["channels"][0]["alternatives"][0]
            # Define transcript
            transcript = str(info["transcript"])

            # When only detect transcription
            if not detect_words:
                return transcript, None

            # Return both transcript and segments
            segments = self.__contruct_segments(segments = info['words'],
                                                in_milliseconds = in_milliseconds)
            return transcript, segments

        except:
            return None, None

    def transcribe(self,
                   audio_file :str,
                   timeout :Optional[float] = None,
                   connect_time :float = 5,
                   **kwargs) -> str:
        """
        Synchronous function to return transcription from audio
        :param audio_file: Path/URL to the input file
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param kwargs:
        :return: str
        """
        # Get transcription
        transcription, _ = self._detect_segments(audio_file = audio_file,
                                                 timeout = timeout,
                                                 connect_time = connect_time,
                                                 detect_words = False)
        return transcription

    async def atranscribe(self,
                          audio_file :str,
                          timeout :Optional[float] = None,
                          connect_time :float = 5,
                          **kwargs) -> str:
        """
        Asynchronous function to return transcription from audio
        :param audio_file: Path/URL to the input file
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param kwargs:
        :return: str
        """
        # Get transcription
        transcription,_ = await self._adetect_segments(audio_file = audio_file,
                                                        timeout = timeout,
                                                        connect_time = connect_time,
                                                        detect_words = False)
        return transcription

    def segment(self,
                audio_file :str,
                in_milliseconds :bool = True,
                timeout: Optional[float] = None,
                connect_time: float = 5,
                **kwargs) -> List[Word]:
        """
        Synchronous function to return segmented words from audio
        :param audio_file: Path/URL to the input file
        :param in_milliseconds: Whether return time under second or millisecond type
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param kwargs:
        :return: List[Word]
        """
        # Get segments
        _, segments = self._detect_segments(audio_file = audio_file,
                                            timeout = timeout,
                                            connect_time = connect_time,
                                            detect_words = True,
                                            in_milliseconds = in_milliseconds)
        return segments

    async def asegment(self,
                       audio_file :str,
                       in_milliseconds :bool = True,
                       timeout: Optional[float] = None,
                       connect_time: float = 5,
                       **kwargs) -> List[Word]:
        """
        Asynchronous function to return segmented words from audio
        :param audio_file: Path/URL to the input file
        :param in_milliseconds: Whether return time under second or millisecond type
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param kwargs:
        :return: List[Word]
        """
        # Get segments
        _, segments = await self._adetect_segments(audio_file = audio_file,
                                                   timeout = timeout,
                                                   connect_time = connect_time,
                                                   detect_words = True,
                                                   in_milliseconds = in_milliseconds)
        return segments

    @property
    def model_name(self) -> str:
        return self.__model_name