from ..utils.types import BaseRecognizer, Word, AudioType, StatusCode, TranscriptionResponse
from typing import Union, Literal, Optional, List, BinaryIO
from ..config import DEEPGRAM_KEY
from deepgram import (DeepgramClient,
                      PrerecordedOptions,
                      FileSource,
                      BufferSource)
import httpx, aiofiles

class DeepGramRecognizer(BaseRecognizer):
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

    @property
    def model_name(self) -> str:
        """Return model name property"""
        return self.__model_name

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

    def transcribe(self,
                   audio: Union[str, BinaryIO,bytes],
                   timeout: Optional[float] = None,
                   connect_time: float = 5,
                   in_milliseconds: bool = True,
                   detect_words :bool = False,
                   **kwargs) -> TranscriptionResponse:
        """
        Synchronous function to return transcription from audio
        :param audio: Audio object ( Accepted types: str (file path), bytes and BinaryIO).
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :param kwargs:
        :return: RecognizerResponse
        """
        # Define timeout
        if timeout != None:
            timeout = httpx.Timeout(timeout=timeout, connect=connect_time)

        # Get AudioType from audio input
        audio_type = self.get_audio_type(audio)

        # Default response
        response = None
        status_code = StatusCode.SUCCESS

        # Switch case
        match audio_type:
            case AudioType.LINK:
                # Audio Link case
                try:
                    # Return response from url
                    response = self.__client.listen.rest.v("1").transcribe_url(source = audio,
                                                                               options = self.__options,
                                                                               timeout = timeout)
                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

            case AudioType.LOCAL_FILE:
                # Local file case
                if not self._is_existed_path(audio):
                    raise FileNotFoundError(f"Local path: {audio} not found")

                try:
                    # Read file as buffer
                    with open(audio, "rb") as file:
                        buffer_data = file.read()
                    # Create payload
                    payload: FileSource = {
                        "buffer": buffer_data,
                    }
                    # Return response from prerecorded file
                    response = self.__client.listen.rest.v("1").transcribe_file(source = payload,
                                                                                options = self.__options,
                                                                                timeout = timeout)
                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

            case AudioType.BYTES:
                # Audio Bytes case
                try:
                    # Create payload
                    payload: BufferSource = {
                        "buffer": bytes(audio),
                    }
                    # Return response from url
                    response = self.__client.listen.rest.v("1").transcribe_file(source = payload,
                                                                                options = self.__options,
                                                                                timeout = timeout)
                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

        # Doesnt response
        if response == None:
            return TranscriptionResponse(status_code = status_code)

        # Get info
        info = response["results"]["channels"][0]["alternatives"][0]
        segments = None
        # When detect segment
        if detect_words:
            segments = self.__contruct_segments(segments = info['words'],
                                                in_milliseconds = in_milliseconds)

        # Return
        return TranscriptionResponse(status_code = status_code,
                                     transcription = str(info["transcript"]),
                                     segments = segments)

    async def atranscribe(self,
                          audio: Union[str, BinaryIO, bytes],
                          timeout: Optional[float] = None,
                          connect_time: float = 5,
                          in_milliseconds: bool = True,
                          detect_words: bool = False,
                          **kwargs) -> TranscriptionResponse:
        """
        Asynchronous function to return transcription from audio
        :param audio: Audio object ( Accepted types: str (file path), bytes and BinaryIO).
        :param timeout: Timeout in second (Default :None)
        :param connect_time: Connect time in second
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :param kwargs:
        :return: RecognizerResponse
        """
        # Define timeout
        if timeout != None:
            timeout = httpx.Timeout(timeout=timeout, connect=connect_time)

        # Get AudioType from audio input
        audio_type = self.get_audio_type(audio)

        # Default response
        response = None
        status_code = StatusCode.SUCCESS

        # Switch case
        match audio_type:
            case AudioType.LINK:
                # Audio Link case
                try:
                    # Return response from url
                    response = await self.__client.listen.asyncrest.v("1").transcribe_url(source = audio,
                                                                                          options = self.__options,
                                                                                          timeout = timeout)
                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

            case AudioType.LOCAL_FILE:
                # Local file case
                if not self._is_existed_path(audio):
                    raise FileNotFoundError(f"Local path: {audio} not found")

                try:
                    # Read buffer
                    async with aiofiles.open(audio, "rb") as audio:
                        buffer_data = await audio.read()

                    # Create payload
                    payload: FileSource = {
                        "buffer": buffer_data,
                    }
                    # Return response from prerecorded file
                    response = await self.__client.listen.asyncrest.v("1").transcribe_file(source = payload,
                                                                                           options = self.__options,
                                                                                           timeout = timeout)
                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

            case AudioType.BYTES:
                # Audio Bytes case
                try:
                    # Create payload
                    payload: BufferSource = {
                        "buffer": bytes(audio),
                    }
                    # Return response from bytes
                    response = await self.__client.listen.asyncrest.v("1").transcribe_file(source = payload,
                                                                                           options = self.__options,
                                                                                           timeout = timeout)

                except Exception as e:
                    # Failed status
                    status_code = StatusCode.FAILED

        # Doesnt response
        if response == None:
            return TranscriptionResponse(status_code = status_code)

        # Get info
        info = response["results"]["channels"][0]["alternatives"][0]

        segments = None
        # When detect segment
        if detect_words:
            segments = self.__contruct_segments(segments = info['words'],
                                                in_milliseconds = in_milliseconds)

        # Return
        return TranscriptionResponse(status_code = status_code,
                                     transcription = str(info["transcript"]),
                                     segments = segments)

