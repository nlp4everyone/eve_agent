from ..utils.types import BaseRecognizer
from typing import Literal, Union
from ..config import GROQ_KEY
from httpx import Timeout
from groq import Groq, AsyncGroq
from groq._types import NotGiven, NOT_GIVEN
from groq._constants import DEFAULT_MAX_RETRIES
from groq.resources.audio.transcriptions import AsyncTranscriptions, Transcriptions
import groq, aiofiles

class GroqRecognizer(BaseRecognizer):
    def __init__(self,
                 model :Literal["whisper-large-v3-turbo","distil-whisper-large-v3-en","whisper-large-v3"] = "distil-whisper-large-v3-en",
                 api_key :str = GROQ_KEY,
                 use_async :bool = True,
                 max_retries :int = DEFAULT_MAX_RETRIES,
                 timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
                 **kwargs):
        """
        Initialize Groq recognizer service
        :param model: Have 3 type of models:
        - If your application is error-sensitive and requires multilingual support, use whisper-large-v3.
        - If your application is less sensitive to errors and requires English only, use distil-whisper-large-v3-en.
        - If your application requires multilingual support and you need the best price for performance, use whisper-large-v3-turbo.
        :param api_key: Groq key
        :param use_async: Enable async client. Default is True.
        :param max_retries: Total time retries. Default is 2.
        :param timeout: Set timeout for service. Default is NOT_GIVEN.
        """
        super().__init__()
        # Set model name
        self.__model_name = model
        # Default transcription
        self._transcription = None
        self._async_transcription = None

        if use_async:
            # Async client
            self.__async_client = AsyncGroq(api_key = api_key,
                                            max_retries = max_retries,
                                            timeout = timeout)
            # Async transcription
            self._async_transcription = AsyncTranscriptions(client = self.__async_client)
        else:
            # Sync client
            self.__client = Groq(api_key = api_key,
                                 max_retries = max_retries,
                                 timeout = timeout)
            # Sync transcription
            self._transcription = Transcriptions(client=self.__client)

    def _verify_transcription_condition(self,
                                        audio_file :str,
                                        language :Union[str,NotGiven] = NotGiven,
                                        temperature :float = 0.0) -> None:
        """
        Verify transcription condition
        :param audio_file: Path to the input file
        :param language: Specify the language for transcription. Use ISO 639-1 language codes
        (e.g. "en" for English, "fr" for French, etc.). Specifying a language may improve transcription accuracy and speed.
        Default: Not Given.
        :param temperature: Specify a value between 0 and 1 to control the translation output.
        Default: 0.0 (float).
        :return: None
        """
        # Verify
        if not self._is_existed_path(audio_file):
            raise FileNotFoundError(f"File: {audio_file} is not existed!")

        # Check language supported
        if language != NotGiven and self.__model_name == "distil-whisper-large-v3-en":
            raise ValueError(f"{self.__model_name} only supports English")

        # Check temperature
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError(f"Temperature value only from 0 to 1 !")

    def transcribe(self,
                   audio_file :str,
                   language :Union[str,NotGiven] = NotGiven,
                   prompt : str | NotGiven = NotGiven,
                   temperature :float = 0.0,
                   **kwargs) -> str:
        """
        Synchronous function to return transcription from audio
        :param audio_file: Path to the input file
        :param language: Specify the language for transcription. Use ISO 639-1 language codes
        (e.g. "en" for English, "fr" for French, etc.). Specifying a language may improve transcription accuracy and speed.
        Default: Not Given.
        :param prompt: Provide context or specify how to spell unfamiliar words (limited to 224 tokens).
        Default: Not Given
        :param temperature: Specify a value between 0 and 1 to control the translation output.
        Default: 0.0 (float).
        :return: str
        """
        # Check state of sync client
        assert self._transcription, "Please turn on Synchronous mode by enabling use_async = False"
        # Verify conditions
        self._verify_transcription_condition(audio_file = audio_file,
                                             language = language,
                                             temperature = temperature)

        # Read the transcription
        try:
            with open(audio_file, "rb") as file:
                # Create a transcription of the audio file
                transcription = self._transcription.create(
                    file = (audio_file, file.read()),
                    prompt = prompt,
                    model = self.__model_name,
                    temperature = temperature,
                )
            return transcription.text

        # Catch exceptions
        except groq.BadRequestError as e:
            raise Exception(e.message)
        except groq.AuthenticationError as e:
            raise Exception(e.message)
        except groq.PermissionDeniedError as e:
            raise Exception(e.message)
        except groq.NotFoundError as e:
            raise Exception(e.message)
        except groq.UnprocessableEntityError as e:
            raise Exception(e.message)
        except groq.RateLimitError as e:
            raise Exception("A 429 status code was received; we should back off a bit.")
        except groq.InternalServerError as e:
            raise Exception(e.message)
        except groq.APIConnectionError as e:
            raise Exception("The server could not be reached")

    async def atranscribe(self,
                          audio_file: str,
                          language: Union[str, NotGiven] = NotGiven,
                          prompt: str | NotGiven = NotGiven,
                          temperature: float = 0.0,
                          **kwargs) -> str:
        """
        Asynchronous function to return transcription from audio
        :param audio_file: Path to the input file
        :param language: Specify the language for transcription. Use ISO 639-1 language codes
        (e.g. "en" for English, "fr" for French, etc.). Specifying a language may improve transcription accuracy and speed.
        Default: Not Given.
        :param prompt: Provide context or specify how to spell unfamiliar words (limited to 224 tokens).
        Default: Not Given
        :param temperature: Specify a value between 0 and 1 to control the translation output.
        Default: 0.0 (float).
        :return: str
        """
        # Check state of sync client
        assert self._async_transcription, "Please turn on Asynchronous mode by enabling use_async = True"
        # Verify conditions
        self._verify_transcription_condition(audio_file = audio_file,
                                             language = language,
                                             temperature = temperature)

        # Read the transcription
        try:
            # Read buffer
            async with aiofiles.open(audio_file, "rb") as audio:
                buffer_data = await audio.read()

            # Get transcription
            transcription = await self._async_transcription.create(
                file = (audio_file, buffer_data),
                prompt = prompt,
                model = self.__model_name,
                temperature = temperature
            )
            return transcription.text
        # Catch exceptions
        except groq.BadRequestError as e:
            raise Exception(e.message)
        except groq.AuthenticationError as e:
            raise Exception(e.message)
        except groq.PermissionDeniedError as e:
            raise Exception(e.message)
        except groq.NotFoundError as e:
            raise Exception(e.message)
        except groq.UnprocessableEntityError as e:
            raise Exception(e.message)
        except groq.RateLimitError as e:
            raise Exception("A 429 status code was received; we should back off a bit.")
        except groq.InternalServerError as e:
            raise Exception(e.message)
        except groq.APIConnectionError as e:
            raise Exception("The server could not be reached")

    @property
    def model_name(self) -> str:
        """Return model name property"""
        return self.__model_name