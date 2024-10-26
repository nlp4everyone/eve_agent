from ..utils.types import AdvancedRecognizer, Word, TranscriptionResponse, BaseRecognizer, StatusCode
from typing import Literal, List, Union, Optional, BinaryIO
from faster_whisper.transcribe import TranscriptionInfo
from faster_whisper import WhisperModel
from strenum import StrEnum
import os

class QuantizeType(StrEnum):
    INT8 = "int8",
    INT8_FLOAT32 = "int8_float32",
    INT8_FLOAT16 = "int8_float16",
    INT8_BFLOAT16 = "int8_bfloat16",
    INT16 = "int16",
    FLOAT16 = "float16",
    BFLOAT16 = "bfloat16",
    FLOAT32 = "float32",

class FasterWhisperRecognizer(BaseRecognizer):
    def __init__(self,
                 model_name :str = "small.en",
                 device :Literal["cuda","cpu","auto"] = "auto",
                 device_index : Union[int, List[int]] = 0,
                 compute_type :Union[QuantizeType,str] = "default",
                 cpu_threads: int = 4,
                 num_workers :int = 1,
                 download_root :Optional[str] = None,
                 use_batch :bool = False,
                 **kwargs):
        """
        This class handles interaction with phoneme in word element, powered by FasterWhisper model:
        model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
            large-v2, large-v3, large, distil-large-v2 or distil-large-v3), a path to a
            converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub.
            When a size or a model ID is configured, the converted model is downloaded
            from the Hugging Face Hub.
        :param device: Specify whether GPU will be used or not. Default is auto.
        :param device_index:  Device ID to use. The model can also be loaded on multiple GPUs by passing a list of IDs (e. g. [0, 1, 2, 3])
        :param compute_type: Type to use for computation. See https://opennmt.net/ CTranslate2/ quantization. html.
        :param cpu_threads: Number of threads to use when running on CPU (4 by default).
        :param num_workers: When transcribe() is called from multiple Python threads, having multiple workers enables true parallelism when running the model
        (concurrent calls to self.model.generate() will run in parallel). This can improve the global throughput at the cost of increased memory usage.
        :param download_root: Directory where the models should be saved. If not set, the models are saved in the standard Hugging Face cache directory.
        """
        super().__init__()
        # Define TTS Model with input parameter
        self.__model = WhisperModel(model_size_or_path = model_name,
                                    device = device,
                                    device_index = device_index,
                                    compute_type = compute_type,
                                    cpu_threads = cpu_threads,
                                    num_workers = num_workers,
                                    download_root = download_root,
                                    **kwargs)
        # Used batch
        # To use this feature, you must install FasterWhisper from scratch (pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz")
        if use_batch:
            from faster_whisper import BatchedInferencePipeline
            self.__model = BatchedInferencePipeline(model=self.__model)

    def __contruct_segments(self,
                            segments: List,
                            in_milliseconds: bool = True) -> List[Word]:
        """
        Recontruct segments under standard format
        :param segments: A response from Deepgram Speech to Text
        :param in_milliseconds: Whether return time under second or millisecond type
        :return: List[Word]
        """
        output = []
        for segment in segments:
            for word in segment.words:
                # Specify second or millisecond format
                start = self._convert_to_millisecond(word.start) if in_milliseconds else word.start
                end = self._convert_to_millisecond(word.end) if in_milliseconds else word.end
                # Append to output
                output.append(Word(text=word.word, start=start, end=end, confidence=word.probability))
        return output

    def get_transcription_info(self,
                               audio :Union[str, bytes, BinaryIO]) -> TranscriptionInfo:
        """
        Return information about transcription
        :param audio: Path to the input file (or a file-like object), or the audio waveform.
        :return: TranscriptionInfo
        """
        # File not found
        if isinstance(audio, str) and not os.path.exists(audio):
            raise FileNotFoundError(f"File {audio} not found")

        # Return segmentation and info
        _, information = self.__model.transcribe(audio = audio,
                                                 word_timestamps = False)
        return information

    def transcribe(self,
                   audio :Union[str, bytes, BinaryIO],
                   in_milliseconds: bool = True,
                   detect_words: bool = False,
                   **kwargs) -> TranscriptionResponse:
        """
        Synchronous function to return transcription from audio
        :param audio: Path to the input file (or a file-like object), or the audio waveform.
        :param in_milliseconds: Whether return time under second or millisecond type
        :param detect_words: Enable return list of segmented words.
        :return: TranscriptionResponse
        """
        # Check file path
        if isinstance(audio, str) and not os.path.exists(audio):
            description = f"File {audio} not found"
            # Return value
            return TranscriptionResponse(status_code = StatusCode.FAILED,
                                         description = description)

        if not detect_words:
            # Get segments
            segments, _ = self.__model.transcribe(audio = audio,
                                                  word_timestamps = False,
                                                  without_timestamps = True,
                                                  **kwargs)
            # Define transcription
            transcription = "".join([word.text for word in segments])
            # Return
            return TranscriptionResponse(status_code = StatusCode.SUCCESS,
                                         text = transcription)

        # Return only transcription
        segments, _ = self.__model.transcribe(audio = audio,
                                              word_timestamps = True,
                                              **kwargs)
        # Get segments
        words_timestamp = self.__contruct_segments(segments = segments,
                                                   in_milliseconds = in_milliseconds)

        # Define transcription
        transcription = "".join([word.text for word in words_timestamp])

        # Return value
        return TranscriptionResponse(status_code = StatusCode.SUCCESS,
                                     text = transcription,
                                     segments = words_timestamp)




