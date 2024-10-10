from speech_components.types import BaseRecognizer
from strenum import StrEnum
from typing import Literal, List, Tuple, Union, Optional
from faster_whisper.transcribe import TranscriptionInfo
from faster_whisper import WhisperModel
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

class FasterWhisperSpeechRecognizer(BaseRecognizer):
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
        :param model_name: Model name extract word and timestamp from audio. Default is (small.en)
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
                                    download_root = download_root)
        # Used batch
        # To use this feature, you must install FasterWhisper from scratch (pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz")
        if use_batch:
            from faster_whisper import BatchedInferencePipeline
            self.__model = BatchedInferencePipeline(model=self.__model)

    def _detect_segments(self,
                        audio_file :str,
                        enable_timestamp :bool = True,
                        **kwarg) -> Tuple:
        """
        Function for getting information( segmentation, info) about audio:
        :param audio_file: Path to the input file (or a file-like object), or the audio waveform.
        :param enable_timestamp: Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
        :return:
        """
        # Check file path
        if not os.path.exists(audio_file): raise FileNotFoundError
        # Return segmentation and info
        segments, info = self.__model.transcribe(audio = audio_file, word_timestamps = enable_timestamp)
        return segments, info

    def get_transcription_info(self,
                               audio_file :str) -> TranscriptionInfo:
        """
        Return information about transcription
        :param audio_file: Path to the input file (or a file-like object), or the audio waveform.
        :return:
        """
        # Get pieces
        _, transcription_info = self._detect_segments(audio_file)
        return transcription_info

    def transcribe(self,
                   audio_file :str) -> str:
        """
        Transcribe audio and then return under string
        :param audio_file: Path to the input file (or a file-like object), or the audio waveform.
        :return:
        """
        # Get pieces
        segments, _ = self._detect_segments(audio_file)

        # Iterate over segments
        list_segments = [segment.text for segment in segments]
        return "".join(list_segments)


    def segment(self,
                audio_file :str,
                **kwargs) -> List[dict]:
        """
        Function with returning a list of dictionary information for words appeared in audio:
        :param audio_file: Path to the input file (or a file-like object), or the audio waveform.
        :return:
        """
        # Get pieces
        segments, _ = self._detect_segments(audio_file)

        segments_info = []
        # Iterate over segments
        for segment in segments:
            for word in segment.words:
                # For each word loop
                temp_data = {"begin_time": word.start, "end_time": word.end, "word": word.word}
                # Append data to list
                segments_info.append(temp_data)
        return segments_info

