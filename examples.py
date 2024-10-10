# Example for mapping from word to viseme
# Sample word
from speech_components.speech_recognizer import FasterWhisperSpeechRecognizer
# sample_word = "Hello"
# mapper = PhonemizerMapper()
# print(f"Mapping visemes: {mapper.word_to_viseme(sample_word)}")
import time
# print("\n")
# Audio path
audio_path = "test.wav"
# Example for converting audio into segmentations
from speech_components import FasterWhisperSpeechRecognizer
recognizer = FasterWhisperSpeechRecognizer(model_name="distil-small.en",num_workers=4, cpu_threads=8, use_batch=False)
beginTime = time.time()
transcription = recognizer.transcribe(audio_file = audio_path)
print(transcription)
endTime = time.time() - beginTime
# print(type(segments))
print(f"{endTime}")
