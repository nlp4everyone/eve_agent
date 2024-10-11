# Example for mapping from word to viseme
# Sample word
# sample_word = "Hello"
# mapper = PhonemizerMapper()
# print(f"Mapping visemes: {mapper.word_to_viseme(sample_word)}")
import time
# print("\n")
# Audio path
audio_path = "test.wav"
# Example for converting audio into segmentations
from speech_recognizer import FasterWhisperRecognizer, AssemblyRecognizer

ground_truth = """
Kamala Debbie Harris, born October 20, 1964, is an American politician and 
attorney who has been the 49th and current vice president of the United States since 2021. 
Serving under President Joe Biden.
"""
recognizer = AssemblyRecognizer()
# recognizer = FasterWhisperRecognizer(model_name="distil-small.en",num_workers=4, cpu_threads=8, use_batch=False)
beginTime = time.time()
transcription = recognizer.transcribe(audio_file = audio_path)
print(transcription)
endTime = time.time() - beginTime
# print(type(segments))
print(f"{endTime}")
