# 🛸 Introduction:
Eve Agent is Pythonic package supporting both Speech To Text and Text To Speech convenient features, helping you building Voice Chatbot more easily.  
<br />

# 🐍 Python Version:
Requires at least Python 3.10 for running. Highly recommend higher version of Python.

<br />

# 🔑 Feature:
🗣 Text To Speech integrations:
- [CoquiTTS](https://github.com/coqui-ai/TTS)
- [DeepGram](https://deepgram.com/)
- [ElevenLabs](https://elevenlabs.io/)
- [Google Text To Speech](https://github.com/pndurette/gTTS)
- [LMNT](https://www.lmnt.com/)

📢 Speech To Text integrations:
- [AssemblyAI](https://www.assemblyai.com/)
- [DeepGram](https://deepgram.com/)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Groq](https://groq.com/)

<br />

# 📃 To-do List:
- [x] Add basic integrated functions and classes
- [x] Supports async and sync methods (Depends on providers)
- [ ] Streaming features
<br />

# 🤖 Installation:
Clone project inside your main project.
# 🔗 Requirements: 
- For non-Pytorch requirements, install:
```
pip install -r eve_agent/requirements.txt
```
- For Pytorch requirements, install:
```
pip install -r eve_agent/requirements_full.txt
```
<br />

# 🔤 Examples:
```
from eve_agent.speech_recognizer import GroqRecognizer
recognizer = GroqRecognizer(use_async=False)
print(asyncio.run(recognizer.atranscribe(audio_file="test.wav")))
```
