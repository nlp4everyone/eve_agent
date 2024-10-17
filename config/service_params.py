import os
from dotenv import load_dotenv
# Load env
load_dotenv()

# Key
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
