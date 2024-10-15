import os
from dotenv import load_dotenv
# Load env
load_dotenv()

# Key
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
