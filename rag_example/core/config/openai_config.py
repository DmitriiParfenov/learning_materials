import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROK_BASE_URL = os.getenv("GROK_BASE_URL")
