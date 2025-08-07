# In config/config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Google Search API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")