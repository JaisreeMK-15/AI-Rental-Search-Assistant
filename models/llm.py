import os
import sys
from langchain_groq import ChatGroq

# This line ensures Python can find the 'config' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- FIX IS HERE ---
# We now directly import the variable we need from the config file
from config.config import GROQ_API_KEY

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # Check if the API key was successfully imported
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Make sure it's set in your .env file.")

        # Initialize the Groq chat model with the imported API key
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY, # Use the imported variable directly
            model="llama3-70b-8192", # Using a more standard and powerful model
        )
        print("Groq model initialized successfully.")
        return groq_model
    except Exception as e:
        # This will now pass a more informative error message to the Streamlit app
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

