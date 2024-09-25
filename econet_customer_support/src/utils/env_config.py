from dotenv import load_dotenv
import os

def load_env_vars():
    """Load environment variables from .env file."""
    load_dotenv()

    # Ensure required environment variables are set
    required_vars = ['ELASTICSEARCH_URL', 'GROQ_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")