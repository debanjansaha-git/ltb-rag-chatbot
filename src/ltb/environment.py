import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from ltb import logger


class Environment:
    def __init__(self):
        if not load_dotenv(find_dotenv()):
            logger.error("API keys or .env file is missing!")
        else:
            load_dotenv(find_dotenv())
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            logger.info("OpenAI key registered")

    def setup_environment(self):
        return OpenAI(api_key=self.openai_api_key)
