from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


class LLM:
    """
    A class for loading and using different Language Models (LLMs) for generating text responses.

    Methods:
    - load_llm(): Loads the specified Language Model based on the provided name.

    Args:
    - llm_name (str): The name of the Language Model to load.

    Returns:
    - ChatModel: The loaded Language Model for generating text responses.
    """

    def __init__(self, llm_name: str) -> None:
        self.llm_name = llm_name

    def load_llm(self):
        if self.llm_name == "gpt-4":
            print("LLM: Using GPT-4")
            return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
        elif self.llm_name == "gpt-3.5":
            print("LLM: Using GPT-3.5")
            return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
        elif self.llm_name == "claude-opus":
            print("LLM: Using Claude-3 Opus")
            return ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
        elif self.llm_name == "claude-sonnet":
            print("LLM: Using Claude-3 Sonnet")
            return ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
        elif self.llm_name == "claudev2":
            print("LLM: ClaudeV2")
            return BedrockChat(
                model_id="anthropic.claude-v2",
                model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
                streaming=True,
            )
        elif self.llm_name == "gemini-pro":
            print("LLM: GeminiPro")
            return ChatGoogleGenerativeAI(
                model="gemini-pro", temperature=0.0, top_p=0.85
            )
        elif len(self.llm_name):
            print(f"LLM: Using Ollama: {self.llm_name}")
            return ChatOllama(
                temperature=0,
                base_url=config["ollama_base_url"],
                model=self.llm_name,
                streaming=True,
                # seed=2,
                top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
                top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
                num_ctx=3072,  # Sets the size of the context window used to generate the next token.
            )
        print("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
