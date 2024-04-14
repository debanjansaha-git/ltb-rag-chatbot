from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    OllamaEmbeddings,
    BedrockEmbeddings,
)
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingModel:
    """
    A class for generating embeddings using different embedding models.

    Methods:
    - generate_embeddings(texts, stage="train"): Generates embeddings for the input texts using the selected model.

    Args:
    - texts (list): A list of texts to generate embeddings for.
    - stage (str, optional): The stage of embedding generation, defaults to "train".

    Returns:
    - list: A list of embeddings generated for the input texts.
    """

    def __init__(self, embedding_model_name: str):
        if embedding_model_name == "openai":
            self.embeddings = OpenAIEmbeddings()
            self.dimension = 1536
            print("Embedding: Using OpenAI")
        elif embedding_model_name == "aws":
            self.embeddings = BedrockEmbeddings()
            self.dimension = 1536
            print("Embedding: Using AWS")
        elif embedding_model_name == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="./embedding_model/embedding-001"
            )
            self.dimension = 768
            print("Embedding: Using Google Generative AI Embeddings")
        else:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2", cache_folder="./embedding_model"
            )
            self.dimension = 384
            print("Embedding: Using SentenceTransformer")

    def generate_embeddings(self, texts, stage="train"):
        all_embeddings = []
        if stage == "train":
            for text in texts:
                embeddings = self.embeddings.embed_query(text)
                all_embeddings.append(embeddings)
        else:
            all_embeddings.append(self.embeddings.embed_query(texts))
        return all_embeddings
