import os
import jq
from time import time
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from ltb import logger


class Indexer:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.db = None

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["context"] = record.get("context")
        return metadata

    def load_jsonl(self, file_path):
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".",
            content_key="text",
            metadata_func=self.metadata_func,
            json_lines=True,
        )
        data = loader.load()
        return data

    def get_embeddings(self, tokenizer):
        if tokenizer == "openai_ada":
            embeds_model = "text-embedding-ada-002"
            index_dim = 1536
            embeddings = OpenAIEmbeddings(model=embeds_model)
        elif tokenizer == "cohere":
            embeds_model = "embed-english-light-v3.0"
            embeddings = CohereEmbeddings(model=embeds_model)
        return embeddings

    def index_documents(self, docs, embeddings, reindex=False):
        self.db = FAISS.from_documents(docs, embeddings)
        if reindex:
            self.db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
            logger.info("Indexed the documents in FAISS")
        else:
            if os.path.exists(f"index/{os.getenv('FAISS_INDEX_NAME')}"):
                self.db = FAISS.load_local(
                    f"index/{os.getenv('FAISS_INDEX_NAME')}",
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Inferencing from saved index")
            else:
                logger.info("FAISS index not found in path. Reindexing...")
                self.db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
        return self.db

    def similarity_search(self, query):
        return self.db.similarity_search_with_score(query)
