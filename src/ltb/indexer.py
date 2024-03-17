import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from ltb import logger


class Indexer:
    def __init__(self, top_k=5, reindex=False):
        self.reindex = reindex
        self.top_k = top_k
        self.db = None

    def index_documents(self, docs, embeddings):
        self.db = FAISS.from_documents(docs, embeddings)
        if self.reindex:
            self.db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
            logger.info("Indexed the documents in FAISS")
        else:
            if os.path.exists(f"index/{os.getenv('FAISS_INDEX_NAME')}"):
                self.db = FAISS.load_local(
                    f"index/{os.getenv('FAISS_INDEX_NAME')}", embeddings
                )
                logger.info("Inferencing from saved index")
            else:
                self.db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
                logger.info("Indexed the documents in FAISS")
        return self.db

    def similarity_search(self, query):
        return self.db.similarity_search_with_score(query)
