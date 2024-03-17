import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from ltb import logger
from ltb.indexer import Indexer


class Retriever:
    def __init__(self, db, top_k=5, top_n=5, bm25_weight=0.5, faiss_weight=0.5):
        self.db = db
        self.top_k = top_k
        self.top_n = top_n
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight

    def bm25_retriever(self, documents):
        logger.info("Using BM25 Retriever")
        self.bm25_r = BM25Retriever.from_documents(self.documents)
        self.bm25_r.k = self.top_k
        return self.bm25_r

    def faiss_retriever(self):
        logger.info("Using FAISS Retriever")
        self.faiss_r = self.db.as_retriever(search_kwargs={"k": self.top_k})
        return self.faiss_r

    def ensemble_retriever(self):
        logger.info("Using Both BM25 & FAISS Retrievers")
        self.ensemble_r = EnsembleRetriever(
            retrievers=[self.bm25_r, self.faiss_r],
            weights=[self.bm25_weight, self.faiss_weight],
        )
        return self.ensemble_r

    def cohere_reranker(self):
        self.co_rerank = CohereRerank(top_n=self.top_n)
        # combine ensemble retriever and reranker
        self.rerank_retriever = ContextualCompressionRetriever(
            base_retriever=self.ensemble_r,
            base_compressor=self.co_rerank,
        )
        return self.co_rerank
