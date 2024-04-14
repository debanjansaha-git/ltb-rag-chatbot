import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# from langchain.retrievers.document_compressors import CohereRerank


class DocumentRetriever:
    """
    A class for document retrieval using embeddings, ensemble methods, and reranking strategies.

    Methods:
    - add_embeddings(text_embedding_pairs): Adds text embeddings to the retriever.
    - retriever(): Returns the retriever based on the added embeddings and top k value.
    - ensemble(): Creates an ensemble retriever based on the faiss retriever.
    - reranker(): Sets up a reranker using an ensemble retriever and a reranking strategy.
    - search(query_text): Searches for relevant documents based on a query text.
    - reranked_texts(query_text): Retrieves reranked texts based on a query text.
    - save_index(): Saves the Faiss index locally.
    """

    def __init__(self, embeds_model, top_k=3):
        self.embeds_model = embeds_model
        self.top_k = top_k

    def add_embeddings(self, text_embedding_pairs):
        self.faiss = FAISS.from_embeddings(text_embedding_pairs, self.embeds_model)

    def retriever(self):
        return self.faiss.as_retriever(search_kwargs={"k": self.top_k})

    def ensemble(self):
        faiss_r = self.retriever()
        self.ensemble_r = EnsembleRetriever(
            retrievers=[faiss_r],
            weights=[1.0],
        )
        return self.ensemble_r

    def reranker(self):
        base_retriever = self.ensemble()
        co_rerank = CohereRerank()
        rerank_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=co_rerank,
        )
        return rerank_retriever

    def search(self, query_text):
        return self.faiss.similarity_search(query_text)

    def reranked_texts(self, query_text):
        rerank_retriever = self.reranker()
        resp = rerank_retriever.get_relevant_documents(query_text)
        resp_text = "\n".join(doc.page_content for doc in resp)
        return resp_text

    def save_index(self):
        self.faiss.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
