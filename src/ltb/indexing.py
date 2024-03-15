import os
from langchain_community.vectorstores import FAISS


class Indexer:
    def __init__(self, reindex=False):
        self.reindex = reindex

    def index_documents(self, docs, embeddings):
        db = FAISS.from_documents(docs, embeddings)
        if self.reindex:
            db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
            print("Indexed the documents in FAISS")
        else:
            if os.path.exists(f"index/{os.getenv('FAISS_INDEX_NAME')}"):
                db = FAISS.load_local(
                    f"index/{os.getenv('FAISS_INDEX_NAME')}", embeddings
                )
                print("Inferencing from saved index")
            else:
                db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
                print("Indexed the documents in FAISS")
        return db

    def similarity_search(self, query, db):
        return db.similarity_search_with_score(query)
