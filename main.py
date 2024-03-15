import os
from ltb import logger
from ltb.environment import Environment
from ltb.data_processing import DataProcessor
from ltb.indexing import Indexer


class LTB:
    def __init__(self) -> None:
        pass

    def print_results(self, results):
        for doc, score in results:
            logger.info(f"Content: {doc.page_content}, Score: {score}\n\n")

    def main(self):
        env_setup = Environment()
        openai_client = env_setup.setup_environment()

        data_processor = DataProcessor()
        data = data_processor.process_data("data/corpus.json")
        documents = data_processor.split_documents(data)

        embeddings = data_processor.get_embeddings()

        indexer = Indexer()
        db = indexer.index_documents(documents, embeddings)
        query = "How to resolve disputes between landloards and tenants"

        search_metadata = indexer.similarity_search(query, db)

        self.print_results(search_metadata)


if __name__ == "__main__":
    # os.system("python setup.py install")
    obj = LTB()
    obj.main()
