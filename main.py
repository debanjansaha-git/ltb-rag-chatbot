import os
from ltb import logger
from ltb.environment import Environment
from ltb.data_preprocessing import DataProcessor
from ltb.indexing import Indexer


class LTB:
    def __init__(self) -> None:
        self.max_key_idx = 3

    # def print_results(self, results):
    #     for doc, score in results:
    #         logger.info(f"Content: {doc.page_content}, Score: {score}\n\n")

    # def write_file(self, documents):
    #     with open(f"outputs/data.txt", "w") as f:
    #         f.write(documents)

    def main(self):
        env_setup = Environment()
        openai_client = env_setup.setup_environment()

        data_processor = DataProcessor()
        data, data_keys = data_processor.read_data("data/corpus.json")
        # (*data_keys,) = data.keys()
        for key in data_keys:
            print("Key: ", key)
            data_processor.split_documents(data[key])
        data_processor.write_json("data/preprocessed_data.json")

        # indexer = Indexer()
        # db = indexer.index_documents(documents, embeddings)
        # query = "How to resolve disputes between landloards and tenants"

        # search_metadata = indexer.similarity_search(query, db)

        # self.print_results(search_metadata)


if __name__ == "__main__":
    # os.system("python setup.py install")
    obj = LTB()
    obj.main()
