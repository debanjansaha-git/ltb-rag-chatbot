import os
from time import time
from ltb import logger
from ltb.environment import Environment
from ltb.data_preprocessing import DataProcessor
from ltb.indexer import Indexer
from ltb.retriever import Retriever
from ltb.generator import Generator


class LTB:
    def __init__(self) -> None:
        self.max_key_idx = 3
        self.top_k = 5

    def print_results(self, results):
        for doc, score in results:
            logger.info(f"Content: {doc.page_content}, Score: {score}\n\n")

    # def write_file(self, documents):
    #     with open(f"outputs/data.txt", "w") as f:
    #         f.write(documents)

    def main(self, query, model_params):
        ## Register environment vars
        env_setup = Environment()
        openai_client = env_setup.setup_environment()

        # ## Pre-process RAW data
        # data_processor = DataProcessor()
        # data, data_keys = data_processor.read_data("data/corpus.json")
        # # (*data_keys,) = data.keys()
        # for key in data_keys:
        #     print("Key: ", key)
        #     data_processor.split_documents(data[key])
        # data_processor.write_json("data/preprocessed_data.jsonl")

        ## Index Documents in FAISS
        indexer = Indexer(self.top_k)
        jsonl = indexer.load_jsonl("data/preprocessed_data.jsonl")
        short_data = jsonl[4:6]
        embeds = indexer.get_embeddings(tokenizer="cohere")
        db = indexer.index_documents(short_data, embeds, reindex=False)

        ## Check results
        # search_metadata = indexer.similarity_search(query)
        # self.print_results(search_metadata)

        ## Call Retriever/s
        ret = Retriever(db)
        ret_bm25 = ret.bm25_retriever(short_data)
        ret_doc = ret.faiss_retriever()
        ret_esb = ret.ensemble_retriever()
        ret_cor = ret.cohere_reranker()
        resp = ret_cor.get_relevant_documents(query)
        ## Checked working uptil here...

        ## Call Generate
        genai = Generator(ret_cor)
        genai.initiate_llm(**model_params)
        response = genai.rag_chain(query)
        print(resp)


if __name__ == "__main__":
    # os.system("python setup.py install")
    obj = LTB()
    query = "How to resolve disputes between landloards and tenants"
    model_params = {"temperature": 0.1, "top_p": 1, "max_new_tokens": 300}
    obj.main(query, model_params)
