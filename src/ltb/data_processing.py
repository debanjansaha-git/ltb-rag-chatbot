import json
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from ltb import logger

EMBEDDINGS_MODEL = "text-embedding-ada-002"


class DataProcessor:
    def __init__(
        self, window_size=3, max_key_idx=100, chunk_size=1000, chunk_overlap=500
    ):
        self.window_size = window_size
        self.max_key_idx = max_key_idx
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def get_embeddings(self):
        return OpenAIEmbeddings()

    def process_data(self, json_file):
        # Read data from JSON file and process it
        with open(json_file) as json_file:
            raw_data = json.load(json_file)
        return raw_data

    def process_context(self, data):
        # Extract text and context from data
        print("Key: %s" % data["__ID__"])
        text_str, context_str = "", ""
        if "__Rules__" in data:
            for i in range(len(data["__Rules__"])):
                # Sliding window to incorporate multiple array elements
                window_text = ""
                window_context = ""
                for j in range(i, min(i + self.window_size, len(data["__Rules__"]))):
                    if data["__Rules__"][j]["text"]:
                        window_text += data["__Rules__"][j]["text"] + "\n"
                    if data["__Rules__"][j]["Context"]:
                        window_context += data["__Rules__"][j]["Context"] + "\n"
                text_str += window_text + "\n"
                context_str += window_context + "\n"

        return text_str, context_str

    def split_documents(self, data):
        # Split documents for indexing
        documents = []
        (*data_keys,) = data.keys()
        for key in data_keys[0 : self.max_key_idx]:
            data = data[key]
            content_type = data["__Type__"]
            id = data["__ID__"]
            url = data["__URL__"]
            sub_links = data["__SUB_Link__"]
            if content_type == "link":
                window_text, window_context = self.process_context(data)
                # Include related content referenced by sub-links
                for sub_link in sub_links:
                    if sub_link in data:
                        if data[sub_link]["__Type__"] == "link":
                            sublink_text, sublink_content = self.process_context(
                                data[sub_link]
                            )
                        window_text += "\n" + sublink_text
                        window_context += "\n" + sublink_content
                        documents.append(
                            Document(
                                page_content=window_text,
                                metadata={"Context": window_context},
                            )
                        )

            elif content_type == "pdf":
                for item in data["data"]:
                    forms = data["form"]
                    documents.append(
                        Document(page_content=item, metadata={"Context": forms})
                    )
                context = data["form"]
            else:
                print("Type of Document : ", content_type)

        docs = self.text_splitter.split_documents(documents)
        return docs
