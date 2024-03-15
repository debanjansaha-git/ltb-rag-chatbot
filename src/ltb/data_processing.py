import json
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from ltb import logger

EMBEDDINGS_MODEL = "text-embedding-ada-002"


class DataProcessor:
    def __init__(self, window_size=3, chunk_size=1000, chunk_overlap=500):
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.visited = set()  # To keep track of visited links
        self.documents = []

    def get_embeddings(self):
        return OpenAIEmbeddings()

    def process_data(self, json_file):
        # Read data from JSON file and process it
        with open(json_file) as json_file:
            raw_data = json.load(json_file)
        return raw_data

    def process_context(self, data):
        # Extract text and context from data
        # print("Key: %s" % data["__ID__"])
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

    def split_documents(self, data, sub_data, depth=0):
        if depth < 4:
            # Split documents for indexing
            content_type = sub_data["__Type__"]
            id = sub_data["__ID__"]
            url = sub_data["__URL__"]
            sub_links = sub_data["__SUB_Link__"]

            if url in self.visited:
                logger.warning(f"Skipping - Already Indexed: {url}.")

            else:
                if content_type == "link":
                    logger.info(f"Processing Key={id}")
                    window_text, window_context = self.process_context(sub_data)
                    self.documents.extend(
                        [{"text": window_text, "context": window_context}]
                    )
                    self.visited.add(url)
                    # Include related content referenced by sub-links
                    for sub_link in sub_links:
                        if sub_link in data:
                            subb_data = data[sub_link]
                            sub_docs = self.split_documents(data, subb_data, depth + 1)
                            self.documents.extend(sub_docs)
                        else:
                            logger.warning(f"Invalid sub-link type for {sub_link}")

                elif content_type == "pdf":
                    for item in sub_data["data"]:
                        forms = sub_data["form"]
                        self.documents.append(
                            Document(page_content=item, metadata={"Context": forms})
                        )
                else:
                    logger.warning(f"Invalid content type: {content_type}")

        if depth == 0:
            self.visited.clear()  # Reset visited links at the root level

        # docs = self.text_splitter.split_documents(self.documents)
        return self.documents
