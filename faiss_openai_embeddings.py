import os
import json
import copy
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
import sys
import logging

logs_dir = "logs"
log_path = os.path.join(logs_dir, "running_logs.log")
os.makedirs(logs_dir, exist_ok=True)

# Set the logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

print(
    os.getenv(
        "OPENAI_API_KEY",
        "OpenAI API Key not found - check if the env variables are set correctly",
    )
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_DIMENSIONS = 1536  # specific for "text-embedding-ada-002"

# check requests to OpenAI
# print(client.models.list())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
with open("data/corpus.json") as json_file:
    raw_data = json.load(json_file)
    (*data_keys,) = raw_data.keys()
    # print(raw_data[data_keys[0]])

MAX_KEY_IDX = 100
WINDOW_SIZE = 3


def extend_data(text, context, documents):
    if len(text) != 0 and len(context) != 0:
        documents.extend([{"text": text, "context": context}])
    elif len(text) != 0 and len(context) == 0:
        documents.extend([{"text": text, "context": ""}])
    elif len(text) == 0 and len(context) != 0:
        documents.extend([{"text": "", "context": context}])
    else:
        print("No information present")
    return documents


def process_context(data):
    print("Key: %s" % data["__ID__"])
    text_str, context_str = "", ""
    if "__Rules__" in data:
        print("Found Rules")
        for i in range(len(data["__Rules__"])):
            print("Iterating inside rules")
            # Sliding window to incorporate multiple array elements
            window_text = ""
            window_context = ""
            for j in range(i, min(i + WINDOW_SIZE, len(data["__Rules__"]))):
                if data["__Rules__"][j]["text"]:
                    window_text += data["__Rules__"][j]["text"] + "\n"
                if data["__Rules__"][j]["Context"]:
                    window_context += data["__Rules__"][j]["Context"] + "\n"
            print("Completed processing context and text")
            text_str += window_text + "\n"
            context_str += window_context + "\n"
    print("Going out of processing")
    return text_str, context_str


# Extracting relevant information for indexing
visited = set()
documents = []


def split_documents(data, sub_data, depth=0):
    # print(self.visited)

    # Split documents for indexing
    content_type = sub_data["__Type__"]
    id = sub_data["__ID__"]
    url = sub_data["__URL__"]
    sub_links = sub_data["__SUB_Link__"]

    if url in visited:
        logger.warning(f"Skipping - Already Indexed: {url}.")

    else:
        if content_type == "link":
            logger.info(f"Processing Key={id}")

            window_text, window_context = process_context(sub_data)
            print("Came out of process")
            extended_data = extend_data(window_text, window_context, documents)
            documents.extend(extended_data)
            visited.add(url)
            # Include related content referenced by sub-links
            for sub_link in sub_links:
                if depth == 4:
                    break
                print("Looking for sub-links")
                if sub_link in data:
                    subb_data = data[sub_link]
                    print(
                        f"Found sub-link, recursive call with depth={depth}, id= {sub_link}"
                    )
                    if depth < 4:
                        split_documents(data, subb_data, depth + 1)
                    # documents.extend(sub_docs)
                else:
                    logger.warning(f"Invalid sub-link type for {sub_link}")

        elif content_type == "pdf":
            print("Found pdf file")
            for item in sub_data["data"]:
                forms = sub_data["form"]
                extended_data = extend_data(item, forms, documents)
                documents.extend(extended_data)
        else:
            logger.warning(f"Invalid content type: {content_type}")

    # else:
    #     depth = 0
    # if depth == 0:
    #     visited.clear()  # Reset visited links at the root level

    # docs = self.text_splitter.split_documents(self.documents)
    # return documents


(*data_keys,) = raw_data.keys()
for key in data_keys[0:MAX_KEY_IDX]:
    # print("Data Keys: ", data_keys[0:MAX_KEY_IDX])
    documents = split_documents(raw_data, raw_data[key])
# docs = text_splitter.split_documents(documents)


# embeddings = OpenAIEmbeddings()
# REINDEX = True
# if not os.path.exists(f"index/{os.getenv('FAISS_INDEX_NAME')}") or REINDEX:
#     db = FAISS.from_documents(docs, embeddings)
#     db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
#     print("Indexed the documents in FAISS")
# else:
#     db = FAISS.load_local(f"index/{os.getenv('FAISS_INDEX_NAME')}", embeddings)
#     print("Inferencing from saved index")


# # Similarity Search using FAISS
# query = "How to resolve disputes between landloards and tenants"
# results_with_scores = db.similarity_search_with_score(query)
# for doc, score in results_with_scores:
#     print(f"Query: {query}, \nContent: {doc.page_content}, \nScore: {score}\n\n")
