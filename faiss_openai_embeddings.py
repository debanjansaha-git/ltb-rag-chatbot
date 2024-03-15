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

print(os.getenv('OPENAI_API_KEY', 'OpenAI API Key not found - check if the env variables are set correctly'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EMBEDDINGS_MODEL = "text-embedding-ada-002"
INDEX_DIMENSIONS = 1536 # specific for "text-embedding-ada-002"

# check requests to OpenAI
# print(client.models.list())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
with open("data/corpus.json") as json_file:
    raw_data = json.load(json_file)
    *data_keys, = raw_data.keys()
    # print(raw_data[data_keys[0]])

MAX_KEY_IDX = 100
WINDOW_SIZE = 3

def process_context(data):
    print("Key: %s" % data['__ID__'])
    text_str, context_str = "", ""
    if '__Rules__' in data:
        for i in range(len(data['__Rules__'])):
            # Sliding window to incorporate multiple array elements
            window_text = ""
            window_context = ""
            for j in range(i, min(i+WINDOW_SIZE, len(data['__Rules__']))): 
                if data['__Rules__'][j]['text']: 
                    window_text += data['__Rules__'][j]['text'] + "\n"
                if data['__Rules__'][j]['Context']: 
                    window_context += data['__Rules__'][j]['Context'] + "\n"
            text_str += window_text + "\n"
            context_str += window_context + "\n"
            
    return text_str, context_str

# Extracting relevant information for indexing
documents = []
for key in data_keys[0:MAX_KEY_IDX]:
    data = raw_data[key]
    content_type = data['__Type__']
    id = data['__ID__']
    url = data['__URL__']
    sub_links = data['__SUB_Link__']
    if content_type == 'link':
        window_text, window_context = process_context(data)
        # Include related content referenced by sub-links
        for sub_link in sub_links:
            if sub_link in raw_data:
                if raw_data[sub_link]['__Type__'] == 'link':
                    sublink_text, sublink_content = process_context(raw_data[sub_link])
                window_text += "\n" + sublink_text
                window_context += "\n" + sublink_content
                documents.append(Document(page_content=window_text, metadata={'Context': window_context}))
            
    elif content_type == 'pdf':
        for item in data['data']:
            forms = data['form']
            documents.append(Document(page_content=item, metadata={'Context': forms}))
        context = data['form']
    else:
        print("Type of Document : ", content_type)

docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
REINDEX = True
if not os.path.exists(f"index/{os.getenv('FAISS_INDEX_NAME')}") or REINDEX:
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(f"index/{os.getenv('FAISS_INDEX_NAME')}")
    print("Indexed the documents in FAISS")
else:
    db = FAISS.load_local(f"index/{os.getenv('FAISS_INDEX_NAME')}", embeddings)
    print("Inferencing from saved index")


# Similarity Search using FAISS
query = "How to resolve disputes between landloards and tenants"
results_with_scores = db.similarity_search_with_score(query)
for doc, score in results_with_scores:
    print(f"Query: {query}, \nContent: {doc.page_content}, \nScore: {score}\n\n")
