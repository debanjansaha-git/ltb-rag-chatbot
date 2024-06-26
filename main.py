import os
import numpy as np
import pandas as pd
from data_loader import DataCleaner, DataLoader
from data_transform import DataTransformer
from embeddings import EmbeddingModel
from retriever import DocumentRetriever
from generator import Generator
from evaluation import Evaluator
from llm import LLM
from utils import load_json, encode_context_columns
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cleaner = DataCleaner()
loader = DataLoader()
data = loader.load_data("data/corpus.json")
transformer = DataTransformer(cleaner)
cleaned_data = transformer.aggregate_context(data, "data/cleaned_data.json")
df, hash_to_context = encode_context_columns(cleaned_data)
embeds = EmbeddingModel("sentence_transformers")
embeddings_file_path = "data/embeddings.npy"
df_filtered = df.dropna(subset=["text"]).copy()
df_filtered = df_filtered[df_filtered["text"].str.strip() != ""]
texts = df_filtered["text"].tolist()
if os.path.exists(embeddings_file_path):
    print("Embeddings file already exists.")
    embeddings_np = np.load(embeddings_file_path)
else:
    embeddings = embeds.generate_embeddings(texts, stage="train")
    embeddings_np = np.vstack(embeddings).astype(np.float32)
    os.makedirs("data", exist_ok=True)
    np.save(embeddings_file_path, embeddings_np)
    print("Embeddings file created and saved.")

text_embedding_pairs = zip(texts, embeddings_np)
faiss = DocumentRetriever(embeds.embeddings, top_k=10)
faiss.add_embeddings(text_embedding_pairs)
faiss.save_index()

eval_data = loader.load_data("data/eval_data.json")
queries = []
ground_truth = []
for item in eval_data["landlord"]:
    queries.append(item["instruction"])
    ground_truth.append(item["output"])


# query_embedding = embeds.generate_embeddings(query_text, stage="inference")
# response = faiss.reranked_texts(query_text)
reranker = faiss.reranker()

llm_name = "gpt-3.5"
llm_model = LLM(llm_name)
llm = llm_model.load_llm()
genai = Generator()
qa_chain = genai.rag_chain(retriever=reranker, llm=llm)
dataset = genai.process(reranker, llm, queries, ground_truth)

eval = Evaluator()
eval_results = eval.evaluate_ragas(dataset=dataset)
eval.plot_evaluation(eval_results, llm_name, filepath="plots/rag_evaluation.png")

# result = qa_chain({"query": query_text})
# answer = result["result"]
# response = f""" {answer}\n\ngenerated by {llm_name}"""
# print(response)
