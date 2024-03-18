import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms import Replicate
from transformers import pipeline
from ltb import logger

with open("prompts/rag_prompt_1.txt", "r") as file:
    RAG_PROMPT_TEMPLATE = file.read()


class Generator:
    def __init__(self, reranker) -> None:
        replicate_key = os.getenv("REPLICATE_API_KEY")
        self.model = f"meta-llama/Llama-2-70b-chat-hf:{replicate_key}"
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.str_output_parser = StrOutputParser()
        self.reranker = reranker
        self.combine_context()

    def combine_context(self):
        # parallel execution of context retrieval and question passing
        self.entry_point_and_elevated_retriever = RunnableParallel(
            {"context": self.reranker, "question": RunnablePassthrough()}
        )

    def initiate_llm(self, **kwargs):
        # initialize Llama 2 model with specific parameters
        # pipe = pipeline("text-generation", model=self.model)
        self.llm = Replicate(
            model=self.model,
            model_kwargs=kwargs,
        )

    def rag_chain(self):
        # chain components to form final elevated RAG system using LangChain Expression Language (LCEL)
        self.elevated_rag_chain = (
            self.entry_point_and_elevated_retriever | self.rag_prompt | self.llm
        )

    def query_rag(self, query):
        self.elevated_rag_chain.invoke(query)
