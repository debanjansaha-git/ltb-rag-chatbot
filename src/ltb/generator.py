from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms import Replicate
from transformers import pipeline
from retriever import Retriever
from ltb import logger

with open("prompts/rag_prompt_1.txt", "r") as file:
    RAG_PROMPT_TEMPLATE = file.read()


class Generator:
    def __init__(self) -> None:
        self.model = "meta-llama/Llama-2-70b-chat-hf"
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.str_output_parser = StrOutputParser()
        self.reranker = Retriever.cohere_reranker()

    def combine_context(self):
        # parallel execution of context retrieval and question passing
        self.entry_point_and_elevated_retriever = RunnableParallel(
            {"context": self.rerank_retriever, "question": RunnablePassthrough()}
        )

    def initiate_llm(self):
        # initialize Llama 2 model with specific parameters
        pipe = pipeline("text-generation", model=self.model)
        self.llm = Replicate(
            model=self.llama2_70b,
            model_kwargs={"temperature": 0.5, "top_p": 1, "max_new_tokens": 1000},
        )

    def rag_chain(self):
        # chain components to form final elevated RAG system using LangChain Expression Language (LCEL)
        self.elevated_rag_chain = (
            self.entry_point_and_elevated_retriever | self.rag_prompt | self.llm
        )
