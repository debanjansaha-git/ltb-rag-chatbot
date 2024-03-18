import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePick, RunnablePassthrough, RunnableParallel
from langchain_community.llms import Replicate
from transformers import pipeline
from ltb import logger

with open("prompts/rag_prompt_2.txt", "r") as file:
    RAG_PROMPT_TEMPLATE = file.read()


class Generator:
    def __init__(self, reranker) -> None:
        self.rag_prompt = ChatPromptTemplate.from_template(template=RAG_PROMPT_TEMPLATE)
        self.reranker = reranker

    def format_docs(self, docs):
        print(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    def initiate_llm(self, model, **kwargs):
        # initialize Llama 2 model with specific parameters
        if model == "llama2":
            logger.info("Using LlaMA 2 for inferencing")
            self.llm = Replicate(
                model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
                model_kwargs=kwargs,
            )
        elif model == "gpt3.5":
            logger.info("Using GPT 3.5 for inferencing")
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=300)

    def rag_chain(self, query):
        print("Reranker Input: \n", self.reranker)
        # chain components to form final elevated RAG system using LangChain Expression Language (LCEL)
        self.elevated_rag_chain = (
            # RunnablePassthrough.assign(
            #     context=RunnablePick("context") | self.format_docs
            # )
            RunnableParallel(
                {"context": self.reranker, "question": RunnablePassthrough()}
            )
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        response = self.elevated_rag_chain.invoke(
            {"context": self.reranker, "question": query}
        )
        return response
