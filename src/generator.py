import time
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset


class Generator:
    """
    A class for generating answers to questions using a retrieval-based question answering system.

    Methods:
    - generate(llm, reranker): Generates answers using a retrieval-based QA system.
    - rag_chain(retriever, llm): Defines the RAG-based processing chain for question answering.
    - process(retriever, llm, queries, ground_truth): Processes a list of queries to generate answers and relevant contexts.

    Args:
    - llm: The language model for question answering.
    - reranker: The retriever for ranking relevant documents.
    - retriever: The retriever for retrieving relevant documents.
    - queries (list): A list of queries to generate answers for.
    - ground_truth: The ground truth answers for evaluation.

    Returns:
    - Dataset: A dataset containing questions, answers, contexts, and ground truth for evaluation.
    """

    def __init__(self) -> None:

        self.prompt_template = """

        Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context

        Question: {question}

        Assistant:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

    def generate(self, llm, reranker):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=reranker,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.PROMPT},
        )
        return qa

    def rag_chain(self, retriever, llm):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def process(self, retriever, llm, queries, ground_truth):
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": ground_truth,
        }
        rag = self.rag_chain(retriever, llm)
        for i, query in enumerate(queries):
            data["question"].append(query)
            data["answer"].append(rag.invoke(query))
            data["contexts"].append(
                [doc.page_content for doc in retriever.get_relevant_documents(query)]
            )
            print(f"Processed Query {i}/{len(queries)}...")
            time.sleep(30)

        dataset = Dataset.from_dict(data)
        return dataset
