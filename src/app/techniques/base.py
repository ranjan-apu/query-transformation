from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

class BaseRAGTechniques(ABC):

    def __init__(self,db , embedding, llm = None):
        self.db = db
        self.embedding = embedding
        self.llm = llm or ChatOpenAI(
            model="qwen/qwen3-32b:free",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        pass

    def generate(self, query: str, documents: List[Document]) -> str:
        """Generates an answer to the question based on the given context."""
        context = "\n".join([doc.page_content for doc in documents])
        prompt = f"""
        You are an AI helpful assistant. Answer the question based on the given context.
        context: {context}
        question: {query}
        """
        response = self.llm.invoke(prompt)
        return response.content

    def query(self, query: str) -> str:
        documents = self.retrieve(query)
        return self.generate(query, documents)